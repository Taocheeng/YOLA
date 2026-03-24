# YOLA + SAM3（替换 YOLO/TOOD 检测头）落地指南

> 目标：保留 YOLA 的低照增强/不变特征建模，把 `extract_feat` 之后的检测分支由 `bbox_head`（YOLOv3/TOOD）替换为你已复刻的 `SAM3` 框预测分支，并输出可复现实验量化指标。

---

## 0. 先明确当前代码的“替换点”

当前仓库中，检测调用集中在 `mmdet/models/detectors/yola.py`：

- 训练时在 `loss()` 中调用：`self.bbox_head.loss(x, batch_data_samples)`
- 推理时在 `predict()` 中调用：`self.bbox_head.predict(x, batch_data_samples, rescale=rescale)`
- `extract_feat()` 的返回是 `fpn_out, feats`，其中 `fpn_out` 会送入检测头

因此你要做的是：**把 `bbox_head` 的接口替换成 `sam3_head/sam3_model` 接口，或增加一层适配器，让 SAM3 接口“伪装成”bbox_head 的 loss/predict 形式。**

---

## 1. 第一步：实现 SAM3 适配器（建议）

### 1.1 新建适配模块

建议新建：

- `mmdet/models/detectors/sam3_adapter.py`

推荐接口：

```python
class SAM3Adapter(nn.Module):
    def __init__(self, sam3_model, train_cfg=None, test_cfg=None):
        ...

    def loss(self, feats, batch_data_samples):
        """返回 dict，至少包含 loss_det / loss_cls / loss_bbox 等键"""
        ...

    def predict(self, feats, batch_data_samples, rescale=True):
        """返回 results_list，元素是 InstanceData: bboxes/scores/labels"""
        ...
```

### 1.2 适配输入

YOLA 当前 `extract_feat()` 输出是 FPN 多尺度特征（tuple/list tensor）。你要确认 SAM3 复刻版接受：

- A) 图像 token（需图像输入）
- B) 多尺度特征（可直接用 FPN 输出）

若是 A：在 adapter 内加入 `feat2token`（1x1 conv + flatten + pos embed）。  
若是 B：直接喂 `fpn_out`，通常最稳。

### 1.3 适配输出

MMDetection 期望 `predict()` 最终能写入：

- `pred_instances.bboxes` (N,4)
- `pred_instances.scores` (N,)
- `pred_instances.labels` (N,)

如果 SAM3 原生输出是 mask + box quality，需要在 adapter 内：

1. mask -> bbox（`x1,y1,x2,y2`）
2. quality/objectness -> score
3. class logits -> label（若 SAM3 是 class-agnostic，需要加一个分类层）

---

## 2. 第二步：新增检测器类（YOLAWithSAM3）

新增：

- `mmdet/models/detectors/yola_sam3.py`

可复用 `YOLABaseDetector` 结构，关键是把 `bbox_head` 替换成 `sam3_head`：

```python
@MODELS.register_module()
class YOLAWithSAM3(YOLABaseDetector):
    def __init__(self, backbone, neck, sam3_head, ...):
        ...
        self.sam3_head = MODELS.build(sam3_head)

    def loss(self, batch_inputs, batch_data_samples):
        x, feats = self.extract_feat(batch_inputs)
        feat_ii, feat_ii_gma = feats
        losses = self.sam3_head.loss(x, batch_data_samples)
        losses.update({'loss_consist': self.loss_consistency(feat_ii, feat_ii_gma)})
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x, _ = self.extract_feat(batch_inputs)
        results_list = self.sam3_head.predict(x, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)
```

> 注意：如果你不想动 `YOLABaseDetector`，直接新建一个从 `BaseDetector` 继承的类也可以。

---

## 3. 第三步：完成模块注册

确保以下 `__init__.py` 做注册：

- `mmdet/models/detectors/__init__.py`
  - 增加 `from .yola_sam3 import YOLAWithSAM3`
  - 加入 `__all__`

如果 SAM3 适配器放在 `dense_heads`，则对应更新 `mmdet/models/dense_heads/__init__.py`。

---

## 4. 第四步：配置文件改造（核心）

以 `configs/tood/tood_yola_exdark.py` 为底，复制一份：

- `configs/sam3/sam3_yola_exdark.py`

关键改动：

1. `model.type = 'YOLAWithSAM3'`
2. 删除 `bbox_head`，改为 `sam3_head=dict(...)`
3. `num_classes` 对齐数据集（ExDark=12，DarkFace=1）
4. `test_cfg` 中阈值策略要与 SAM3 输出分数分布匹配（score_thr/nms）

建议起始配置：

- `score_thr=0.01`（先放低防漏检）
- `nms_iou_threshold=0.6`
- `max_per_img=300`

---

## 5. 第五步：训练流程（先跑通，再提效）

### 5.1 冷启动策略

- 冻结 SAM3 backbone 前几层，仅训练 adapter + YOLA + detection decoder，5~10 epoch
- 再全量解冻训练到 24 epoch（对齐你当前 1x schedule）

### 5.2 损失推荐

初版建议：

- `loss_cls`: Focal/QFL（二选一）
- `loss_bbox`: GIoU/L1
- `loss_consist`: 继续沿用 YOLA 的一致性损失（建议权重先保持 0.01）

总损失示例：

\[
\mathcal{L}=\lambda_{cls}L_{cls}+\lambda_{box}L_{box}+\lambda_{cons}L_{cons}
\]

---

## 6. 第六步：推理与后处理

你的 SAM3 若输出多个候选框/点提示结果：

1. 先按 `score` 过滤
2. 再做 class-wise NMS
3. 统一坐标到原图（rescale）

确保最终格式与 MMDet evaluator 对齐，否则评估时会出现空框或尺度错位。

---

## 7. 第七步：必须拿到的量化指标（建议最小闭环）

> 你提到“需要拿到相应量化指标”，建议至少包含 **精度 + 速度 + 稳定性** 三类。

### 7.1 精度指标（检测主指标）

- VOC/ExDark 常用：`mAP@0.5`
- COCO 风格：`mAP@[0.5:0.95]`、`AP50`、`AP75`、`APS/APM/APL`
- 召回：`AR@100`

### 7.2 速度指标（部署必备）

- 单卡 batch=1 FPS
- 单图 latency（ms）：P50 / P90 / P99
- 显存峰值（MiB）

### 7.3 稳定性指标（低照任务建议）

- 不同照度子集下 mAP（dark / normal / bright）
- 不同随机种子（至少 3 个）均值±方差

---

## 8. 第八步：实验矩阵（你可以直接照抄执行）

### 8.1 对比组

1. `YOLA + YOLOv3(head)`（现基线）
2. `YOLA + TOOD(head)`（现强基线）
3. `YOLA + SAM3(head)`（你的新方案）

### 8.2 统一训练设置

- 输入尺度：608x608
- epoch：24
- optimizer/lr 与基线一致
- 同一数据划分、同一评估脚本

### 8.3 报表模板

| Method | ExDark mAP50 | ExDark AP50:95 | DarkFace mAP50 | FPS | Latency(ms) |
|---|---:|---:|---:|---:|---:|
| YOLA+YOLOv3 | - | - | - | - | - |
| YOLA+TOOD | - | - | - | - | - |
| YOLA+SAM3 | - | - | - | - | - |

---

## 9. 第九步：命令级落地（MMDet）

### 9.1 训练

```bash
python tools/train.py configs/sam3/sam3_yola_exdark.py
```

### 9.2 测试 + 指标导出

```bash
python tools/test.py configs/sam3/sam3_yola_exdark.py \
  work_dirs/sam3_yola_exdark/latest.pth \
  --cfg-options test_evaluator.metric=mAP
```

如需 COCO 指标，把 evaluator 改成 `CocoMetric` 并开启 `metric=['bbox']`。

### 9.3 速度测试（示例）

```bash
python tools/analysis_tools/benchmark.py \
  configs/sam3/sam3_yola_exdark.py \
  work_dirs/sam3_yola_exdark/latest.pth \
  --repeat-num 200
```

---

## 10. 常见坑位（你大概率会遇到）

1. **SAM3 输出不是 bbox 格式**：需要统一到 `InstanceData`。
2. **类别数不一致**：ExDark 12 类，DarkFace 通常 1 类。
3. **输入归一化冲突**：YOLA 的 `ReflectedConvolution` 依赖 [0,1] 输入，别提前做 ImageNet 归一化后直接喂进去。
4. **在线/离线评估波动**：仓库 README 已提示 BN 可能带来偏差，建议固定 eval 模式并多 seed 汇报。

---

## 11. 你可以直接执行的“最短路径”

1. 写 `SAM3Adapter`，提供 `loss/predict`。
2. 写 `YOLAWithSAM3`，在 `loss/predict` 调 adapter。
3. 注册模块并新建 `configs/sam3/sam3_yola_exdark.py`。
4. 先跑 1 epoch 冒烟（只看是否能输出合法 bbox）。
5. 跑满 24 epoch，导出 mAP/FPS/latency。
6. 与 TOOD/YOLOv3 基线同表对比。

如果你愿意，我下一步可以直接给你一版**可粘贴的 `yola_sam3.py` 与 `sam3_adapter.py` 代码骨架**（按 MMDet 3.x 接口），你只需要把你复刻的 SAM3 前向函数名替进去即可。
