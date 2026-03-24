from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from mmdet.registry import MODELS


@MODELS.register_module()
class SAM3Adapter(nn.Module):
    """Adapter that makes an external SAM3 model look like an MMDet bbox head.

    Notes:
    1) SAM3 repo and YOLA repo can live in different directories.
    2) Prompt labels are loaded from user-provided label files.
    3) No enhanced image is saved; only metric artifacts should be exported.
    """

    def __init__(
        self,
        sam3_repo_path: str = '/home/taocheng/sam3/sam3/sam3',
        sam3_module: str = 'sam3.build',
        sam3_builder: str = 'build_sam3_detector',
        sam3_kwargs: Optional[dict] = None,
        prompt_label_file: Optional[str] = None,
        prompt_template: str = 'detect {label}',
        num_classes: int = 12,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.sam3_repo_path = sam3_repo_path
        self.sam3_module = sam3_module
        self.sam3_builder = sam3_builder
        self.sam3_kwargs = sam3_kwargs or {}
        self.prompt_label_file = prompt_label_file
        self.prompt_template = prompt_template
        self.num_classes = num_classes
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        self._sam3_model = None
        self.labels = self._load_labels(prompt_label_file)
        self._validate_labels()

    def _lazy_build_sam3(self):
        if self._sam3_model is not None:
            return self._sam3_model

        if self.sam3_repo_path and self.sam3_repo_path not in sys.path:
            sys.path.insert(0, self.sam3_repo_path)

        try:
            module = importlib.import_module(self.sam3_module)
        except Exception as e:
            raise ImportError(
                f'Failed to import SAM3 module={self.sam3_module} from '
                f'path={self.sam3_repo_path}. Please replace placeholder path '
                'with your actual SAM3 repo path.'
            ) from e

        builder = getattr(module, self.sam3_builder, None)
        if builder is None:
            raise AttributeError(
                f'Cannot find builder `{self.sam3_builder}` in module '
                f'`{self.sam3_module}`. Please align with your SAM3 replica API.'
            )
        self._sam3_model = builder(**self.sam3_kwargs)
        return self._sam3_model

    def _load_labels(self, label_file: Optional[str]) -> List[str]:
        if not label_file:
            return []

        path = Path(label_file)
        if not path.exists():
            raise FileNotFoundError(
                f'prompt_label_file={label_file} not found. '
                'Please provide the ExDark/DarkFace label file path.'
            )

        if path.suffix.lower() == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                labels = [str(v) for _, v in sorted(data.items(), key=lambda x: int(x[0]))]
            elif isinstance(data, list):
                labels = [str(v) for v in data]
            else:
                raise ValueError('JSON label file must be list or dict.')
            return labels

        labels = []
        for raw in path.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # support formats: "0 person" | "person"
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                labels.append(' '.join(parts[1:]))
            else:
                labels.append(line)
        return labels

    def _validate_labels(self) -> None:
        if not self.labels:
            return
        if len(self.labels) < self.num_classes:
            raise ValueError(
                f'label count={len(self.labels)} < num_classes={self.num_classes}. '
                'Please ensure each class has a prompt label.'
            )

    def _prompts(self) -> List[str]:
        labels = self.labels if self.labels else [str(i) for i in range(self.num_classes)]
        return [self.prompt_template.format(label=lb) for lb in labels[: self.num_classes]]

    def _call_first_available(self, model, candidates: Sequence[str], **kwargs):
        for fn_name in candidates:
            fn = getattr(model, fn_name, None)
            if fn is not None:
                return fn(**kwargs)
        raise AttributeError(
            f'SAM3 model does not provide any of {candidates}. '
            'Please adapt `SAM3Adapter._call_first_available` to your API.'
        )

    def loss(self, feats: Tuple[torch.Tensor], batch_data_samples):
        model = self._lazy_build_sam3()
        prompts = self._prompts()  # ensure each label has a prompt

        outputs = self._call_first_available(
            model,
            candidates=('loss', 'forward_train', 'compute_loss'),
            feats=feats,
            batch_data_samples=batch_data_samples,
            prompts=prompts,
        )
        if not isinstance(outputs, dict):
            raise TypeError('SAM3 loss output must be a dict of losses.')
        return outputs

    def predict(self, feats, batch_data_samples, rescale: bool = True):
        model = self._lazy_build_sam3()
        prompts = self._prompts()  # ensure each label has a prompt

        pred_list = self._call_first_available(
            model,
            candidates=('predict', 'forward_test', 'predict_bboxes'),
            feats=feats,
            batch_data_samples=batch_data_samples,
            prompts=prompts,
            rescale=rescale,
            score_thr=self.test_cfg.get('score_thr', 0.01),
            nms_cfg=self.test_cfg.get('nms', dict(type='nms', iou_threshold=0.6)),
            max_per_img=self.test_cfg.get('max_per_img', 300),
        )

        results_list = []
        for pred in pred_list:
            ins = InstanceData()
            ins.bboxes = pred['bboxes']
            ins.scores = pred['scores']
            ins.labels = pred['labels']
            results_list.append(ins)
        return results_list
