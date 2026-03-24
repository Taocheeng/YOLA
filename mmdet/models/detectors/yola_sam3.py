# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .base import BaseDetector
from .yola_utils import IIBlock


@MODELS.register_module()
class YOLAWithSAM3(BaseDetector):
    """YOLA detector with SAM3 bbox predictor.

    - Keep YOLA illumination-invariant module.
    - Replace YOLO/TOOD bbox head with SAM3 adapter head.
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        sam3_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        loss_consistency: ConfigType = dict(
            type='SmoothL1Loss', loss_weight=1.0, reduction='sum'
        ),
        kernel_nums: int = 8,
        kernel_size: int = 3,
        Gtheta: List[float] = [0.6, 0.8],
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        sam3_head = sam3_head.copy()
        sam3_head.update(train_cfg=train_cfg)
        sam3_head.update(test_cfg=test_cfg)
        self.sam3_head = MODELS.build(sam3_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.iim = IIBlock(kernel_nums, kernel_size, Gtheta)
        self.loss_consistency = MODELS.build(loss_consistency)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        out, feats = self.iim(batch_inputs)
        out = self.backbone(out)
        fpn_out = self.neck(out)
        return fpn_out, feats

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        x, feats = self.extract_feat(batch_inputs)
        feat_ii, feat_ii_gma = feats
        losses = self.sam3_head.loss(x, batch_data_samples)
        losses.update({'loss_consist': self.loss_consistency(feat_ii, feat_ii_gma)})
        return losses

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        x, _ = self.extract_feat(batch_inputs)
        results_list = self.sam3_head.predict(x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[List[Tensor]]:
        x, _ = self.extract_feat(batch_inputs)
        results = self.sam3_head.forward(x)
        return results
