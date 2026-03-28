from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from PIL import Image

from mmdet.registry import MODELS


@MODELS.register_module(name='SAM3DetectorWrapper')
@MODELS.register_module(name='SAM3Adapter')
class SAM3DetectorWrapper(nn.Module):
    """Wrap SAM3 image model to mimic a detector head used by YOLA.

    This wrapper runs text-prompt detection for each class and aggregates all
    class predictions into bbox detections. Masks are intentionally not used.
    """

    def __init__(
        self,
        checkpoint_path: str,
        class_names: Sequence[str],
        device: Union[str, torch.device] = 'cuda',
        resolution: int = 1008,
        confidence_threshold: float = 0.3,
        score_thr: float = 0.01,
        nms_iou_threshold: float = 0.6,
        max_per_img: int = 300,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError('`class_names` must contain at least one category name.')

        # Import lazily so the rest of YOLA can still be imported without SAM3.
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = torch.device(device)
        self.class_names = list(class_names)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        self.score_thr = float(self.test_cfg.get('score_thr', score_thr))
        nms_cfg = self.test_cfg.get('nms', {})
        self.nms_iou_threshold = float(nms_cfg.get('iou_threshold', nms_iou_threshold))
        self.max_per_img = int(self.test_cfg.get('max_per_img', max_per_img))

        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            eval_mode=True,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(
            model=self.model,
            resolution=resolution,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """Convert tensor image [C,H,W] to RGB PIL expected by SAM3 processor."""
        if image.dim() != 3:
            raise ValueError(f'Expected CHW image tensor, but got shape={tuple(image.shape)}')

        tensor = image.detach().float().cpu()
        # Heuristic: convert to [0, 1] first if range is [0, 255] or others.
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)

        np_img = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _run_single_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SAM3 on one image and return (boxes, scores, labels)."""
        pil_image = self._tensor_to_pil(image)
        state = self.processor.set_image(pil_image)

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for class_id, class_name in enumerate(self.class_names):
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(class_name, state)

            boxes = output.get('boxes')
            scores = output.get('scores')
            if boxes is None or scores is None:
                continue

            boxes = torch.as_tensor(boxes, device=self.device, dtype=torch.float32)
            scores = torch.as_tensor(scores, device=self.device, dtype=torch.float32)
            if boxes.numel() == 0 or scores.numel() == 0:
                continue

            keep = scores >= self.score_thr
            if keep.sum() == 0:
                continue

            boxes = boxes[keep]
            scores = scores[keep]
            labels = torch.full(
                (scores.shape[0],), class_id, dtype=torch.long, device=self.device
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_boxes:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            empty_scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
            empty_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
            return empty_boxes, empty_scores, empty_labels

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # class-aware NMS
        try:
            from torchvision.ops import batched_nms

            keep = batched_nms(boxes, scores, labels, self.nms_iou_threshold)
            keep = keep[: self.max_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        except Exception:
            # Fallback: keep top-k without NMS if torchvision op is unavailable.
            topk = min(self.max_per_img, scores.shape[0])
            keep = torch.argsort(scores, descending=True)[:topk]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        return boxes, scores, labels

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Return YOLO-style tensor outputs per image: [x1, y1, x2, y2, score, cls]."""
        if images.dim() != 4:
            raise ValueError(f'Expected BCHW tensor, but got shape={tuple(images.shape)}')

        outputs: List[torch.Tensor] = []
        for i in range(images.shape[0]):
            boxes, scores, labels = self._run_single_image(images[i])
            if boxes.numel() == 0:
                pred = torch.zeros((0, 6), dtype=torch.float32, device=self.device)
            else:
                pred = torch.cat(
                    [boxes, scores.unsqueeze(1), labels.float().unsqueeze(1)], dim=1
                )
            outputs.append(pred)
        return outputs

    def predict(self, images: torch.Tensor, batch_data_samples=None, rescale: bool = True):
        """MMDetection-compatible predict output: list[InstanceData]."""
        _ = batch_data_samples, rescale
        yolo_style = self.forward(images)

        results_list = []
        for pred in yolo_style:
            ins = InstanceData()
            ins.bboxes = pred[:, :4]
            ins.scores = pred[:, 4]
            ins.labels = pred[:, 5].long()
            results_list.append(ins)
        return results_list

    def loss(self, images: torch.Tensor, batch_data_samples=None):
        """SAM3 prompt detector is inference-only here; return a zero det loss."""
        _ = images, batch_data_samples
        return {'loss_det': torch.tensor(0.0, device=self.device, requires_grad=True)}
