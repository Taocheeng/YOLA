#!/usr/bin/env python3
"""Export key detection metrics to a json file only (no image saving).

Required in output:
- mAP50
- overall recall

Usage:
python scripts/export_metrics.py \
  --log work_dirs/xxx/20260324_123456/vis_data/scalars.json \
  --out work_dirs/xxx/metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='Path to scalars.json or json log.')
    parser.add_argument('--out', required=True, help='Output metric json path.')
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_map50(record: Dict[str, Any]) -> Optional[float]:
    candidates = [
        'pascal_voc/mAP',  # VOC usually reports mAP@0.5
        'bbox_mAP_50',
        'coco/bbox_mAP_50',
        'mAP_50',
        'AP50',
    ]
    for key in candidates:
        if key in record:
            return _safe_float(record[key])
    return None


def _extract_recall(record: Dict[str, Any]) -> Optional[float]:
    candidates = [
        'recall',
        'bbox_recall',
        'AR@100',
        'coco/bbox_AR@100',
        'pascal_voc/recall',
    ]
    for key in candidates:
        if key in record:
            return _safe_float(record[key])

    # Fallback: average over recall-at-k keys if present.
    recall_values = []
    for key, value in record.items():
        key_l = key.lower()
        if 'recall' in key_l or key_l.startswith('ar@'):
            val = _safe_float(value)
            if val is not None:
                recall_values.append(val)
    if recall_values:
        return float(sum(recall_values) / len(recall_values))
    return None


def load_last_eval_record(path: Path):
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        map50 = _extract_map50(obj)
        recall = _extract_recall(obj)
        if map50 is not None or recall is not None:
            records.append(obj)

    if not records:
        raise RuntimeError(f'No evaluation metrics found in {path}')
    return records[-1]


def main():
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)

    result = load_last_eval_record(log_path)
    summary = {
        'mAP50': _extract_map50(result),
        'recall': _extract_recall(result),
        'raw': result,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] metrics written to: {out_path}')


if __name__ == '__main__':
    main()
