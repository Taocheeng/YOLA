#!/usr/bin/env python3
"""Export MMDetection eval metrics to a json file only (no image saving).

Usage:
python scripts/export_metrics.py \
  --log work_dirs/xxx/20260324_123456/vis_data/scalars.json \
  --out work_dirs/xxx/metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='Path to scalars.json or json log.')
    parser.add_argument('--out', required=True, help='Output metric json path.')
    return parser.parse_args()


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
        if any(k in obj for k in ('coco/bbox_mAP', 'pascal_voc/mAP', 'mAP')):
            records.append(obj)

    if not records:
        raise RuntimeError(f'No evaluation metrics found in {path}')
    return records[-1]


def main():
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)

    result = load_last_eval_record(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] metrics written to: {out_path}')


if __name__ == '__main__':
    main()
