#!/usr/bin/env python3
"""Quick preflight check before running YOLA+SAM3 training/testing.

Checks:
1) SAM3 repo path exists and is importable.
2) SAM3 module+builder exists.
3) Prompt label file exists and class count is enough.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sam3-repo', required=True)
    p.add_argument('--sam3-module', default='sam3.build')
    p.add_argument('--sam3-builder', default='build_sam3_detector')
    p.add_argument('--label-file', required=True)
    p.add_argument('--num-classes', type=int, required=True)
    return p.parse_args()


def load_labels(path: Path):
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
        if isinstance(data, list):
            return [str(v) for v in data]
        raise ValueError('JSON label file must be list/dict')

    labels = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        labels.append(' '.join(parts[1:]) if len(parts) > 1 and parts[0].isdigit() else line)
    return labels


def main():
    args = parse_args()

    sam3_repo = Path(args.sam3_repo)
    if not sam3_repo.exists():
        raise FileNotFoundError(f'SAM3 repo not found: {sam3_repo}')

    if str(sam3_repo) not in sys.path:
        sys.path.insert(0, str(sam3_repo))

    module = importlib.import_module(args.sam3_module)
    if not hasattr(module, args.sam3_builder):
        raise AttributeError(f'Builder `{args.sam3_builder}` not found in `{args.sam3_module}`')

    label_path = Path(args.label_file)
    if not label_path.exists():
        raise FileNotFoundError(f'Label file not found: {label_path}')

    labels = load_labels(label_path)
    if len(labels) < args.num_classes:
        raise ValueError(
            f'label count({len(labels)}) < num_classes({args.num_classes}), please add missing labels'
        )

    print('[OK] SAM3 repo/module/builder checked')
    print(f'[OK] label coverage checked: {len(labels)} labels for num_classes={args.num_classes}')


if __name__ == '__main__':
    main()
