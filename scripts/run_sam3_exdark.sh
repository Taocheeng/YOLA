#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for YOLA + SAM3 on ExDark
# It does not save enhanced images; only model ckpt/logs/metrics json.

CONFIG=${1:-configs/sam3/sam3_yola_exdark.py}
WORK_DIR=${2:-work_dirs/sam3_yola_exdark}
CKPT=${3:-"${WORK_DIR}/latest.pth"}

# 1) Preflight check
python scripts/verify_sam3_setup.py \
  --sam3-repo /home/taocheng/sam3/sam3/sam3 \
  --sam3-module sam3.build \
  --sam3-builder build_sam3_detector \
  --label-file /home/taocheng/YOLA_Project/data/exdarkv3/labels.txt \
  --num-classes 12

# 2) Train
python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}"

# 3) Test (bbox metric only; no image dumping)
python tools/test.py "${CONFIG}" "${CKPT}" \
  --cfg-options default_hooks.visualization=None

# 4) Export metrics JSON
LATEST_SCALARS=$(ls -1t "${WORK_DIR}"/*/vis_data/scalars.json | head -n 1)
python scripts/export_metrics.py --log "${LATEST_SCALARS}" --out "${WORK_DIR}/metrics_summary.json"

echo "[DONE] Metrics exported to ${WORK_DIR}/metrics_summary.json"
