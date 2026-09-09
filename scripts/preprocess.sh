#!/usr/bin/env bash
# Build training pickles from raw Excel + DICOM/NIfTI folders.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXCEL_PATH="${EXCEL_PATH:?Set EXCEL_PATH to the metadata workbook}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to Image_and_Segmentation}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

python prepare_dataset.py \
  --excel_path "$EXCEL_PATH" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --image_size "$IMAGE_SIZE" \
  --skip_external \
  "$@"
