#!/usr/bin/env bash
# Sample a synthetic pool from an FM or RL checkpoint.
#
# Prefer an RL checkpoint chosen by development-validation probing (not last.ckpt
# and not by external metrics). Example:
#   MODEL_PATH=checkpoints/fm/....ckpt bash scripts/sample.sh
#   MODEL_PATH=checkpoints/rl/prob/best.ckpt OUT=data/samples_rl_prob.pkl bash scripts/sample.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/fm_mask_class.yaml}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Lightning .ckpt}"
OUT="${OUT:-data/samples_synth.pkl}"
PER_CLASS="${PER_CLASS:-1000}"
STEPS="${STEPS:-10}"
BATCH="${BATCH:-32}"
SEED="${SEED:-0}"
GEN_SPLIT="${GEN_SPLIT:-train}"

python inferer.py \
  --config_path "$CONFIG" \
  --model_path "$MODEL_PATH" \
  --num_samples_per_class "$PER_CLASS" \
  --num_inference_steps "$STEPS" \
  --batch_size "$BATCH" \
  --seed "$SEED" \
  --gen_split "$GEN_SPLIT" \
  --output_path "$OUT" \
  "$@"
