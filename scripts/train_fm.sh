#!/usr/bin/env bash
# Train base mask+class conditioned Flow Matching.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/fm_mask_class.yaml}"
echo "Training FM with config=$CONFIG"
python trainer.py --config_path "$CONFIG" "$@"
