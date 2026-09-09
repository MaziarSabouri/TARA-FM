#!/usr/bin/env bash
# Fine-tune FM with a task reward. Requires REWARD_TYPE and CLASSIFIER_REWARD_CKPT.
#
#   REWARD_TYPE=prob|focal|utility \
#   CLASSIFIER_REWARD_CKPT=checkpoints/reward/your_reward.pth \
#   FM_CKPT=checkpoints/fm/....ckpt \
#   bash scripts/train_rl.sh
#
# After training, select among step*.ckpt / last.ckpt using a short downstream
# probe on the *development* validation split only. Do not use the external
# holdout for RL checkpoint selection; reserve it for final evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/rl_mask_class.yaml}"
REWARD_TYPE="${REWARD_TYPE:?Set REWARD_TYPE to prob, focal, or utility}"
CLASSIFIER_REWARD_CKPT="${CLASSIFIER_REWARD_CKPT:?Set CLASSIFIER_REWARD_CKPT to a reward .pth (or comma-separated ensemble)}"
FM_CKPT="${FM_CKPT:?Set FM_CKPT to the base FM Lightning .ckpt}"

case "$REWARD_TYPE" in
  prob|focal|utility) ;;
  *)
    echo "REWARD_TYPE must be prob, focal, or utility (got: $REWARD_TYPE)" >&2
    exit 1
    ;;
esac

export REWARD_TYPE
export CLASSIFIER_REWARD_CKPT
export REWARD_FOCAL_GAMMA="${REWARD_FOCAL_GAMMA:-2.0}"
export REWARD_UTILITY_WRONG_PENALTY="${REWARD_UTILITY_WRONG_PENALTY:-1.0}"

IFS=',' read -r -a REWARD_CKPTS <<< "$CLASSIFIER_REWARD_CKPT"
if [[ "${#REWARD_CKPTS[@]}" -ne 3 ]]; then
  echo "The paper protocol requires exactly three comma-separated reward checkpoints." >&2
  exit 1
fi

# Optional: isolate runs by reward name under checkpoints/rl/<type>
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/rl/${REWARD_TYPE}}"

echo "RL fine-tune: REWARD_TYPE=$REWARD_TYPE"
echo "  CLASSIFIER_REWARD_CKPT=$CLASSIFIER_REWARD_CKPT"
echo "  FM_CKPT=$FM_CKPT"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "  gamma=$REWARD_FOCAL_GAMMA utility_eta=$REWARD_UTILITY_WRONG_PENALTY"

python rl_trainier.py \
  --config_path "$CONFIG" \
  --model_path "$FM_CKPT" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
