# RL checkpoint selection (no external leakage)

The multi-center **external** test set must stay untouched until **final**
downstream evaluation. Do **not** pick `step*.ckpt` / `last.ckpt` using external
accuracy or macro-F1.

## Correct protocol

1. Sample a small probe pool from each RL checkpoint (e.g. 200 images/class, 10 Euler steps).
2. Train a short mixed-pool classifier on the **development** train split
   (optional nested `downstream` role if you use a nested reward/FM split).
3. Rank checkpoints by **development-validation** macro-F1
   (tie-break: validation accuracy, then mean reward confidence).
4. Copy the winner to your “best” path and only then sample the full 1000/class pool.
5. Run the full scenario grid; evaluate the external cohort **once** at the end.

In the full journal codebase this is `probe_rl_checkpoints.py`, which calls
`train_classifier.py --skip_external` so the holdout pickle is never loaded
during probing.

## What not to do

- Ranking RL steps by external probe accuracy / F1 (information leakage into generator choice).
- Treating the synthetic 90%/10% carve-out as an “external” set — that split is
  internal early-stopping / validation matching only.
