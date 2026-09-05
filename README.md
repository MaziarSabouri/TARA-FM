# Task-Aware RL of Flow Matching for Thyroid Scintigraphy

Companion code for the IEEE TMI manuscript:

**Task-Aware Reinforcement Learning of Flow Matching: Aligning Thyroid Scintigraphy Augmentation With Downstream Classification**

This repository contains the minimal pipeline used in the paper:

1. Preprocess a thyroid scintigraphy dataset into training pickles  
2. Train base **mask + class** conditioned Flow Matching (FM)  
3. Fine-tune with three RL rewards: `prob`, `focal`, `utility`  
4. Sample synthetic image pools from FM / RL checkpoints  

Patient data and trained weights are **not** included. Contact the authors for access.

## Install

```bash
conda create -n rlfm python=3.10 -y
conda activate rlfm
pip install -r requirements.txt
```

A CUDA-capable GPU is required for FM/RL training and sampling.

## Preprocess

```bash
EXCEL_PATH=/path/to/metadata.xlsx \
DATA_ROOT=/path/to/Image_and_Segmentation \
bash scripts/preprocess.sh
```

See [`data/README.md`](data/README.md) for the pickle schema. Configs expect `./data/train_data_128.pkl`.

## Train base FM

```bash
bash scripts/train_fm.sh
```

Checkpoints are written under `checkpoints/fm/` (see [`checkpoints/README.md`](checkpoints/README.md)).

## Train RL (three rewards)

Provide a frozen reward classifier checkpoint and the base FM weights, then choose a reward:

```bash
export CLASSIFIER_REWARD_CKPT=checkpoints/reward/reward_resnet18.pth
export FM_CKPT=checkpoints/fm/<your_fm>.ckpt

REWARD_TYPE=prob    bash scripts/train_rl.sh
REWARD_TYPE=focal   bash scripts/train_rl.sh
REWARD_TYPE=utility bash scripts/train_rl.sh
```

Each run writes to `checkpoints/rl/<REWARD_TYPE>/`. Optional realism / coverage terms are documented in `rewards.py` (env vars); the paper’s main comparisons use the three `REWARD_TYPE` settings above.

## Sample synthetics

```bash
MODEL_PATH=checkpoints/fm/<your_fm>.ckpt \
OUT=data/samples_fm.pkl \
bash scripts/sample.sh

MODEL_PATH=checkpoints/rl/prob/<your_rl>.ckpt \
OUT=data/samples_rl_prob.pkl \
bash scripts/sample.sh
```

Override `PER_CLASS`, `STEPS`, `BATCH`, and `SEED` as needed.

## Citation

```bibtex
@article{Sabouri2026TMI,
  title   = {Task-Aware Reinforcement Learning of Flow Matching: Aligning Thyroid Scintigraphy Augmentation With Downstream Classification},
  author  = {Sabouri, Maziar and Yazdani, Milad and others},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026},
  note    = {Under review / to appear}
}
```

## License / data

Code is provided for research reproducibility. Dataset and model weights require institutional agreements; please contact the corresponding authors.
