"""Per-sample realism scoring against the real-image feature manifold.

The classifier reward answers "does this image look like class c to a frozen
classifier". It says nothing about whether the image still looks like a
scintigraphy scan, so maximizing it alone lets the generator drift off the real
manifold -- which is exactly what happened in the v2 RL run (overall FID rose
from 28.9 to 35.3 while the reward kept climbing).

This module supplies the missing term: a bounded, per-sample realism score in
the *same* InceptionV3 pool-feature space used to report FID, so the penalty and
the headline distribution metric cannot disagree about what "realistic" means.

Score definition
----------------
For a candidate x with intended class c:

    d(x) = mean distance to the k nearest real features of class c
    realism(x) = -(d(x) - mu_c) / sigma_c          (clamped)

where mu_c / sigma_c are the leave-one-out kNN distances *among real images of
class c*. A typical real image therefore scores ~0, an off-manifold sample
scores negative, and the term is scale-free so it can be added to `margin`
without retuning per class. Features are L2-normalized before distances are
taken: raw Inception pool magnitudes vary enough between classes that the
un-normalized distances are dominated by overall image intensity.

A kNN distance is used rather than a Mahalanobis distance to the class Gaussian
because the per-class real manifold is multimodal (uptake patterns differ a lot
within, say, Nodular Goiter) and a single Gaussian would reward the class mean.

Usage
-----
    # once, before RL / selection (cached to disk):
    python -m utils.realism_utils --pickle_path data/train_data_128.pkl \\
        --split_manifest data/nested_split.json --split_role fm \\
        --output checkpoints/realism/bank_fm.pt

    # then:
    scorer = RealismScorer("checkpoints/realism/bank_fm.pt", device)
    realism = scorer.score(images_01, target_idx)   # higher = more realistic
"""

import argparse
import hashlib
import json
import os
import pickle
from typing import Dict, Optional, Sequence, Tuple

import torch

from utils.fid_utils import (
    _get_inception_feature_extractor,
    inception_features_chunked,
)

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BANK_DIR = os.path.join(_THIS_DIR, "checkpoints", "realism")
DEFAULT_K = 5


def _as_image_01(value) -> torch.Tensor:
    """Pickle image -> [1, H, W] float tensor in [0, 1] (matches the classifier path)."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected image of shape [H,W] or [C,H,W], got {tuple(tensor.shape)}")
    return tensor.clamp(0.0, 1.0)


def _sample_id(sample: dict) -> str:
    for key in ("name", "folder_id", "folder"):
        value = sample.get(key)
        if value is not None and str(value) != "":
            return str(value)
    raise KeyError("Sample has no name/folder_id/folder key; cannot apply a nested split filter.")


def load_manifest_ids(split_manifest: Optional[str], split_role: Optional[str]) -> Optional[Sequence[str]]:
    if not split_manifest:
        return None
    if not split_role:
        raise ValueError("--split_role is required when --split_manifest is given")
    with open(split_manifest) as f:
        payload = json.load(f)
    key = f"{split_role}_ids"
    if key not in payload:
        raise KeyError(f"Manifest {split_manifest} has no '{key}'")
    return list(payload[key])


def load_real_samples(
    pickle_path: str,
    split: str = "train",
    allowed_ids: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, object]]:
    """Real images in [0,1] plus class indices, using the same class ordering as
    `load_and_prepare_data` (sorted unique labels) so indices line up with the
    one-hot conditioning the RL rollouts use."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    samples = list(data[split])

    if allowed_ids is not None:
        allowed = set(allowed_ids)
        before = len(samples)
        samples = [s for s in samples if _sample_id(s) in allowed]
        print(f"[realism] pickle split '{split}': {before} -> {len(samples)} images")
    if not samples:
        raise ValueError(f"No samples left in split '{split}' of {pickle_path}")

    class_list = [s["class"] for s in samples]
    all_classes = sorted(set(class_list))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for i, c in enumerate(all_classes)}

    images = torch.stack([_as_image_01(s["image"]) for s in samples], dim=0)
    class_idx = torch.tensor([class_to_idx[c] for c in class_list], dtype=torch.long)
    return images, class_idx, idx_to_class


@torch.no_grad()
def extract_features(
    images_01: torch.Tensor,
    device: torch.device,
    inception_weights_path: Optional[str] = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """L2-normalized InceptionV3 pool features [N, 2048] on CPU."""
    model = _get_inception_feature_extractor(device, weights_path=inception_weights_path)
    feats = inception_features_chunked(model, images_01, device, batch_size=batch_size)
    return torch.nn.functional.normalize(feats.float(), dim=1)


def _loo_knn_stats(features: torch.Tensor, k: int) -> Tuple[float, float]:
    """Leave-one-out mean kNN distance statistics within one real class."""
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(float("inf"))
    k_eff = max(1, min(k, features.shape[0] - 1))
    per_sample = distances.topk(k_eff, largest=False).values.mean(dim=1)
    std = float(per_sample.std(unbiased=False))
    return float(per_sample.mean()), max(std, 1e-6)


def build_feature_bank(
    pickle_path: str,
    split: str = "train",
    allowed_ids: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    k: int = DEFAULT_K,
    inception_weights_path: Optional[str] = None,
    batch_size: int = 32,
) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, class_idx, idx_to_class = load_real_samples(pickle_path, split, allowed_ids)
    print(f"[realism] extracting Inception features for {images.shape[0]} real images ...")
    features = extract_features(images, device, inception_weights_path, batch_size)

    per_class: Dict[int, Dict] = {}
    for cls in sorted(class_idx.unique().tolist()):
        feats_c = features[class_idx == cls]
        mean, std = _loo_knn_stats(feats_c, k)
        per_class[int(cls)] = {"features": feats_c, "knn_mean": mean, "knn_std": std}
        print(
            f"[realism] class {cls} ({idx_to_class[cls]}): n={feats_c.shape[0]} "
            f"knn_mean={mean:.4f} knn_std={std:.4f}"
        )

    return {
        "per_class": per_class,
        "class_map": idx_to_class,
        "k": int(k),
        "feature": "inception_pool_l2",
        "source": {"pickle_path": pickle_path, "split": split, "n_ids": len(allowed_ids or [])},
    }


def default_bank_path(pickle_path: str, split: str, split_role: Optional[str]) -> str:
    tag = hashlib.sha1(f"{pickle_path}|{split}|{split_role}".encode()).hexdigest()[:8]
    return os.path.join(DEFAULT_BANK_DIR, f"bank_{split_role or 'all'}_{tag}.pt")


class RealismScorer:
    """Scores candidates by proximity to the real per-class feature manifold."""

    def __init__(
        self,
        bank_path: str,
        device: torch.device,
        k: Optional[int] = None,
        clamp: float = 5.0,
        inception_weights_path: Optional[str] = None,
        feature_batch_size: int = 32,
    ):
        if not os.path.isfile(bank_path):
            raise FileNotFoundError(
                f"Realism feature bank not found at {bank_path!r}. Build it with "
                "`python -m utils.realism_utils --output <path> ...`."
            )
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        self.bank_path = bank_path
        self.device = device
        self.k = int(k or bank.get("k", DEFAULT_K))
        self.clamp = float(clamp)
        self.class_map = bank.get("class_map", {})
        self.inception_weights_path = inception_weights_path
        self.feature_batch_size = int(feature_batch_size)
        self._per_class = {
            int(cls): {
                "features": payload["features"].to(device),
                "knn_mean": float(payload["knn_mean"]),
                "knn_std": float(payload["knn_std"]),
            }
            for cls, payload in bank["per_class"].items()
        }
        print(
            f"[realism] bank {bank_path} | {len(self._per_class)} classes | k={self.k} "
            f"| feature={bank.get('feature')}"
        )

    @torch.no_grad()
    def embed(self, images_01: torch.Tensor) -> torch.Tensor:
        """L2-normalized Inception pool features on the scorer's device.

        Exposed separately so callers that also need coverage/diversity in this
        feature space (e.g. farthest-point pool selection) pay for one forward
        pass instead of two.
        """
        model = _get_inception_feature_extractor(self.device, weights_path=self.inception_weights_path)
        feats = inception_features_chunked(
            model, images_01, self.device, batch_size=self.feature_batch_size
        )
        return torch.nn.functional.normalize(feats.float(), dim=1).to(self.device)

    @torch.no_grad()
    def score_from_features(self, feats: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        target_idx = target_idx.to(self.device)
        scores = torch.zeros(feats.shape[0], device=self.device)
        for cls in target_idx.unique().tolist():
            payload = self._per_class.get(int(cls))
            if payload is None:
                continue
            rows = target_idx == cls
            distances = torch.cdist(feats[rows], payload["features"])
            k_eff = max(1, min(self.k, distances.shape[1]))
            per_sample = distances.topk(k_eff, largest=False).values.mean(dim=1)
            scores[rows] = -(per_sample - payload["knn_mean"]) / payload["knn_std"]
        return scores.clamp(-self.clamp, self.clamp)

    @torch.no_grad()
    def score(self, images_01: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """images_01: [B,C,H,W] in [0,1]; target_idx: [B] intended class indices.

        Returns [B] realism scores: ~0 for manifold-typical, negative for
        off-manifold candidates.
        """
        return self.score_from_features(self.embed(images_01), target_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real-image realism feature bank.")
    parser.add_argument("--pickle_path", default=os.path.join(_THIS_DIR, "data", "train_data_128.pkl"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split_role", default=None, help="reward | fm | downstream")
    parser.add_argument("--output", default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output or default_bank_path(args.pickle_path, args.split, args.split_role)
    if os.path.isfile(output) and not args.overwrite:
        print(f"Already present: {output} (use --overwrite to rebuild)")
        return

    allowed_ids = load_manifest_ids(args.split_manifest, args.split_role)
    bank = build_feature_bank(
        pickle_path=args.pickle_path,
        split=args.split,
        allowed_ids=allowed_ids,
        k=args.k,
        batch_size=args.batch_size,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(bank, output)
    print(f"Realism bank saved to {output}")


if __name__ == "__main__":
    main()
