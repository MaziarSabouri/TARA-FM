"""Reward hook for RL fine-tuning (see rl_trainier.py).

A frozen classifier scores every generated candidate and the reward expresses
how well that candidate represents its intended disease class. The conference
version used a single reward (the softmax probability of the true class); the
journal version adds alternatives so the reward design itself can be ablated,
and allows the reward model to be a different architecture from the downstream
classifier (which is what rules out reward gaming).

Configuration is by environment variable so no RL training code has to change:

    CLASSIFIER_REWARD_CKPT   path to the frozen reward checkpoint, or a
                             comma-separated list to average an ensemble
                             (default: checkpoints/reward/reward_resnet18_prob.pth)
    REWARD_TYPE              prob | margin | logprob | entropy_reg | focal
    REWARD_ENTROPY_WEIGHT    lambda for entropy_reg (default 0.5)
    REWARD_FOCAL_GAMMA       gamma for the focal reward (default 2.0)
    REWARD_REALISM_WEIGHT    lambda for the realism term (default 0.0 = off)
    REWARD_REALISM_BANK      path to the realism feature bank (utils/realism_utils.py)
    REWARD_REALISM_K         neighbours for the realism kNN distance (default: bank value)
    REWARD_COVERAGE_WEIGHT   lambda for in-group novelty (default 0.0 = off)
    REWARD_COVERAGE_REALISM_FLOOR   gate: reject rollouts below this realism (default -1.0)
    REWARD_COVERAGE_REQUIRE_CORRECT gate: reject rollouts the reward model misreads (default 1)
    REWARD_COVERAGE_GATE_PENALTY    how far a gated rollout is pushed down (default 10.0)

Ensembling the reward model
---------------------------
A single ResNet18 trained on the ~640-patient reward pool is a noisy judge: in
the v2 run it was only ~57% accurate on the rollouts it was scoring, and nearly
blind to the Normal class. Averaging the softmax of a K-fold ensemble (same
backbone, disjoint validation folds) lowers that variance without touching the
patient-disjointness of the nested split.

The realism term
----------------
`prob`/`margin` reward "looks like class c to a classifier", which a generator
can satisfy while drifting off the scintigraphy manifold entirely -- the v2 RL
run raised overall FID from 28.9 to 35.3 while its reward improved. With
REWARD_REALISM_WEIGHT > 0 the reward becomes

    R = base_reward + lambda * realism

where `realism` is the per-sample manifold score from utils/realism_utils.py
(0 for a manifold-typical sample, negative off-manifold), so drifting away from
real scans is paid for in the same currency as class confidence.

Why coverage, and why it is scored per group
--------------------------------------------
The selection_v2 gate (8 criteria, 5 seeds, 1000/class each) found that no
confidence-based criterion beats a random subset downstream, and that `margin` --
the reward RL v2 maximized -- is the only one that consistently hurts. The one
criterion with a large effect was coverage: farthest-point selection halved FID
(23.97 -> 11.27), and gating it on reward-correctness (`coverage_conf`) gave the
best accuracy and the tightest seed variance of any pool. Coverage is a set
property, so it is measured here within each rollout group, where the candidates
share a mask and class and "different from its siblings" is well defined.

Reward variants
---------------
prob         p_true. Simple and bounded in [0,1], but saturates once the
             classifier is already confident, so late training signal is weak.
margin       p_true - max_{j != true} p_j. Bounded in [-1,1] and keeps rewarding
             progress after p_true saturates, because it also penalizes the
             strongest competing class. This directly targets the confusions
             that limit external accuracy (e.g. Normal vs. Diffuse Goiter).
logprob      log p_true, clamped for stability. Unbounded below, so it punishes
             off-target samples much harder than `prob`.
entropy_reg  p_true - lambda * H(p)/log(C). Rewards confident *and* correct
             samples, discouraging the ambiguous images that `prob` tolerates.
focal        (1 - p_true)^gamma * p_true. Emphasizes candidates that are still
             hard for the classifier, a curriculum-style signal.
utility      focal, but only for candidates the reward model classifies
             correctly; misclassified rollouts get a fixed penalty. Intended
             for a *downstream* reward (O+FM+CA), i.e. "would this image help
             the classifier we actually report" rather than "looks like class c".
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.classifier_utils import load_classifier_bundle
from utils.realism_utils import RealismScorer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT = os.path.join(_THIS_DIR, "checkpoints", "reward", "reward_resnet18_prob.pth")

REWARD_TYPES = ("prob", "margin", "logprob", "entropy_reg", "focal", "zero", "utility")


class ClassifierReward:
    """Wraps a frozen classifier checkpoint (any backbone) for reward scoring."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        bundle = load_classifier_bundle(checkpoint_path, device)
        self.checkpoint_path = checkpoint_path
        self.model = bundle["model"]
        self.class_names = bundle["class_names"]
        self.backbone = bundle["backbone"]
        self.input_size = bundle["input_size"]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.mean = torch.tensor(bundle["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(bundle["std"], device=device).view(1, 3, 1, 1)

    def to(self, device: torch.device) -> "ClassifierReward":
        if device != self.device:
            self.model.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            self.device = device
        return self

    @torch.no_grad()
    def predict_probs(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] in [0, 1] -> softmax probabilities [B, num_classes]."""
        images = images.to(device=self.device, dtype=torch.float32)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            images = images[:, :1].repeat(1, 3, 1, 1)
        if images.shape[-1] != self.input_size:
            images = F.interpolate(
                images, size=(self.input_size, self.input_size), mode="bilinear",
                align_corners=False, antialias=images.shape[-1] > self.input_size,
            )
        images = (images - self.mean) / self.std
        return F.softmax(self.model(images).float(), dim=1)


class EnsembleClassifierReward:
    """Averages the softmax of several frozen reward checkpoints."""

    def __init__(self, checkpoint_paths: List[str], device: torch.device):
        self.checkpoint_path = ",".join(checkpoint_paths)
        self.members = [ClassifierReward(path, device=device) for path in checkpoint_paths]
        names = self.members[0].class_names
        for member in self.members[1:]:
            if member.class_names != names:
                raise ValueError(
                    "Reward ensemble members disagree on class ordering: "
                    f"{names} vs {member.class_names}"
                )
        self.class_names = names
        self.backbone = "+".join(sorted({m.backbone for m in self.members}))
        self.input_size = self.members[0].input_size
        self.device = device

    def to(self, device: torch.device) -> "EnsembleClassifierReward":
        for member in self.members:
            member.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def predict_probs(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.predict_probs(images) for m in self.members], dim=0).mean(dim=0)


_REWARD_MODEL = None
_REALISM_SCORER: Optional[RealismScorer] = None


def _reward_checkpoint_paths() -> List[str]:
    raw = os.environ.get("CLASSIFIER_REWARD_CKPT", _DEFAULT_CHECKPOINT)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _get_reward_model(device: torch.device):
    global _REWARD_MODEL
    paths = _reward_checkpoint_paths()
    key = ",".join(paths)
    if _REWARD_MODEL is None or _REWARD_MODEL.checkpoint_path != key:
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"ClassifierReward checkpoint not found at {path!r}. Train it first with "
                    "`python train_classifier.py --experiment reward --scenario O --checkpoint_out ...`, "
                    "or point CLASSIFIER_REWARD_CKPT at an existing checkpoint."
                )
        _REWARD_MODEL = (
            ClassifierReward(paths[0], device=device)
            if len(paths) == 1
            else EnsembleClassifierReward(paths, device=device)
        )
        print(
            f"[reward] {_REWARD_MODEL.backbone} @ {_REWARD_MODEL.input_size}px "
            f"| members={len(paths)} | type={os.environ.get('REWARD_TYPE', 'prob')} "
            f"| realism_weight={_realism_weight()}"
        )
    return _REWARD_MODEL.to(device)


def _realism_weight() -> float:
    return float(os.environ.get("REWARD_REALISM_WEIGHT", 0.0) or 0.0)


def _get_realism_scorer(device: torch.device) -> RealismScorer:
    global _REALISM_SCORER
    bank_path = os.environ.get("REWARD_REALISM_BANK")
    if not bank_path:
        raise ValueError(
            "REWARD_REALISM_WEIGHT > 0 requires REWARD_REALISM_BANK to point at a feature bank "
            "built with `python -m utils.realism_utils`."
        )
    if _REALISM_SCORER is None or _REALISM_SCORER.bank_path != bank_path:
        k_env = os.environ.get("REWARD_REALISM_K")
        _REALISM_SCORER = RealismScorer(
            bank_path, device=device, k=int(k_env) if k_env else None
        )
    return _REALISM_SCORER


def reward_from_probs(probs: torch.Tensor, target_idx: torch.Tensor, reward_type: str) -> torch.Tensor:
    """Map class probabilities to a scalar reward per candidate."""
    p_true = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    if reward_type == "prob":
        return p_true

    if reward_type == "margin":
        masked = probs.clone()
        masked.scatter_(1, target_idx.unsqueeze(1), -1.0)
        return p_true - masked.max(dim=1).values

    if reward_type == "logprob":
        return torch.log(p_true.clamp_min(1e-6))

    if reward_type == "entropy_reg":
        lam = float(os.environ.get("REWARD_ENTROPY_WEIGHT", 0.5))
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        normalized_entropy = entropy / torch.log(torch.tensor(float(probs.shape[1]), device=probs.device))
        return p_true - lam * normalized_entropy

    if reward_type == "focal":
        gamma = float(os.environ.get("REWARD_FOCAL_GAMMA", 2.0))
        return ((1.0 - p_true) ** gamma) * p_true

    if reward_type == "utility":
        # Hard-but-correct: useful *to this classifier*. Easy prototypes
        # (p_true ~ 1) get ~0; wrong-class samples are dumped.
        gamma = float(os.environ.get("REWARD_FOCAL_GAMMA", 2.0))
        penalty = float(os.environ.get("REWARD_UTILITY_WRONG_PENALTY", 1.0))
        focal = ((1.0 - p_true) ** gamma) * p_true
        correct = probs.argmax(dim=1) == target_idx
        return torch.where(correct, focal, torch.full_like(focal, -penalty))

    if reward_type == "zero":
        # No class-confidence term at all. Used when the reward is meant to be
        # carried entirely by the coverage/realism terms, with the classifier
        # acting only as a correctness gate: the selection_v2 gate showed that
        # ranking by confidence is at best neutral and at worst harmful.
        return torch.zeros_like(p_true)

    raise ValueError(f"Unknown REWARD_TYPE={reward_type!r}. Choose from {REWARD_TYPES}.")


def _coverage_weight() -> float:
    return float(os.environ.get("REWARD_COVERAGE_WEIGHT", 0.0) or 0.0)


@torch.no_grad()
def group_novelty(feats: torch.Tensor, group_size: int) -> torch.Tensor:
    """Mean feature distance from each rollout to the others in its group.

    Coverage is a property of a *set*, so it cannot be scored one sample at a
    time. Within a GRPO-style group the candidates share a conditioning mask and
    class, which makes "how different is this rollout from its siblings" a
    well-posed per-candidate quantity: rewarding it pushes the generator to
    spread out where it currently produces near-duplicates.

    `feats` is [B*G, D] with groups in consecutive blocks (repeat_interleave).
    Returned scores are z-scored within each group, so they are comparable
    across groups and directly usable by the best-of-N selector.
    """
    dim = feats.shape[1]
    grouped = feats.view(-1, group_size, dim)
    distances = torch.cdist(grouped, grouped)
    novelty = distances.sum(dim=2) / max(1, group_size - 1)
    mean = novelty.mean(dim=1, keepdim=True)
    std = novelty.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return ((novelty - mean) / std).reshape(-1)


@torch.no_grad()
def compute_rewards(
    images: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    batch: Optional[Dict[str, torch.Tensor]] = None,
    group_size: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward hook for RL fine-tuning.

    Inputs:
    - images: generated candidates in display range [0, 1], shape [B*G, C, H, W].
    - masks: repeated conditioning masks when mask conditioning is enabled (unused here).
    - classes: repeated one-hot class tensor [B*G, num_classes], if class conditioning
      is enabled. Used as the reward's target class when available.
    - batch: repeated dataloader batch; a fallback source of "classes" when the model
      itself isn't class-conditioned (mask-only configs still carry ground-truth classes).

    Returns one scalar reward per candidate (higher = better) plus logging metrics.
    """
    target_onehot = classes if classes is not None else (batch.get("classes") if batch else None)
    if target_onehot is None:
        raise ValueError(
            "ClassifierReward requires class labels, but neither `classes` nor `batch['classes']` "
            "was provided. Make sure every sample in your dataset pickle has a 'class' field."
        )
    target_idx = target_onehot.argmax(dim=1).to(images.device)

    reward_model = _get_reward_model(images.device)
    probs = reward_model.predict_probs(images)

    reward_type = os.environ.get("REWARD_TYPE", "prob")
    base_rewards = reward_from_probs(probs, target_idx, reward_type)

    p_true = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    accuracy = (probs.argmax(dim=1) == target_idx).float().mean()

    metrics = {
        "classifier_prob": p_true.detach(),
        "classifier_accuracy": accuracy.detach(),
        "base_value": base_rewards.detach(),
    }

    rewards = base_rewards
    lam_realism = _realism_weight()
    lam_coverage = _coverage_weight()

    realism = None
    if lam_realism > 0.0 or lam_coverage > 0.0:
        # One Inception pass serves both terms.
        scorer = _get_realism_scorer(images.device)
        feats = scorer.embed(images)
        realism = scorer.score_from_features(feats, target_idx).to(
            device=rewards.device, dtype=rewards.dtype
        )
        metrics["realism"] = realism.detach()

    if lam_realism > 0.0:
        rewards = rewards + lam_realism * realism

    if lam_coverage > 0.0:
        if group_size < 2:
            raise ValueError(
                "REWARD_COVERAGE_WEIGHT > 0 needs group_size >= 2: in-group novelty is "
                "undefined for a single rollout per condition."
            )
        novelty = group_novelty(feats, group_size).to(device=rewards.device, dtype=rewards.dtype)
        rewards = rewards + lam_coverage * novelty
        metrics["coverage_novelty"] = novelty.detach()

        # Gates, mirroring the `coverage_conf` selection rule that gave the best
        # downstream accuracy: unconstrained novelty is maximized by the tails of
        # the distribution, which is where the artifacts are. A gated candidate is
        # pushed far enough below its siblings that best-of-N never accepts it
        # unless the whole group is gated.
        floor = float(os.environ.get("REWARD_COVERAGE_REALISM_FLOOR", -1.0))
        penalty = float(os.environ.get("REWARD_COVERAGE_GATE_PENALTY", 10.0))
        pass_realism = realism >= floor
        pass_gate = pass_realism.clone()
        if os.environ.get("REWARD_COVERAGE_REQUIRE_CORRECT", "1") == "1":
            pass_gate = pass_gate & (probs.argmax(dim=1) == target_idx)

        # Per-group escape hatch: the single best candidate of a group is never
        # gated. An absolute bar alone is only as well calibrated as the feature
        # bank -- in a smoke test with a small bank it rejected 82% of rollouts,
        # which leaves most groups uniformly penalized and their best-of-N choice
        # decided by novelty alone. Exempting each group's leader keeps every
        # group informative while the absolute bar still catches global drift.
        grouped = pass_gate.view(-1, group_size)
        leader = realism.view(-1, group_size).argmax(dim=1, keepdim=True)
        grouped.scatter_(1, leader, True)

        gate = pass_gate.to(rewards.dtype)
        rewards = rewards - penalty * (1.0 - gate)
        metrics["coverage_gate_pass"] = gate.mean().detach()
        metrics["coverage_realism_pass"] = pass_realism.float().mean().detach()

    metrics["reward_value"] = rewards.detach()
    return rewards.detach(), metrics
