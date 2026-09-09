import argparse
import json
import os
import sys
import warnings
import pickle
import time

from tqdm import tqdm

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch
from typing import Optional

from utils.general_utils import (
    create_dataloader,
    image_to_display_range,
    load_and_prepare_data,
    load_config,
    resolve_allowed_ids,
    resolve_mask_conditioning_config,
)
from utils.utils_fm import sample_batch
from utils.checkpoint_utils import resolve_checkpoint_path, load_model_from_checkpoint
from utils.solver_utils import build_solver_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/fm_mask_class.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Total number of samples to save, drawn i.i.d. from --gen_split. "
        "Ignored if --num_samples_per_class is set.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=None,
        help="If set, generate exactly this many samples PER CLASS (e.g. 1000 -> 4000 total "
        "for 4 classes), each conditioned on a mask/class drawn from that class's real "
        "samples. Requires class labels in the dataset. Overrides --num_samples.",
    )
    parser.add_argument(
        "--gen_split",
        type=str,
        default=None,
        help="Dataset split to draw conditioning masks/classes from. Defaults to "
        "data_args.split_train when --num_samples_per_class is set (more mask diversity "
        "per class), else data_args.split_val (legacy behavior).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint file or directory. If omitted, "
            "`fm_train_args.checkpoint_dir/<config_basename>` is used."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
        help="Number of inference steps during sampling.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Explicit output pickle path. If omitted, a name is derived from the "
        "config/checkpoint/step count and written next to the checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sampling batch size (defaults to fm_train_args.batch_size).",
    )
    parser.add_argument(
        "--only_class",
        type=int,
        default=None,
        help="Generate only this class index (used with --num_samples_per_class). Large "
             "pools are generated one class per process so host RAM stays bounded.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for noise and mask selection.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if the output pickle already exists.",
    )
    return parser.parse_args()


def build_sample_dict(final_img, mask, class_idx, name, true_img):
    return {
        "image": final_img,
        "mask": mask,
        "class": class_idx,
        "name": name,
        "true_data": true_img,
    }


def generate_iid(
    model,
    solver_config,
    data,
    device,
    class_conditioning: bool,
    mask_conditioning: bool,
    idx_to_class,
    num_samples: int,
) -> list:
    """Legacy mode: draw `num_samples` batches i.i.d. from the (shuffled) split."""
    loader = create_dataloader(
        Images=data["images"],
        Masks=data["masks"],
        classes=data.get("classes"),
        batch_size=num_samples if num_samples < 64 else 64,
        # Shuffle so a limited num_samples reflects the full distribution
        # (important for rare classes in boost-based augmentation).
        shuffle=True,
    )

    samples = []
    total_time = 0.0
    iterator = iter(loader)

    with tqdm(total=num_samples, desc="Generating Samples") as pbar:
        while len(samples) < num_samples:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            start_time = time.time()
            final_imgs = sample_batch(
                model=model,
                solver_config=solver_config,
                batch=batch,
                device=device,
                class_conditioning=class_conditioning,
                mask_conditioning=mask_conditioning,
            )
            total_time += time.time() - start_time

            images = batch["images"]
            masks = batch.get("masks") if mask_conditioning else None
            classes = batch.get("classes") if class_conditioning else None

            final_imgs = image_to_display_range(final_imgs).cpu().detach().numpy()
            masks = masks.cpu().detach().numpy() if masks is not None else None
            classes = classes.cpu().detach().numpy() if classes is not None else None
            true_imgs = image_to_display_range(images).cpu().detach().numpy()

            for i in range(final_imgs.shape[0]):
                if len(samples) >= num_samples:
                    break
                cls_idx = (
                    idx_to_class[classes[i].argmax()]
                    if (idx_to_class is not None and classes is not None)
                    else None
                )
                samples.append(
                    build_sample_dict(
                        final_imgs[i],
                        masks[i] if masks is not None else None,
                        cls_idx,
                        f"sample_{len(samples)}",
                        true_imgs[i],
                    )
                )
                pbar.update(1)

    if samples:
        print(f"Average time per batch: {total_time / max(1, (len(samples) // 1)):.4f}s (over generation loop)")
    return samples, total_time


def generate_balanced_per_class(
    model,
    solver_config,
    data,
    device,
    class_conditioning: bool,
    mask_conditioning: bool,
    num_per_class: int,
    batch_size: int,
    only_class: Optional[int] = None,
) -> list:
    """Generate exactly `num_per_class` samples for every class in `data["classes"]`.

    Each generated image is conditioned on a mask (and, if class_conditioning is on,
    the corresponding one-hot class vector) drawn from real samples of that class --
    cycling through them with replacement if num_per_class exceeds how many real
    examples of that class exist.

    `only_class` restricts generation to a single class index. Large pools (the
    reward-selection experiment draws thousands per class) otherwise accumulate
    every sample in one list before pickling, which peaks at several GB of host
    RAM; generating one class per process keeps that bounded.
    """
    classes_onehot = data.get("classes")
    if classes_onehot is None:
        raise ValueError(
            "--num_samples_per_class requires class labels in the dataset "
            "(data_args split must include a 'class' field for every sample)."
        )
    class_indices = classes_onehot.argmax(dim=1)
    num_classes = classes_onehot.shape[1]

    samples = []
    total_time = 0.0

    for cls in range(num_classes):
        if only_class is not None and cls != int(only_class):
            continue
        pool = (class_indices == cls).nonzero(as_tuple=True)[0]
        if pool.numel() == 0:
            print(f"[WARN] No real samples for class {cls}; skipping.")
            continue

        collected = 0
        with tqdm(total=num_per_class, desc=f"Class {cls}") as pbar:
            while collected < num_per_class:
                take = min(batch_size, num_per_class - collected)
                # Sample with replacement so classes with fewer real examples still
                # reach num_per_class, cycling through their real masks repeatedly.
                idx = pool[torch.randint(0, pool.numel(), (take,))]

                batch = {
                    "images": data["images"].index_select(0, idx),
                    "masks": data["masks"].index_select(0, idx),
                    "classes": classes_onehot.index_select(0, idx),
                }

                start_time = time.time()
                final_imgs = sample_batch(
                    model=model,
                    solver_config=solver_config,
                    batch=batch,
                    device=device,
                    class_conditioning=class_conditioning,
                    mask_conditioning=mask_conditioning,
                )
                total_time += time.time() - start_time

                final_imgs = image_to_display_range(final_imgs).cpu().detach().numpy()
                masks_np = batch["masks"].cpu().detach().numpy() if mask_conditioning else None
                true_imgs = image_to_display_range(batch["images"]).cpu().detach().numpy()

                for i in range(final_imgs.shape[0]):
                    samples.append(
                        build_sample_dict(
                            final_imgs[i],
                            masks_np[i] if masks_np is not None else None,
                            cls,
                            f"class{cls}_sample_{collected}",
                            true_imgs[i],
                        )
                    )
                    collected += 1
                    pbar.update(1)

    if samples:
        print(f"Total sampling time: {total_time:.1f}s for {len(samples)} samples "
              f"({1000 * total_time / len(samples):.1f} ms/image).")
    return samples, total_time


def main() -> None:
    args = parse_args()
    config_path = args.config_path

    config = load_config(config_path)
    config["train_args"] = config["fm_train_args"]
    resolve_mask_conditioning_config(config)
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(args.model_path, config, config_path)
    print(f"Using checkpoint: {checkpoint_path}")

    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Inference device:", device)

    model, metadata = load_model_from_checkpoint(checkpoint_path, config, device)
    solver_config = build_solver_config(config, args.num_inference_steps)
    print(
        f"Solver config: method={solver_config['method']}, "
        f"step_size={solver_config['step_size']}, "
        f"time_points={solver_config['time_points']}"
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    ckpt_name = metadata["checkpoint_name"]
    per_class_suffix = f"_perclass{args.num_samples_per_class}" if args.num_samples_per_class else ""
    output_path = args.output_path or os.path.join(
        checkpoint_dir,
        f"samples_{config_name}_{ckpt_name}_steps{solver_config['time_points']}{per_class_suffix}.pkl",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving generated data to {output_path}")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"File {output_path} already exists (use --overwrite to regenerate). Exiting.")
        sys.exit()

    data_config = config["data_args"]
    default_split = data_config["split_train"] if args.num_samples_per_class else data_config["split_val"]
    gen_split = args.gen_split or default_split
    print(f"Drawing conditioning masks/classes from split: {gen_split!r}")

    data = load_and_prepare_data(
        pickle_path=data_config["pickle_path"],
        split=gen_split,
        convert_classes_to_onehot=True,
        mask_num_classes=data_config.get("mask_num_classes"),
        allowed_ids=resolve_allowed_ids(data_config),
    )

    model_args = config.get("model_args", {})
    class_conditioning = bool(model_args.get("with_conditioning", False))
    mask_conditioning = bool(model_args.get("mask_conditioning", False))

    if args.num_samples_per_class:
        samples, total_time = generate_balanced_per_class(
            model=model,
            solver_config=solver_config,
            data=data,
            device=device,
            class_conditioning=class_conditioning,
            mask_conditioning=mask_conditioning,
            num_per_class=args.num_samples_per_class,
            batch_size=args.batch_size or config["train_args"].get("batch_size", 32),
            only_class=args.only_class,
        )
    else:
        idx_to_class = data.get("class_map")
        num_samples = len(data["images"]) if args.num_samples is None else args.num_samples
        print(f"Number of samples to save: {num_samples}")
        samples, total_time = generate_iid(
            model=model,
            solver_config=solver_config,
            data=data,
            device=device,
            class_conditioning=class_conditioning,
            mask_conditioning=mask_conditioning,
            idx_to_class=idx_to_class,
            num_samples=num_samples,
        )

    print(f"Collected {len(samples)} samples.")
    generated_data = {"train": samples}
    with open(output_path, "wb") as f:
        pickle.dump(generated_data, f)
    print(f"Generated data saved to {output_path}")

    # Sidecar with sampling cost, consumed by the ODE-step speed/utility ablation.
    timing = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "num_inference_steps": solver_config["time_points"],
        "solver_method": solver_config["method"],
        "n_samples": len(samples),
        "total_seconds": round(total_time, 3),
        "seconds_per_image": round(total_time / max(1, len(samples)), 5),
        "batch_size": args.batch_size or config["train_args"].get("batch_size", 32),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
    }
    timing_path = os.path.splitext(output_path)[0] + ".timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Timing metadata saved to {timing_path}")


if __name__ == "__main__":
    main()
