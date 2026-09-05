"""Render image + mask + overlay QC panels for every sample prepared by
prepare_dataset.py, for manual double-checking of the preprocessing.

Reads train_data_128.pkl (train + valid splits) and external_data_128.pkl
(external split) and writes one PNG per sample to:

    <Dataset>/QC/<split>/<class_name>/<folder_id>.png

Each PNG shows, left to right: the normalized image, the binary mask, and
an overlay (mask tinted red on top of the image), plus a header with the
folder id, class, and split. This makes misaligned masks or frame-selection
errors (e.g. the 13 dual-energy-window DICOMs) easy to spot by eye.
"""

import argparse
import os
import pickle
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES_DEFAULT = ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "Dataset"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_pickle", default=os.path.join(_THIS_DIR, "data", "train_data_128.pkl"))
    parser.add_argument("--external_pickle", default=os.path.join(_THIS_DIR, "data", "external_data_128.pkl"))
    parser.add_argument("--output_dir", default=os.path.join(_DEFAULT_DATASET_DIR, "QC"))
    parser.add_argument("--panel_size", type=int, default=256, help="Each panel is upscaled to this size for visibility.")
    parser.add_argument("--mask_alpha", type=float, default=0.45, help="Red overlay opacity over the mask region.")
    return parser.parse_args()


def to_uint8_gray(chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(chw)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def to_uint8_mask(chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(chw)
    if arr.ndim == 3:
        arr = arr[0]
    return (np.asarray(arr) > 0).astype(np.uint8) * 255


def make_panel(image_u8: np.ndarray, mask_u8: np.ndarray, size: int, mask_alpha: float) -> Image.Image:
    img = Image.fromarray(image_u8, mode="L").convert("RGB").resize((size, size), Image.NEAREST)
    mask_gray = Image.fromarray(mask_u8, mode="L").resize((size, size), Image.NEAREST)
    mask_rgb = mask_gray.convert("RGB")

    red = Image.new("RGB", img.size, (255, 0, 0))
    alpha = mask_gray.point(lambda p: int(p * mask_alpha))
    overlay = Image.composite(red, img, alpha)

    header_h = 40
    label_h = 22
    canvas = Image.new("RGB", (size * 3 + 20, size + header_h + label_h), (255, 255, 255))
    y = header_h
    canvas.paste(img, (0, y))
    canvas.paste(mask_rgb, (size + 10, y))
    canvas.paste(overlay, (2 * size + 20, y))
    return canvas


def render_sample(
    sample: Dict,
    split: str,
    class_names: List[str],
    panel_size: int,
    mask_alpha: float,
    output_dir: str,
) -> None:
    image_u8 = to_uint8_gray(sample["image"])
    mask_u8 = to_uint8_mask(sample["mask"])
    canvas = make_panel(image_u8, mask_u8, panel_size, mask_alpha)

    cls_idx = sample.get("class")
    cls_name = (
        class_names[cls_idx] if (isinstance(cls_idx, int) and 0 <= cls_idx < len(class_names)) else str(cls_idx)
    )
    folder_id = sample.get("folder_id", sample.get("name", "unknown"))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{folder_id}   class={cls_name}   split={split}", fill=(0, 0, 0))
    label_y = panel_size + 44
    draw.text((8, label_y), "image", fill=(0, 0, 0))
    draw.text((panel_size + 18, label_y), "mask", fill=(0, 0, 0))
    draw.text((2 * panel_size + 28, label_y), "overlay", fill=(0, 0, 0))

    out_dir = os.path.join(output_dir, split, cls_name)
    os.makedirs(out_dir, exist_ok=True)
    canvas.save(os.path.join(out_dir, f"{folder_id}.png"))


def iter_pickle_samples(pickle_path: str, splits: List[str]) -> Iterator[Tuple[str, Dict, List[str]]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    class_names = data.get("class_names", CLASS_NAMES_DEFAULT)
    for split in splits:
        if split not in data:
            continue
        for sample in data[split]:
            yield split, sample, class_names


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sources = [
        (args.train_pickle, ["train", "valid"]),
        (args.external_pickle, ["external"]),
    ]

    total = 0
    for pickle_path, splits in sources:
        if not os.path.isfile(pickle_path):
            print(f"[WARN] Missing pickle: {pickle_path}")
            continue
        for split, sample, class_names in iter_pickle_samples(pickle_path, splits):
            render_sample(sample, split, class_names, args.panel_size, args.mask_alpha, args.output_dir)
            total += 1
            if total % 500 == 0:
                print(f"Rendered {total} QC images...")

    print(f"Done. Rendered {total} QC images under {args.output_dir}")


if __name__ == "__main__":
    main()
