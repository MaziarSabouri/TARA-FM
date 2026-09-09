"""Build pickled train/valid/external datasets for RL-FM-Spect from the raw
Thyroid scintigraphy dataset (Excel metadata + per-patient DICOM/NIfTI files).

Inputs:
  - Excel metadata with "Train" and "External" sheets. Column "Folder" gives
    the per-patient subfolder name; column "Class1" gives the disease class.
  - Image_and_Segmentation/<Folder>/<Folder>.dcm: single-frame planar
    scintigraphy image (DICOM).
  - Image_and_Segmentation/<Folder>/Untitled.nii.gz: whole-thyroid binary
    segmentation mask (NIfTI), same spatial size as the image.

Outputs (written to --output_dir):
  - train_data_128.pkl: {"train": [...], "valid": [...], "class_names": [...]}
      built from the "Train" sheet, split per-class (stratified) so it can be
      consumed directly by trainer.py / fm_lightning.py.
  - external_data_128.pkl (optional): {"external": [...], "class_names": [...]}
      built only when --skip_external is not passed; never used by FM/RL.
  - class_map.json: human-readable class name <-> index mapping.

Each sample dict has:
  image:      float32 ndarray [1, H, W], per-image min-max normalized to [0, 1]
  mask:       uint8 ndarray [1, H, W], whole-thyroid binary mask (0/1)
  class:      int class index, see CLASS_NAMES
  folder_id:  str, original patient/scan folder name
  name:       same as folder_id (kept for compatibility with utils/general_utils.py)
  metadata:   dict with best-effort gender/age/center_id/manufacturer

Notes on data-quality decisions (verified against all 2,954 rows/files):
  - The "image" file is a DICOM, NOT a NIfTI file as originally assumed; only
    the mask is NIfTI. Both are read with SimpleITK, which handles single
    slice/frame files for either format identically.
  - Native sizes are 128x128 or 256x256, single-frame, single-component.
    Of 2,954 cases, 319 (all from center E) are natively 256x256; the rest
    are already 128x128. The 256x256 (center E) images are downsampled to
    128x128 with BSpline interpolation, and their masks with
    nearest-neighbor interpolation (kept binary); the already-128x128 cases
    are passed through unchanged (no resampling).
  - One "Train" row has Class1="TH" (typo): Class2 and Class4 both say "TI"
    for that row, so it is normalized to Thyroiditis (see CLASS1_MAP).
  - 13 of 2,954 cases have 2-frame DICOMs (a dual-energy-window acquisition;
    DICOM tag (0054,0400) confirms both frames are "ANT"/anterior, not an
    anterior/posterior pair). For the image, frame 0 consistently holds more
    counts (the primary photopeak window). For the paired mask, frame 0 is
    always EMPTY (sum=0) and frame 1 holds the actual segmentation. Both
    cases are handled by the same rule: pick the frame with the larger sum
    (highest information content), independently for the image and the mask.
  - The source scans are static (single-frame) planar images -- there is no
    dynamic frame/time axis to reposition, so the "averaged points placed at
    the end of the frame" constraint from the project plan does not apply at
    this pickling stage.

This script needs pandas / openpyxl / SimpleITK / opencv-python, which are
NOT in rl-fm-spect/requirements.txt (that env is for model training only).
Run it with an environment that has them, e.g.:
    conda run -n thyroid python prepare_dataset.py
"""

import argparse
import json
import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

CLASS_NAMES = ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Raw Excel `Class1` values -> canonical class name.
CLASS1_MAP = {
    "Normal": "Normal",
    "DG": "Diffuse Goiter",
    "MNG": "Nodular Goiter",
    "Thyroiditis": "Thyroiditis",
    # Typo in source data: this Class1 value doesn't otherwise exist, but the
    # single row that has it also has Class2="TI" and Class4="TI"
    # (Thyroiditis), so it is normalized here instead of dropped.
    "TH": "Thyroiditis",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel_path", required=True, help="Path to the metadata workbook.")
    parser.add_argument("--data_root", required=True, help="Root of patient image/mask folders.")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of the 'Train' sheet held out per-class as the FM 'valid' split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_external",
        action="store_true",
        help="Build development train/valid data only (recommended for this FM/RL release).",
    )
    return parser.parse_args()


def _sitk_resample(arr: np.ndarray, size: int, interpolator) -> np.ndarray:
    """Resample a 2D pixel-grid array to size x size with a SimpleITK
    interpolator (spacing is set purely so the resample maps the full
    original grid onto the new grid; there is no physical spacing info in
    the source DICOM/NIfTI planar images to preserve here)."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    orig_size = img.GetSize()  # (W, H)
    new_spacing = [osz / size for osz in orig_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([size, size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    out = resampler.Execute(img)
    return sitk.GetArrayFromImage(out)


def resize_image(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D count image to `size`x`size`.

    Per the dataset's documented acquisition mix: all cases from center E
    are natively 256x256 and everyone else is already 128x128. The 256x256
    (center E) cases are downsampled with BSpline interpolation; anything
    already at the target size is a no-op passthrough. (Upscaling isn't
    expected to occur in this dataset, but falls back to BSpline too.)
    """
    if arr.shape == (size, size):
        return arr.astype(np.float32)
    return _sitk_resample(arr, size, sitk.sitkBSpline).astype(np.float32)


def resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a binary mask to `size`x`size` with nearest-neighbor
    interpolation so the segmentation stays binary (0/1)."""
    if arr.shape == (size, size):
        return arr.astype(np.uint8)
    resized = _sitk_resample(arr, size, sitk.sitkNearestNeighbor)
    return (resized > 0.5).astype(np.uint8)


def read_slice(path: str) -> np.ndarray:
    """Read a 2D planar image/mask file (DICOM or NIfTI) as an [H, W] array.

    A handful of source files have 2 frames from a dual-energy-window
    acquisition (both confirmed anterior via DICOM tag (0054,0400)). The
    frame with the larger sum is used: for images that's the primary
    (photopeak) window; for masks, the other frame is always empty (sum=0),
    so this also picks out the real segmentation.
    """
    sitk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitk_img)  # [F, H, W]
    if arr.ndim == 3:
        if arr.shape[0] > 1:
            sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
            best = int(np.argmax(sums))
            print(f"[INFO] {path}: {arr.shape[0]} frames found (sums={sums.tolist()}), using frame {best}.")
            arr = arr[best]
        else:
            arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported array shape {arr.shape} in {path}")
    return arr


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1]; preserves raw count shape/statistics."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def resolve_class(row: pd.Series, folder: str) -> Optional[int]:
    raw = row.get("Class1")
    raw = None if pd.isna(raw) else str(raw).strip()
    name = CLASS1_MAP.get(raw)
    if name is None:
        print(f"[WARN] Skipping folder={folder}: unmapped Class1={raw!r}")
        return None
    return CLASS_TO_IDX[name]


def _clean_meta(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def build_sample(data_root: str, row: pd.Series) -> Optional[Dict]:
    folder = str(row["Folder"])
    image_path = os.path.join(data_root, folder, f"{folder}.dcm")
    mask_path = os.path.join(data_root, folder, "Untitled.nii.gz")

    if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
        print(f"[WARN] Skipping folder={folder}: missing image or mask file")
        return None

    cls = resolve_class(row, folder)
    if cls is None:
        return None

    try:
        image = read_slice(image_path)
        mask = read_slice(mask_path)
    except Exception as e:
        print(f"[WARN] Skipping folder={folder}: failed to read ({e})")
        return None

    if image.shape != mask.shape:
        print(f"[WARN] Skipping folder={folder}: image/mask shape mismatch {image.shape} vs {mask.shape}")
        return None

    return {
        "folder": folder,
        "class": cls,
        "raw_image": image,
        "raw_mask": mask,
        "gender": row.get("Gender"),
        "age": row.get("Age"),
        "center_id": row.get("CenterID"),
        "manufacturer": row.get("Manufacturer"),
    }


def finalize_sample(raw_sample: Dict, image_size: int) -> Dict:
    image = minmax_normalize(resize_image(raw_sample["raw_image"], image_size))[None]  # [1,H,W]
    mask = resize_mask(raw_sample["raw_mask"], image_size)[None]  # [1,H,W]

    return {
        "image": image,
        "mask": mask,
        "class": raw_sample["class"],
        "folder_id": raw_sample["folder"],
        "name": raw_sample["folder"],
        "metadata": {
            "gender": _clean_meta(raw_sample["gender"]),
            "age": _clean_meta(raw_sample["age"]),
            "center_id": _clean_meta(raw_sample["center_id"]),
            "manufacturer": _clean_meta(raw_sample["manufacturer"]),
        },
    }


def load_sheet_samples(excel_path: str, data_root: str, sheet: str, image_size: int) -> List[Dict]:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    samples = []
    for _, row in df.iterrows():
        raw = build_sample(data_root, row)
        if raw is None:
            continue
        samples.append(finalize_sample(raw, image_size))
    return samples


def stratified_split(samples: List[Dict], val_frac: float, seed: int):
    by_class: Dict[int, List[Dict]] = {}
    for s in samples:
        by_class.setdefault(s["class"], []).append(s)

    rng = random.Random(seed)
    train, valid = [], []
    for group in by_class.values():
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        valid.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def print_class_counts(name: str, samples: List[Dict]) -> None:
    counts = Counter(s["class"] for s in samples)
    summary = ", ".join(f"{CLASS_NAMES[c]}={counts.get(c, 0)}" for c in range(len(CLASS_NAMES)))
    print(f"{name}: n={len(samples)} ({summary})")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading 'Train' sheet from {args.excel_path} ...")
    train_pool = load_sheet_samples(args.excel_path, args.data_root, "Train", args.image_size)

    train_samples, valid_samples = stratified_split(train_pool, args.val_frac, args.seed)

    print_class_counts("train", train_samples)
    print_class_counts("valid", valid_samples)

    train_pkl = {"train": train_samples, "valid": valid_samples, "class_names": CLASS_NAMES}

    train_path = os.path.join(args.output_dir, "train_data_128.pkl")
    class_map_path = os.path.join(args.output_dir, "class_map.json")

    with open(train_path, "wb") as f:
        pickle.dump(train_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(class_map_path, "w") as f:
        json.dump({"class_names": CLASS_NAMES, "class_to_idx": CLASS_TO_IDX}, f, indent=2)

    print(f"Saved {train_path}")
    print(f"Saved {class_map_path}")

    if not args.skip_external:
        print(f"Reading 'External' sheet from {args.excel_path} ...")
        external_samples = load_sheet_samples(
            args.excel_path, args.data_root, "External", args.image_size
        )
        print_class_counts("external", external_samples)
        external_path = os.path.join(args.output_dir, "external_data_128.pkl")
        with open(external_path, "wb") as f:
            pickle.dump(
                {"external": external_samples, "class_names": CLASS_NAMES},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved {external_path}")


if __name__ == "__main__":
    main()
