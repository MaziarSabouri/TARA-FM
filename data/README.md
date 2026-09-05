# Data layout

Raw patient images and labels are **not** shipped with this repository. Contact the authors for access, then build training pickles locally.

## Expected raw inputs

Point `prepare_dataset.py` at:

- `--excel_path`: Excel workbook with `Train` and `External` sheets  
  (`Folder` = patient subfolder; `Class1` = disease label)
- `--data_root`: directory of per-patient folders  
  - `<Folder>/<Folder>.dcm` — planar scintigraphy (DICOM)  
  - `<Folder>/Untitled.nii.gz` — whole-thyroid binary mask (NIfTI)

## Build pickles

From the repository root:

```bash
python prepare_dataset.py \
  --excel_path /path/to/metadata.xlsx \
  --data_root /path/to/Image_and_Segmentation \
  --output_dir ./data \
  --image_size 128
```

Optional visualization:

```bash
python visualize_dataset.py --pickle_path data/train_data_128.pkl --output_dir data/viz
```

## Pickle schema

**`train_data_128.pkl`**

```text
{
  "train": [sample, ...],
  "valid": [sample, ...],
  "class_names": ["Normal", "Diffuse Goiter", "Nodular Goiter", "Thyroiditis"]
}
```

**`external_data_128.pkl`** (held-out evaluation; not required for FM/RL training)

```text
{
  "external": [sample, ...],
  "class_names": [...]
}
```

Each `sample` dict:

| key | type | description |
|-----|------|-------------|
| `image` | `float32 [1,H,W]` | min–max normalized to `[0,1]` |
| `mask` | `uint8 [1,H,W]` | binary thyroid mask `{0,1}` |
| `class` | `int` | class index (see `class_names`) |
| `folder_id` / `name` | `str` | patient/scan id |
| `metadata` | `dict` | optional demographics / center |

Configs default to `data_args.pickle_path: ./data/train_data_128.pkl`.
