# Replicate 2026

This repository adapts the EEGMoE paper,
`EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning`,
to landslide mapping from multimodal geospatial data.

The current repo is a working baseline for:

- atlas-derived weak-label segmentation
- multimodal raster preprocessing
- supervised training with an SSMoE-style backbone
- tile-wise inference and visualization review

The main workflow is:

1. keep raw GIS assets under `datasets/`
2. keep label sources under `labels/`
3. convert raw scenes into processed tiles with `prepare_tiles.py`
4. inspect processed data with `analyze_dataset.py`
5. train with `train.py`
6. review predictions with `predict.py`
7. reset processed data safely with `clear_processed_data.py`

## Competition Context

The competition asks for the EEGMoE architecture to be adapted to landslide prediction, using the Landslide Atlas as ground truth.

In practice, that means:

- the atlas PDF is the reference source
- training still needs machine-usable labels
- labels must become raster masks or vector polygons
- the current project uses manually created atlas-derived polygon layers as supervision

Related notes are in [ATLAS_LABEL_PLAN.md](./ATLAS_LABEL_PLAN.md).

## Current Status

What is already working:

- multimodal preprocessing from Sentinel-1, Sentinel-2, DEM, rainfall, soil moisture, and vector/raster labels
- support for multiple label files merged with `union` or `intersection`
- segmentation training with class-imbalance handling
- lightweight training augmentations for flips, rotations, and Gaussian noise
- run-level experiment tracking with saved configs and epoch logs
- dataset QC summaries
- checkpoint-based prediction review with probability maps and visualization PNGs
- stitched scene export to GeoTIFF and NumPy outputs
- safe cleanup of processed datasets between experiments

What is not yet implemented:

- paper-style self-supervised pretraining
- full-scene stitched geospatial export
- broader event-wise evaluation across many independent landslide events

## Repository Overview

- [config.py](./config.py): dataclass config objects
- [dataset.py](./dataset.py): processed tile loading, manifest support, folder discovery
- [satellite_ssmoe.py](./satellite_ssmoe.py): patch embedding plus specific/shared MoE routing
- [model.py](./model.py): segmentation/classification model wrapper
- [losses.py](./losses.py): BCE and Dice-based losses
- [prepare_tiles.py](./prepare_tiles.py): raw GIS to processed tiles
- [prepare_manifest.py](./prepare_manifest.py): JSONL manifest generation
- [analyze_dataset.py](./analyze_dataset.py): class balance and schema inspection
- [train.py](./train.py): supervised training and checkpoint saving
- [predict.py](./predict.py): tile inference, probability maps, binary masks, review PNGs
- [clear_processed_data.py](./clear_processed_data.py): safe cleanup for new preprocessing attempts
- [ATLAS_LABEL_PLAN.md](./ATLAS_LABEL_PLAN.md): atlas labeling strategy notes

## Model Architecture

The model follows this flow:

1. load a multimodal tile tensor with shape `[channels, height, width]`
2. convert the tile into patch tokens
3. route each token through:
   - a `SpecificMoE` branch with top-k routing
   - a `SharedMoE` branch with soft routing across all shared experts
4. combine expert outputs back into a feature map
5. apply a task head for:
   - segmentation logits, or
   - tile-level classification logits

Core implementation:

- [satellite_ssmoe.py](./satellite_ssmoe.py)
- [model.py](./model.py)

## Supported Tasks

- `segmentation`
  - input: multimodal tile tensor
  - target: binary mask
  - output: per-pixel logits
- `classification`
  - input: multimodal tile tensor
  - target: scalar `0` or `1`
  - output: one logit per tile

## Environment Setup

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The pinned dependencies include PyTorch, NumPy, Matplotlib, rasterio, xarray, netCDF4, geopandas, shapely, Fiona, and pyproj.

## Data Layout

Recommended project structure:

```text
datasets/
  2024-12-11/
    DEM/
    Rainfall Data/
    Sentinel-1/
    Sentinel-2/
    Soil_moisture/
  2024-12-16/
    DEM/
    Sentinel-1/
    Sentinel-2/
labels/
  vector/
  raster/
pdfs/
references/
data/
  train/
    images/
    targets/
    metadata/
  val/
    images/
    targets/
    metadata/
data_dense/
  train/
  val/
checkpoints/
predictions/
reports/
```

Folder meaning:

- `datasets/`: untouched raw GIS data
- `labels/`: vector or raster supervision layers
- `pdfs/`: atlas and reference PDFs
- `data/`: processed baseline tiles
- `data_dense/`: denser tiling experiments
- `checkpoints/`: trained models
- `predictions/`: inference outputs and review images
- `reports/`: QC summaries

## Training Data Format

Training code does not consume raw GIS files directly. It expects processed tensors:

- image tensor: `[channels, height, width]`
- segmentation mask: `[1, height, width]` or `[height, width]`
- classification label: scalar float

Supported formats:

- `.npy`
- `.npz`
- `.pt`
- `.pth`

## Folder Dataset Layout

If training from folder roots, the expected structure is:

```text
data/
  train/
    images/
      wayanad_2024/
        2024-12-11_704_1216_image.npy
    targets/
      wayanad_2024/
        2024-12-11_704_1216_mask.npy
    metadata/
      wayanad_2024.json
  val/
    images/
      wayanad_2024_val/
        2024-12-16_704_1216_image.npy
    targets/
      wayanad_2024_val/
        2024-12-16_704_1216_mask.npy
    metadata/
      wayanad_2024_val.json
```

Matching is name-based after removing common suffixes such as:

- `_image`
- `_img`
- `_mask`
- `_target`
- `_label`

For segmentation, either `targets/` or `masks/` is accepted.

## Manifest Format

You can also train from JSONL manifests.

Segmentation example:

```json
{"id":"wayanad_tile_0001","image":"data/train/images/wayanad_2024/tile_0001_image.npy","target":"data/train/targets/wayanad_2024/tile_0001_mask.npy","event_id":"wayanad_2024"}
```

Classification example:

```json
{"id":"puthumala_tile_0007","image":"data/train/images/puthumala_2019/tile_0007_image.npy","label":1.0,"event_id":"puthumala_2019"}
```

Generate manifests with:

```powershell
python prepare_manifest.py --data-root data/train --output data/train.jsonl --task-type segmentation
python prepare_manifest.py --data-root data/val --output data/val.jsonl --task-type segmentation
```

## Labels And Atlas Ground Truth

The atlas PDF is not directly trainable. Training needs one of:

- `.gpkg`
- `.geojson`
- `.shp`
- raster label files such as `.tif`

Current label strategy:

- use atlas pages as reference
- digitize landslide polygons manually when needed
- store vector labels under `labels/vector/`
- let `prepare_tiles.py` rasterize them onto the model grid

Multiple label files are supported and can be merged during preprocessing.

## Preprocessing With `prepare_tiles.py`

`prepare_tiles.py`:

- discovers Sentinel-2 bands
- loads Sentinel-1, DEM, rainfall, and soil moisture
- picks a common grid
- reprojects and resamples modalities
- rasterizes labels when needed
- writes processed tiles and metadata

### Inputs Supported

- Sentinel-2 `.tif`
- Sentinel-1 `.tif`
- DEM `.tif`
- rainfall NetCDF
- soil moisture `.tif` or zipped `.tif`
- vector or raster labels

### Basic Example

```powershell
python prepare_tiles.py `
  --scene-root datasets/2024-12-11 `
  --output-root data `
  --split train `
  --event-id wayanad_2024 `
  --scene-date 2024-12-11 `
  --label-path labels/vector/wayanad_2024_atlas.gpkg `
  --rainfall-path "datasets/2024-12-11/Rainfall Data/kerala_rainfall_data.nc" `
  --soil-moisture-path "datasets/2024-12-11/Soil_moisture/Soil_Mositure.zip" `
  --tile-size 64 `
  --stride 64 `
  --skip-empty-targets `
  --output-format npy
```

### Multiple Label Files

Repeat `--label-path` to combine multiple layers:

```powershell
python prepare_tiles.py `
  --scene-root datasets/2024-12-11 `
  --output-root data_dense `
  --split train `
  --event-id wayanad_2024 `
  --scene-date 2024-12-11 `
  --label-path labels/vector/wayanad_2024_atlas.gpkg `
  --label-path labels/vector/wayanad_2024_atlas_v1.gpkg `
  --label-merge-mode union `
  --rainfall-path "datasets/2024-12-11/Rainfall Data/kerala_rainfall_data.nc" `
  --soil-moisture-path "datasets/2024-12-11/Soil_moisture/Soil_Mositure.zip" `
  --tile-size 64 `
  --stride 32 `
  --skip-empty-targets `
  --output-format npy
```

Merge modes:

- `union`: keep any positive pixel from any label source
- `intersection`: keep only overlapping positives

### Dense Tiling Example

For stronger segmentation coverage:

```powershell
python prepare_tiles.py `
  --scene-root datasets/2024-12-11 `
  --output-root data_dense `
  --split train `
  --event-id wayanad_2024 `
  --scene-date 2024-12-11 `
  --label-path labels/vector/wayanad_2024_atlas.gpkg `
  --rainfall-path "datasets/2024-12-11/Rainfall Data/kerala_rainfall_data.nc" `
  --soil-moisture-path "datasets/2024-12-11/Soil_moisture/Soil_Mositure.zip" `
  --tile-size 64 `
  --stride 32 `
  --skip-empty-targets `
  --output-format npy
```

### Important Preprocessing Notes

- supervised segmentation requires a real label source
- train and val should use the same modality schema
- event-wise splits are better than random nearby-tile splits
- `--skip-empty-targets` is very useful for extremely sparse labels

## Dataset QC With `analyze_dataset.py`

Before training, inspect processed data:

```powershell
python analyze_dataset.py --data-root data/train --task-type segmentation
python analyze_dataset.py --data-root data/val --task-type segmentation
```

This reports:

- sample count
- event count
- image shape preview
- channel count
- per-channel min, max, mean, and std
- positive tile fraction
- positive pixel fraction
- metadata-derived channel names

Save summaries with:

```powershell
python analyze_dataset.py --data-root data_dense/train --task-type segmentation --output-json reports/data_dense_train_summary.json
python analyze_dataset.py --data-root data_dense/val --task-type segmentation --output-json reports/data_dense_val_summary.json
```

## Training With `train.py`

Train from folder roots:

```powershell
python train.py `
  --train-root data/train `
  --val-root data/val `
  --task-type segmentation `
  --in-channels 17 `
  --output checkpoints/landslide_seg.pt
```

Train from manifests:

```powershell
python train.py `
  --train-manifest data/train.jsonl `
  --val-manifest data/val.jsonl `
  --task-type segmentation `
  --in-channels 17 `
  --output checkpoints/landslide_seg.pt
```

Dense-tile training example:

```powershell
python train.py `
  --train-root data_dense/train `
  --val-root data_dense/val `
  --task-type segmentation `
  --in-channels 17 `
  --batch-size 2 `
  --epochs 20 `
  --patch-size 8 `
  --specific-experts 4 `
  --shared-experts 2 `
  --top-k 2 `
  --positive-class-weight 8 `
  --augment-flip `
  --augment-rotate `
  --output checkpoints/landslide_seg_dense_w8.pt
```

### Training Notes

- folder-based training can infer channel schema from processed data
- validation checkpoints are saved when the metric improves
- segmentation uses IoU for validation
- classification uses accuracy
- checkpoint files now store model/data/train config for reproducible inference
- each run writes configs and epoch logs under `reports/runs/`

### Helpful Training Flags

- `--task-type segmentation`
- `--task-type classification`
- `--batch-size 2`
- `--epochs 20`
- `--learning-rate 1e-4`
- `--weight-decay 1e-4`
- `--positive-class-weight 8`
- `--dim 128`
- `--patch-size 8`
- `--specific-experts 4`
- `--shared-experts 2`
- `--top-k 2`

## Prediction Review With `predict.py`

After training, run:

```powershell
python predict.py `
  --data-root data_dense/val `
  --checkpoint checkpoints/landslide_seg_dense_w8.pt `
  --output-root predictions/dense_val_review `
  --threshold 0.5 `
  --max-visualizations 8
```

Outputs:

- probability maps under `predictions/.../probabilities/`
- binary masks under `predictions/.../binaries/`
- qualitative PNGs under `predictions/.../visualizations/`
- one summary JSON per run

The PNG review panels show:

- RGB tile preview
- target mask
- probability map
- binary prediction
- specific-expert map

The summary JSON includes:

- `sample_count`
- `threshold`
- `mean_iou` for segmentation
- checkpoint path

## Experiment Tracking

`train.py` now writes run metadata automatically under `reports/runs/`.

Example:

```powershell
python train.py `
  --train-root data_dense/train `
  --val-root data_dense/val `
  --task-type segmentation `
  --in-channels 17 `
  --positive-class-weight 8 `
  --augment-flip `
  --augment-rotate `
  --run-name dense_w8_augmented `
  --output checkpoints/landslide_seg_dense_w8_aug.pt
```

This creates a run folder with:

- `config.json`
- `history.jsonl`
- `latest.json`
- `summary.json`

## Full-Scene Stitched Export

After training, you can stitch tile predictions back into a scene-sized raster:

```powershell
python stitch_scene.py `
  --data-root data_dense/val `
  --checkpoint checkpoints/landslide_seg_dense_w8.pt `
  --event-id wayanad_2024_val `
  --output-root scene_exports `
  --threshold 0.5
```

Outputs include:

- scene probability GeoTIFF
- binary prediction GeoTIFF
- coverage GeoTIFF
- stitched target GeoTIFF
- NumPy exports
- summary JSON

## Cleanup With `clear_processed_data.py`

Preview what would be deleted:

```powershell
python clear_processed_data.py --root data --dry-run
```

Delete one event from one split:

```powershell
python clear_processed_data.py `
  --root data `
  --split train `
  --event-id wayanad_2024 `
  --include-cache `
  --remove-empty-parents
```

Delete all processed train/val data in `data_dense`:

```powershell
python clear_processed_data.py `
  --root data_dense `
  --split train `
  --split val `
  --include-cache
```

This is the safest way to reset processed datasets between experiments.

## Interpreting Results

Use the outputs in two layers:

- `summary.json` tells you the overall metric
- `visualizations/` tells you why the model is doing well or badly

When reading prediction results:

- increasing `mean_iou` means better overlap on average
- probability maps show where the model is confident
- binary predictions depend on the chosen threshold
- visual overlays reveal false positives, misses, and shape quality

## Current Best Local Baseline

At the time of writing, the strongest local run in this workspace uses:

- dense tiles
- multiple merged label layers
- `positive-class-weight 8`

Example assets:

- [landslide_seg_dense_w8.pt](./checkpoints/landslide_seg_dense_w8.pt)
- [summary.json](./predictions/dense_val_review/summary.json)

These are workspace artifacts, not guaranteed reference benchmarks for every future run.

## Known Limitations

This is still a baseline repo. It does not yet include:

- self-supervised pretraining from the original EEGMoE paper
- full-scene stitched geospatial export
- advanced augmentation policies
- experiment tracking
- distributed training
- uncertainty estimation
- deployment packaging

## Suggested Next Steps

1. improve and expand vector labels
2. regenerate dense train/val splits
3. run QC with `analyze_dataset.py`
4. train with class-imbalance weighting
5. review predictions with `predict.py`
6. tune thresholds
7. add paper-faithful self-supervised pretraining
8. add full-scene export for final competition deliverables

## Quick Start

For a fast end-to-end smoke test using the demo data:

```powershell
python train.py --train-root demo_data/train --val-root demo_data/val --task-type segmentation --in-channels 14 --output demo_checkpoints/demo_seg.pt
python predict.py --data-root demo_data/val --checkpoint demo_checkpoints/demo_seg.pt --output-root predictions/demo_review
```

## License

This project is released under the MIT License. See [LICENSE](./LICENSE).
