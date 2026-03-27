# Replicate 2026

This repository adapts the EEGMoE paper,
`EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning`,
to landslide susceptibility and detection.

The project goal is to use aligned multimodal raster tiles such as Sentinel-1, Sentinel-2,
rainfall, soil moisture, and Landslide Atlas labels to learn domain-shared and domain-specific
representations for landslide prediction.

## Current Status

- `satellite_ssmoe.py` contains the paper-aligned specific/shared MoE routing block adapted to image tokens.
- `dataset.py` loads preprocessed multimodal tiles either from a JSONL manifest or directly from a simple folder layout.
- `model.py` wraps the SSMoE block in a landslide segmentation or tile-classification model.
- `losses.py` contains binary losses for segmentation and classification.
- `train.py` provides a first supervised training loop.
- `prepare_manifest.py` can generate a manifest automatically from folders.

This is the first working scaffold. The paper-style masked self-supervised pretraining stage is
still a next step.

## Suggested Data Shape

For now, the code assumes that raw geospatial sources have already been aligned and exported into
tile tensors.

- Image tensor shape: `[channels, height, width]`
- Segmentation mask shape: `[1, height, width]` or `[height, width]`
- Classification label shape: scalar `0` or `1`

You can choose any channel layout as long as it matches `--in-channels`.
Examples:

- `10` channels for a Sentinel-2-only experiment
- `14` channels for `10` optical + `2` SAR + `1` rainfall + `1` soil moisture

## Manifest Format

Each line in the manifest is a JSON object. Example segmentation record:

```json
{"id":"wayanad_tile_0001","image":"data/tiles/wayanad_tile_0001_image.pt","target":"data/tiles/wayanad_tile_0001_mask.pt","event_id":"wayanad_2024"}
```

Example classification record:

```json
{"id":"puthumala_tile_0007","image":"data/tiles/puthumala_tile_0007_image.pt","label":1.0,"event_id":"puthumala_2019"}
```

Supported tensor formats:

- `.pt` or `.pth`
- `.npy`
- `.npz`

## Easier Folder Layout

If you do not want to hand-write manifests, use this layout:

```text
data/
  train/
    images/
      wayanad_2024/
        tile_0001_image.pt
        tile_0002_image.pt
    targets/
      wayanad_2024/
        tile_0001_mask.pt
        tile_0002_mask.pt
  val/
    images/
      puthumala_2019/
        tile_0001_image.pt
    targets/
      puthumala_2019/
        tile_0001_mask.pt
```

The loader matches files by name after removing common suffixes such as `_image`, `_img`, `_mask`, and `_target`.
Nested folders are allowed. Their relative path becomes the `event_id`, so `images/wayanad_2024/...` maps to `event_id = "wayanad_2024"`.
You can use either `targets/` or `masks/` for segmentation labels.

## Training

Install PyTorch in your environment first, then run:

```powershell
python train.py --train-manifest data/train.jsonl --val-manifest data/val.jsonl --task-type segmentation --in-channels 10
```

Or train directly from folders:

```powershell
python train.py --train-root data/train --val-root data/val --task-type segmentation --in-channels 14
```

If you want manifests for inspection or reuse:

```powershell
python prepare_manifest.py --data-root data/train --output data/train.jsonl --task-type segmentation
python prepare_manifest.py --data-root data/val --output data/val.jsonl --task-type segmentation
```

Useful flags:

- `--task-type segmentation`
- `--task-type classification`
- `--in-channels 10`
- `--patch-size 8`
- `--specific-experts 4`
- `--shared-experts 2`
- `--top-k 2`

## What Comes Next

1. Build the geospatial preprocessing pipeline to create aligned training tiles.
2. Decide the final input channel layout for Sentinel-1, Sentinel-2, rainfall, and soil moisture.
3. Add the masked reconstruction pretraining stage from the EEGMoE paper.
4. Add event-wise evaluation for Wayanad 2024 and Puthumala 2019.
