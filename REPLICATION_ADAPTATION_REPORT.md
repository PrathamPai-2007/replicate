# Replication & Adaptation Report

## Title

Replication and Adaptation of EEGMoE for Pixel-Wise Landslide Mapping from Multimodal Geospatial Data

## 1. Objective

The objective of this project is to replicate the core architectural ideas of the EEGMoE paper,
`EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning`,
and adapt them to the task of landslide detection and mapping.

The target application is pixel-wise landslide segmentation from multimodal remote sensing and terrain data.
The competition statement asks that the Landslide Atlas be used as ground truth. In this project, atlas-derived
polygon layers were created and used as supervision for segmentation experiments.

## 2. Original Paper Context

The original EEGMoE paper is designed for EEG representation learning. Its central idea is to separate
domain-specific and domain-shared knowledge using a domain-decoupled mixture-of-experts architecture.

The paper includes two key routing branches:

- a specific-expert branch using top-k routing
- a shared-expert branch using soft routing across shared experts

The combined expert output is then used for downstream representation learning.

## 3. What Was Replicated

The current implementation reproduces the key routing logic of the paper in a satellite-image setting.

Implemented components:

- patch-to-token embedding for image tiles
- `SpecificMoE` branch with top-k expert selection
- `SharedMoE` branch with soft routing over all shared experts
- additive fusion of specific and shared expert outputs
- supervised downstream head for segmentation

Code mapping:

- [satellite_ssmoe.py](./satellite_ssmoe.py): patch embedding and MoE routing
- [model.py](./model.py): segmentation wrapper
- [train.py](./train.py): supervised optimization and checkpointing

In practical terms, the following paper-inspired pattern is preserved:

1. convert input into token features
2. route tokens through a domain-specific expert mixture
3. route tokens through a shared expert mixture
4. fuse both outputs
5. decode into a downstream task prediction

## 4. What Was Adapted For Landslide Mapping

The original EEG setting was adapted in the following ways:

- EEG channels were replaced by multimodal geospatial raster channels
- sequence-style EEG inputs were replaced by image tiles
- representation learning output was replaced by a segmentation head
- atlas-derived landslide polygons were used as training supervision

The current input schema is a 17-channel multimodal tile:

- 12 Sentinel-2 channels
- 2 Sentinel-1 channels
- 1 DEM channel
- 1 rainfall channel
- 1 soil moisture channel

The downstream task is binary segmentation:

- `1` = landslide
- `0` = non-landslide

## 5. Ground Truth Adaptation

The Landslide Atlas PDF was not directly machine-readable, so the supervision pipeline required adaptation.

The workflow was:

1. use the atlas PDF as the official reference source
2. manually digitize landslide polygons into vector layers
3. store those labels as `.gpkg` files
4. rasterize the polygons onto the common preprocessing grid during tile generation

Multiple label layers were supported in preprocessing and combined with `union` merging when appropriate.

Related file:

- [ATLAS_LABEL_PLAN.md](./ATLAS_LABEL_PLAN.md)

## 6. Data Pipeline

The preprocessing pipeline is implemented in [prepare_tiles.py](./prepare_tiles.py).

Inputs supported:

- Sentinel-2 GeoTIFF bands
- Sentinel-1 GeoTIFF raster
- DEM raster
- rainfall NetCDF
- soil moisture TIFF or zipped TIFF
- raster or vector label layers

The preprocessing steps are:

1. discover raw modality files for a scene
2. choose a common reference grid
3. reproject and resample each modality
4. stack modalities into a `[channels, height, width]` tensor
5. rasterize label polygons into binary masks
6. extract training tiles
7. write image tensors, mask tensors, and metadata JSON

Two tiling regimes were used:

- baseline tiles: `tile_size=64`, `stride=64`
- dense tiles: `tile_size=64`, `stride=32`

## 7. Processed Dataset Used In The Current Best Run

Current strongest experiment setup:

- train split: `data_dense/train`
- val split: `data_dense/val`
- task: segmentation
- tile size: `64 x 64`
- stride: `32`
- channels: `17`

Dataset summaries:

- train sample count: `24`
- val sample count: `24`
- train positive-pixel fraction: `0.087890625`
- val positive-pixel fraction: `0.087890625`

Summary files:

- [data_dense_train_summary.json](./reports/data_dense_train_summary.json)
- [data_dense_val_summary.json](./reports/data_dense_val_summary.json)

## 8. Model And Training Setup

The supervised segmentation setup used:

- model dimension: `128`
- patch size: `8`
- specific experts: `4`
- shared experts: `2`
- top-k routing: `2`
- batch size: `2`
- epochs: `20`
- positive class weight: `8`

Loss:

- BCEWithLogitsLoss with positive weighting
- Dice loss
- final segmentation loss = BCE + Dice

This combination was chosen because the target masks are sparse and class imbalance is substantial.

## 9. Results

Three main supervised stages were observed locally:

1. initial weak-label baseline with very small tile count
2. class-weighted training with `positive-class-weight=8`
3. dense tiling with merged label layers

Current best local result:

- validation mean IoU: `0.3966`
- threshold: `0.5`
- validation sample count: `24`

Best checkpoint:

- [landslide_seg_dense_w8.pt](./checkpoints/landslide_seg_dense_w8.pt)

Prediction summary:

- [summary.json](./predictions/dense_val_review/summary.json)

This result shows that:

- denser tiling improved learning
- merged label layers improved supervision coverage
- imbalance-aware loss weighting helped prevent all-background predictions

## 10. Qualitative Review

A qualitative inference pipeline was added through [predict.py](./predict.py).

For each validation tile, the review output includes:

- RGB preview
- target mask
- probability map
- binary prediction
- MoE specific-expert map

These review outputs are useful because the summary metric alone does not explain whether the model:

- localizes landslides correctly
- over-predicts outside the target
- misses parts of the scar
- relies on one expert disproportionately

## 11. Replication Status: Honest Assessment

This project should be described as a strong first replication-and-adaptation baseline, not a complete
paper-faithful reimplementation of every EEGMoE training component.

What is faithfully adapted:

- the specific/shared MoE routing idea
- top-k specific routing
- shared soft routing
- additive fusion of expert outputs
- token-wise expert processing

What is still incomplete relative to the original paper:

- paper-style self-supervised pretraining
- full representation-learning objective from the EEG setting
- a richer domain definition beyond the current event/data organization
- broader multi-domain evaluation

This distinction is important for the report and should be stated clearly to judges.

## 12. Why The Adaptation Is Reasonable

Although EEG and landslide mapping are very different tasks, the adaptation is reasonable because both can be
cast into a feature-mixing problem where different expert groups capture different structures.

In this adaptation:

- specific experts can model more specialized local feature patterns
- shared experts can capture cross-scene structure common to landslide regions
- multimodal raster channels provide the remote-sensing equivalent of multi-source structured input

The model therefore preserves the conceptual value of domain-decoupled expert routing, even though the data
modality is different.

## 13. Limitations

Current limitations include:

- labels are weak and atlas-derived rather than official GIS inventory layers
- train and validation coverage are still small
- the current validation set is not yet a large independent benchmark
- the project has not yet implemented full-scene stitched geospatial export
- the self-supervised stage from EEGMoE remains future work

These limitations should be stated explicitly rather than hidden.

## 14. Next Steps

The next strongest improvements are:

1. expand the atlas-derived vector labels
2. run broader event-wise experiments
3. tune prediction thresholds
4. add full-scene stitched inference and geospatial export
5. implement the self-supervised pretraining stage from the original paper

## 15. Conclusion

This project successfully establishes a working replication-inspired baseline of EEGMoE for landslide
segmentation. The MoE routing structure has been transferred into a multimodal satellite setting, a usable
atlas-derived supervision pipeline has been built, and the current best experiment reaches a validation
mean IoU of `0.3966` on the dense tile split.

The present system is best described as:

- a successful architecture adaptation baseline
- a meaningful first replication of the routing core
- a competition-ready experimental scaffold

The strongest future improvement would be to complete the missing self-supervised components and strengthen
the label inventory with more event coverage.
