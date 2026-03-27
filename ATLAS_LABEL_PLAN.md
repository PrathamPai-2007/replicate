# Atlas Label Plan

This project uses the EEGMoE architecture for pixel-wise landslide prediction.
The competition statement says the Landslide Atlas should be used as ground truth.

## What The Atlas PDF Gives Us

File:

- `pdfs/LandslideAtlas_new_2023.pdf`

Useful findings from the atlas:

- The atlas describes a large NRSC/ISRO landslide inventory database created as GIS polygons.
- Kerala is explicitly included in the atlas inventory summary.
- Puthumala, Wayanad district, Kerala is explicitly shown as a representative landslide example.
- The atlas says the inventory is available on the Bhuvan web GIS platform.

Atlas pages that matter for this competition:

- Page 34: Kerala inventory counts and mention of the Bhuvan inventory platform.
- Page 66: Figure 42, Puthumala region, Wayanad district, Kerala.
- Pages 80-81: Puthumala flow and UAV modeling references.

## What The Atlas PDF Does Not Give Us

The PDF itself is not a direct training label file.
It is a report and map product, not a machine-ready raster or vector layer.

Training needs one of these:

- raster mask: `.tif`
- vector polygons: `.shp`, `.geojson`, `.gpkg`

## Preferred Ground-Truth Strategy

Priority order:

1. Obtain the actual NRSC/Bhuvan GIS inventory polygons for Kerala/Wayanad/Puthumala.
2. If the GIS inventory cannot be obtained, digitize polygons from atlas and event maps manually.
3. Use those polygons to rasterize binary masks on the same grid as the model inputs.

## Expected Label Output

For training:

- `data/train/targets/<event_id>/<tile_id>_mask.npy`
- `data/val/targets/<event_id>/<tile_id>_mask.npy`

Mask meaning:

- `1` means landslide pixel
- `0` means non-landslide pixel

## Current Competition-Focused Workflow

1. Use the atlas PDF as the official reference for what counts as landslide ground truth.
2. Build machine-usable labels from atlas-backed polygons or manually digitized event maps.
3. Use `prepare_tiles.py` to align Sentinel-1, Sentinel-2, DEM, rainfall, soil moisture, and labels.
4. Train the segmentation model with `train.py`.
5. Export model predictions as atlas-style masks or polygons.

## Immediate Next Step

The single most valuable next action is to get a usable Kerala/Wayanad landslide polygon or mask layer.
If that is not available, manual digitization of Puthumala and Wayanad event extents becomes the fallback plan.
