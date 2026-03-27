from __future__ import annotations

import argparse
import importlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
except ImportError:
    rasterio = None
    Resampling = None
    rasterize = None
    from_bounds = None
    reproject = None

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


RASTER_LABEL_SUFFIXES = {".tif", ".tiff"}
VECTOR_LABEL_SUFFIXES = {".shp", ".geojson", ".gpkg"}
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


@dataclass
class ReferenceGrid:
    crs: Any
    transform: Any
    height: int
    width: int


@dataclass
class SceneAssets:
    scene_root: str
    split: str
    event_id: str
    scene_date: str
    sentinel2_bands: list[str]
    sentinel2_paths: list[str]
    sentinel1_path: str | None
    dem_path: str | None
    rainfall_path: str | None
    soil_moisture_path: str | None
    label_paths: list[str]
    label_merge_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw geospatial scene folders into tiled tensors for training."
    )
    parser.add_argument("--scene-root", required=True, help="Scene folder under datasets/, for example datasets/2024-12-11.")
    parser.add_argument("--output-root", default="data", help="Root directory for processed outputs.")
    parser.add_argument("--split", default="train", help="Output split name, for example train or val.")
    parser.add_argument("--event-id", required=True, help="Event or domain id, for example wayanad_2024.")
    parser.add_argument(
        "--scene-date",
        help="Scene date in YYYY-MM-DD. If omitted, the script tries to parse it from the scene-root name.",
    )
    parser.add_argument(
        "--label-path",
        action="append",
        default=None,
        help="Raster or vector landslide label path. Repeat the flag to combine multiple layers.",
    )
    parser.add_argument(
        "--label-merge-mode",
        choices=["union", "intersection"],
        default="union",
        help="How to combine multiple label layers. 'union' keeps any positive pixel, 'intersection' keeps only overlap.",
    )
    parser.add_argument(
        "--rainfall-path",
        help="Optional rainfall NetCDF path. If omitted, the script looks under 'Rainfall Data/' in the scene root.",
    )
    parser.add_argument(
        "--soil-moisture-path",
        help="Optional soil moisture tif or zip path. If omitted, the script looks under 'Soil_moisture/' in the scene root.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Square tile size in pixels on the common reference grid.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Sliding-window stride in pixels.",
    )
    parser.add_argument(
        "--sentinel2-bands",
        nargs="*",
        default=None,
        help="Optional subset of Sentinel-2 bands to include. Defaults to all available bands in the folder.",
    )
    parser.add_argument(
        "--skip-empty-targets",
        action="store_true",
        help="Skip tiles whose target mask is completely empty.",
    )
    parser.add_argument(
        "--output-format",
        choices=["npy", "pt"],
        default="npy",
        help="Tile storage format. Use 'npy' in the geospatial env and 'pt' only when torch works there.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_geospatial_stack(label_paths: list[str], rainfall_path: str | None) -> None:
    _require(
        rasterio is not None and Resampling is not None and reproject is not None,
        "prepare_tiles.py requires rasterio. Install the packages listed in requirements.txt.",
    )
    if rainfall_path:
        _require(
            xr is not None and from_bounds is not None,
            "Rainfall NetCDF preprocessing requires xarray. Install requirements.txt.",
        )
    if any(Path(label_path).suffix.lower() in VECTOR_LABEL_SUFFIXES for label_path in label_paths):
        _require(
            gpd is not None and rasterize is not None,
            "Vector labels require geopandas and rasterio.features. Install requirements.txt.",
        )


def _parse_date(value: str) -> date:
    if "-" in value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.strptime(value, "%Y%m%d").date()


def _infer_scene_date(scene_root: Path, explicit_date: str | None) -> date:
    if explicit_date:
        return _parse_date(explicit_date)

    match = DATE_PATTERN.search(scene_root.name)
    if not match:
        raise ValueError(
            "Could not infer scene date from scene-root. Pass --scene-date explicitly."
        )
    return _parse_date(match.group(1))


def _find_first_file(folder: Path, suffixes: tuple[str, ...]) -> Path | None:
    if not folder.exists():
        return None
    matches = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )
    return matches[0] if matches else None


def _discover_sentinel2(scene_root: Path, requested_bands: list[str] | None) -> tuple[list[str], list[Path]]:
    s2_dir = scene_root / "Sentinel-2"
    if not s2_dir.exists():
        raise ValueError(f"Missing Sentinel-2 folder at {s2_dir}.")

    band_paths: dict[str, Path] = {}
    for path in sorted(s2_dir.glob("*.tif")):
        band_paths[path.stem.upper()] = path

    if requested_bands:
        missing = [band for band in requested_bands if band.upper() not in band_paths]
        if missing:
            raise ValueError(f"Missing requested Sentinel-2 bands: {missing}")
        ordered_bands = [band.upper() for band in requested_bands]
    else:
        ordered_bands = sorted(band_paths)

    return ordered_bands, [band_paths[band] for band in ordered_bands]


def _discover_scene_assets(args: argparse.Namespace) -> tuple[SceneAssets, list[Path]]:
    scene_root = Path(args.scene_root)
    if not scene_root.exists():
        raise ValueError(f"Scene root does not exist: {scene_root}")

    scene_date = _infer_scene_date(scene_root, args.scene_date)
    s2_bands, s2_paths = _discover_sentinel2(scene_root, args.sentinel2_bands)
    sentinel1_path = _find_first_file(scene_root / "Sentinel-1", (".tif", ".tiff"))
    dem_path = _find_first_file(scene_root / "DEM", (".tif", ".tiff"))

    rainfall_path = Path(args.rainfall_path) if args.rainfall_path else _find_first_file(
        scene_root / "Rainfall Data", (".nc",)
    )
    soil_moisture_path = Path(args.soil_moisture_path) if args.soil_moisture_path else _find_first_file(
        scene_root / "Soil_moisture", (".zip", ".tif", ".tiff")
    )
    label_paths = [str(Path(path).resolve()) for path in args.label_path] if args.label_path else []

    assets = SceneAssets(
        scene_root=str(scene_root.resolve()),
        split=args.split,
        event_id=args.event_id,
        scene_date=scene_date.isoformat(),
        sentinel2_bands=s2_bands,
        sentinel2_paths=[str(path.resolve()) for path in s2_paths],
        sentinel1_path=str(sentinel1_path.resolve()) if sentinel1_path else None,
        dem_path=str(dem_path.resolve()) if dem_path else None,
        rainfall_path=str(rainfall_path.resolve()) if rainfall_path else None,
        soil_moisture_path=str(soil_moisture_path.resolve()) if soil_moisture_path else None,
        label_paths=label_paths,
        label_merge_mode=args.label_merge_mode,
    )
    return assets, s2_paths


def _reference_grid_from_raster(path: Path) -> ReferenceGrid:
    with rasterio.open(path) as src:
        return ReferenceGrid(
            crs=src.crs,
            transform=src.transform,
            height=src.height,
            width=src.width,
        )


def _reproject_raster(path: Path, reference: ReferenceGrid, resampling_method) -> np.ndarray:
    with rasterio.open(path) as src:
        destination = np.zeros((src.count, reference.height, reference.width), dtype=np.float32)
        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                resampling=resampling_method,
            )
    return destination


def _select_xarray_variable(dataset) -> Any:
    candidate_names = []
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if {"lat", "lon"} <= dims or {"latitude", "longitude"} <= dims:
            candidate_names.append(name)
        elif len(data_array.dims) >= 2:
            candidate_names.append(name)

    if not candidate_names:
        raise ValueError("Could not find a usable rainfall variable in the NetCDF file.")

    return dataset[candidate_names[0]]


def _select_rainfall_slice(path: Path, scene_date: date, reference: ReferenceGrid) -> np.ndarray:
    dataset = xr.open_dataset(path)
    try:
        data_array = _select_xarray_variable(dataset)

        time_name = next((name for name in ("time", "day", "date") if name in data_array.dims), None)
        lat_name = next((name for name in ("lat", "latitude", "y") if name in data_array.coords), None)
        lon_name = next((name for name in ("lon", "longitude", "x") if name in data_array.coords), None)
        if lat_name is None or lon_name is None:
            raise ValueError("Rainfall NetCDF must contain latitude and longitude coordinates.")

        if time_name:
            data_array = data_array.sel({time_name: np.datetime64(scene_date)}, method="nearest")

        for dim in list(data_array.dims):
            if dim not in {lat_name, lon_name}:
                data_array = data_array.isel({dim: 0})

        rainfall = np.asarray(data_array.values, dtype=np.float32)
        lats = np.asarray(data_array[lat_name].values, dtype=np.float32)
        lons = np.asarray(data_array[lon_name].values, dtype=np.float32)

        if rainfall.ndim != 2:
            raise ValueError(f"Expected a 2D rainfall slice, got shape {rainfall.shape}.")

        if lats[0] < lats[-1]:
            lats = lats[::-1]
            rainfall = rainfall[::-1, :]

        src_transform = from_bounds(
            float(lons.min()),
            float(lats.min()),
            float(lons.max()),
            float(lats.max()),
            rainfall.shape[1],
            rainfall.shape[0],
        )
        destination = np.zeros((1, reference.height, reference.width), dtype=np.float32)
        reproject(
            source=rainfall,
            destination=destination[0],
            src_transform=src_transform,
            src_crs="EPSG:4326",
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.bilinear,
        )
        return destination
    finally:
        dataset.close()


def _parse_soil_moisture_dates(name: str) -> tuple[date, date] | None:
    matches = re.findall(r"(\d{8})", name)
    if len(matches) < 2:
        return None
    return _parse_date(matches[0]), _parse_date(matches[1])


def _select_soil_moisture_tif(path: Path, scene_date: date, cache_dir: Path) -> Path:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return path

    with zipfile.ZipFile(path) as archive:
        tif_names = [name for name in archive.namelist() if name.lower().endswith(".tif")]
        if not tif_names:
            raise ValueError(f"No tif files found inside soil moisture archive {path}.")

        def score(name: str) -> int:
            date_range = _parse_soil_moisture_dates(name)
            if not date_range:
                return 10**9
            start_date, end_date = date_range
            if start_date <= scene_date <= end_date:
                return 0
            midpoint = start_date + (end_date - start_date) / 2
            return abs((midpoint - scene_date).days)

        selected_name = min(tif_names, key=score)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / Path(selected_name).name
        if not output_path.exists():
            archive.extract(selected_name, path=cache_dir)
            extracted_path = cache_dir / selected_name
            if extracted_path != output_path:
                output_path.write_bytes(extracted_path.read_bytes())
        return output_path


def _load_label(path: Path, reference: ReferenceGrid) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in RASTER_LABEL_SUFFIXES:
        mask = _reproject_raster(path, reference, Resampling.nearest)
        return (mask[:1] > 0).astype(np.float32)

    if suffix in VECTOR_LABEL_SUFFIXES:
        geodata = gpd.read_file(path).to_crs(reference.crs)
        mask = rasterize(
            ((geom, 1) for geom in geodata.geometry if geom is not None and not geom.is_empty),
            out_shape=(reference.height, reference.width),
            transform=reference.transform,
            fill=0,
            dtype="uint8",
        )
        return mask[None].astype(np.float32)

    raise ValueError(f"Unsupported label format: {path.suffix}")


def _load_labels(paths: list[str], reference: ReferenceGrid, merge_mode: str) -> np.ndarray | None:
    if not paths:
        return None

    masks = [_load_label(Path(path), reference) for path in paths]
    combined = masks[0].copy()
    for mask in masks[1:]:
        if merge_mode == "union":
            combined = np.maximum(combined, mask)
        elif merge_mode == "intersection":
            combined = np.minimum(combined, mask)
        else:
            raise ValueError(f"Unsupported label_merge_mode: {merge_mode}")

    return combined.astype(np.float32)


def _stack_modalities(assets: SceneAssets, s2_paths: list[Path], reference: ReferenceGrid, cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    channel_arrays: list[np.ndarray] = []
    channel_names: list[str] = []

    for band_name, path in zip(assets.sentinel2_bands, s2_paths):
        channel_arrays.append(_reproject_raster(path, reference, Resampling.bilinear))
        channel_names.append(f"sentinel2_{band_name.lower()}")

    if assets.sentinel1_path:
        s1 = _reproject_raster(Path(assets.sentinel1_path), reference, Resampling.bilinear)
        channel_arrays.append(s1)
        channel_names.extend(f"sentinel1_band_{index + 1}" for index in range(s1.shape[0]))

    if assets.dem_path:
        dem = _reproject_raster(Path(assets.dem_path), reference, Resampling.bilinear)
        channel_arrays.append(dem[:1])
        channel_names.append("dem")

    if assets.rainfall_path:
        rainfall = _select_rainfall_slice(Path(assets.rainfall_path), _parse_date(assets.scene_date), reference)
        channel_arrays.append(rainfall)
        channel_names.append("rainfall_current")

    if assets.soil_moisture_path:
        soil_path = _select_soil_moisture_tif(
            Path(assets.soil_moisture_path),
            _parse_date(assets.scene_date),
            cache_dir=cache_dir,
        )
        soil = _reproject_raster(soil_path, reference, Resampling.bilinear)
        channel_arrays.append(soil[:1])
        channel_names.append("soil_moisture_current")

    stacked = np.concatenate(channel_arrays, axis=0).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0), channel_names


def _iter_tile_offsets(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    if tile_size > height or tile_size > width:
        raise ValueError("tile_size is larger than the reference raster dimensions.")

    row_offsets = list(range(0, height - tile_size + 1, stride))
    col_offsets = list(range(0, width - tile_size + 1, stride))

    if row_offsets[-1] != height - tile_size:
        row_offsets.append(height - tile_size)
    if col_offsets[-1] != width - tile_size:
        col_offsets.append(width - tile_size)

    return [(top, left) for top in row_offsets for left in col_offsets]


def _save_tiles(
    image_stack: np.ndarray,
    label_mask: np.ndarray | None,
    *,
    output_root: Path,
    split: str,
    event_id: str,
    scene_date: str,
    tile_size: int,
    stride: int,
    skip_empty_targets: bool,
    output_format: str,
) -> int:
    images_dir = output_root / split / "images" / event_id
    targets_dir = output_root / split / "targets" / event_id
    images_dir.mkdir(parents=True, exist_ok=True)
    if label_mask is not None:
        targets_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for top, left in _iter_tile_offsets(image_stack.shape[1], image_stack.shape[2], tile_size, stride):
        image_tile = image_stack[:, top : top + tile_size, left : left + tile_size]
        target_tile = None
        if label_mask is not None:
            target_tile = label_mask[:, top : top + tile_size, left : left + tile_size]
            if skip_empty_targets and float(target_tile.max()) == 0.0:
                continue

        sample_id = f"{scene_date}_{top}_{left}"
        if output_format == "npy":
            np.save(images_dir / f"{sample_id}_image.npy", image_tile.astype(np.float32))
            if target_tile is not None:
                np.save(targets_dir / f"{sample_id}_mask.npy", target_tile.astype(np.float32))
        else:
            torch = importlib.import_module("torch")
            if not hasattr(torch, "from_numpy"):
                raise RuntimeError(
                    "Torch is not usable in this environment. Run prepare_tiles.py with --output-format npy."
                )
            torch.save(torch.from_numpy(image_tile.copy()), images_dir / f"{sample_id}_image.pt")
            if target_tile is not None:
                torch.save(torch.from_numpy(target_tile.copy()), targets_dir / f"{sample_id}_mask.pt")
        saved += 1

    return saved


def _write_metadata(
    assets: SceneAssets,
    output_root: Path,
    channel_names: list[str],
    tile_size: int,
    stride: int,
    saved_tiles: int,
    output_format: str,
) -> None:
    metadata_dir = output_root / assets.split / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(assets)
    payload["channel_names"] = channel_names
    payload["tile_size"] = tile_size
    payload["stride"] = stride
    payload["saved_tiles"] = saved_tiles
    payload["output_format"] = output_format

    metadata_path = metadata_dir / f"{assets.event_id}.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    assets, s2_paths = _discover_scene_assets(args)
    _require_geospatial_stack(assets.label_paths, assets.rainfall_path)

    reference = _reference_grid_from_raster(s2_paths[0])
    output_root = Path(args.output_root)
    cache_dir = output_root / ".cache" / assets.event_id

    image_stack, channel_names = _stack_modalities(
        assets=assets,
        s2_paths=s2_paths,
        reference=reference,
        cache_dir=cache_dir,
    )
    label_mask = _load_labels(
        assets.label_paths,
        reference,
        merge_mode=assets.label_merge_mode,
    )

    saved_tiles = _save_tiles(
        image_stack=image_stack,
        label_mask=label_mask,
        output_root=output_root,
        split=assets.split,
        event_id=assets.event_id,
        scene_date=assets.scene_date,
        tile_size=args.tile_size,
        stride=args.stride,
        skip_empty_targets=args.skip_empty_targets,
        output_format=args.output_format,
    )
    _write_metadata(
        assets=assets,
        output_root=output_root,
        channel_names=channel_names,
        tile_size=args.tile_size,
        stride=args.stride,
        saved_tiles=saved_tiles,
        output_format=args.output_format,
    )

    print(
        f"prepared_tiles split={assets.split} event_id={assets.event_id} "
        f"channels={len(channel_names)} tiles={saved_tiles}"
    )


if __name__ == "__main__":
    main()
