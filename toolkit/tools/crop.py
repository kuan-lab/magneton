# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import tifffile as tiff
import h5py
from cloudvolume import CloudVolume
from magneton.toolkit.utils.config import load_config


def _detect_filetype(path):
    """Detect file type from path string.

    Returns 'tif', 'h5', 'precomputed', or raises ValueError.
    """
    if path.startswith("file://"):
        return "precomputed"
    lower = path.lower()
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        return "tif"
    if lower.endswith(".h5") or lower.endswith(".hdf5"):
        return "h5"
    raise ValueError(
        f"Unsupported file format: {path}\n"
        "Supported: .tif/.tiff, .h5/.hdf5, or file:// (precomputed)"
    )


def _get_volume_shape(path, filetype, h5_key="vol0"):
    """Get volume shape as (Z, Y, X) without loading data."""
    if filetype == "tif":
        with tiff.TiffFile(path) as t:
            series = t.series[0]
            shape = series.shape
    elif filetype == "h5":
        with h5py.File(path, "r") as f:
            shape = f[h5_key].shape
    elif filetype == "precomputed":
        vol = CloudVolume(path, mip=0, fill_missing=True)
        # vol.shape is (X, Y, Z, C) — return (Z, Y, X)
        shape = (vol.shape[2], vol.shape[1], vol.shape[0])
    return shape


def _check_bounds(coords, shape):
    """Validate that crop coords are within the volume boundary.

    Args:
        coords: [z1, z2, y1, y2, x1, x2]
        shape: volume shape as (Z, Y, X) (or higher-dim, last 3 are Z,Y,X)

    Raises:
        ValueError if any coord is out of bounds.
    """
    z1, z2, y1, y2, x1, x2 = coords
    # Use last 3 dims in case of (C, Z, Y, X)
    sz, sy, sx = shape[-3], shape[-2], shape[-1]

    errors = []
    if z1 < 0 or z1 >= sz:
        errors.append(f"z1={z1} out of range [0, {sz})")
    if z2 < 0 or z2 > sz:
        errors.append(f"z2={z2} out of range [0, {sz}]")
    if y1 < 0 or y1 >= sy:
        errors.append(f"y1={y1} out of range [0, {sy})")
    if y2 < 0 or y2 > sy:
        errors.append(f"y2={y2} out of range [0, {sy}]")
    if x1 < 0 or x1 >= sx:
        errors.append(f"x1={x1} out of range [0, {sx})")
    if x2 < 0 or x2 > sx:
        errors.append(f"x2={x2} out of range [0, {sx}]")
    if z1 >= z2:
        errors.append(f"z1={z1} >= z2={z2} (empty range)")
    if y1 >= y2:
        errors.append(f"y1={y1} >= y2={y2} (empty range)")
    if x1 >= x2:
        errors.append(f"x1={x1} >= x2={x2} (empty range)")

    if errors:
        raise ValueError(
            f"[ERROR] Crop coords out of bounds for volume shape {shape}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _read_crop(path, filetype, coords, h5_key="vol0"):
    """Read a cropped ROI from the given path.

    Args:
        path: input file path
        filetype: one of 'tif', 'h5', 'precomputed'
        coords: [z1, z2, y1, y2, x1, x2]
        h5_key: dataset key for h5 files

    Returns:
        numpy array in (Z, Y, X) order
    """
    z1, z2, y1, y2, x1, x2 = coords

    # Bounds check before loading data
    shape = _get_volume_shape(path, filetype, h5_key=h5_key)
    print(f"[INFO] Volume shape (ZYX): {shape}")
    _check_bounds(coords, shape)

    if filetype == "tif":
        data = tiff.imread(path)
        print(f"[INFO] Read tif: shape={data.shape}, dtype={data.dtype}")
        cropped = data[z1:z2, y1:y2, x1:x2]

    elif filetype == "h5":
        with h5py.File(path, "r") as f:
            ds = f[h5_key]
            print(f"[INFO] Read h5 key='{h5_key}': shape={ds.shape}, dtype={ds.dtype}")
            cropped = ds[z1:z2, y1:y2, x1:x2]

    elif filetype == "precomputed":
        vol = CloudVolume(path, mip=0, fill_missing=True)
        print(f"[INFO] Read precomputed: shape={vol.shape}, dtype={vol.dtype}")
        # CloudVolume indexing is (X, Y, Z)
        data = vol[x1:x2, y1:y2, z1:z2]
        # data shape is (X, Y, Z, C) — squeeze channel and transpose to (Z, Y, X)
        data = np.squeeze(data, axis=-1) if data.shape[-1] == 1 else data
        if data.ndim == 3:
            cropped = np.transpose(data, (2, 1, 0))  # (X,Y,Z) -> (Z,Y,X)
        else:
            # multi-channel: (X, Y, Z, C) -> (C, Z, Y, X)
            cropped = np.transpose(data, (3, 2, 1, 0))

    print(f"[INFO] Cropped ROI shape: {cropped.shape}, dtype={cropped.dtype}")
    return cropped


def _write_crop(data, path, filetype, h5_key="vol0", resolution=None):
    """Write cropped data to the given path.

    Args:
        data: numpy array in (Z, Y, X) order
        path: output file path
        filetype: one of 'tif', 'h5', 'precomputed'
        h5_key: dataset key for h5 files
        resolution: [x_res, y_res, z_res] for precomputed output
    """
    if filetype == "tif":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tiff.imwrite(path, data)
        print(f"[INFO] Saved tif: {path}, shape={data.shape}")

    elif filetype == "h5":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset(h5_key, data=data)
        print(f"[INFO] Saved h5: {path}, key='{h5_key}', shape={data.shape}")

    elif filetype == "precomputed":
        if resolution is None:
            resolution = [1, 1, 1]
            print("[WARN] No resolution specified for precomputed output, using [1,1,1]")

        if data.ndim == 3:
            volume_size = [data.shape[2], data.shape[1], data.shape[0]]  # (X, Y, Z)
            num_channels = 1
        else:
            volume_size = [data.shape[3], data.shape[2], data.shape[1]]
            num_channels = data.shape[0]

        info = CloudVolume.create_new_info(
            num_channels=num_channels,
            layer_type="segmentation" if np.issubdtype(data.dtype, np.integer) else "image",
            data_type=str(data.dtype),
            encoding="raw",
            resolution=resolution,
            voxel_offset=[0, 0, 0],
            chunk_size=[128, 128, 64],
            volume_size=volume_size,
        )

        vol = CloudVolume(path, info=info, compress=True)
        vol.commit_info()
        vol.commit_provenance()

        if data.ndim == 3:
            vol[:, :, :] = np.transpose(data, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
        else:
            vol[:, :, :, :] = np.transpose(data, (3, 2, 1, 0))  # (C,Z,Y,X) -> (X,Y,Z,C)

        print(f"[INFO] Saved precomputed: {path}, volume_size={volume_size}")


def _crop_volume(input_path, output_path, coords, h5_key="vol0", resolution=None):
    """Crop a region from input and save to output."""
    in_type = _detect_filetype(input_path)
    out_type = _detect_filetype(output_path)
    print(f"[INFO] Input:  {input_path} ({in_type})")
    print(f"[INFO] Output: {output_path} ({out_type})")
    print(f"[INFO] Coords: z=[{coords[0]}:{coords[1]}], y=[{coords[2]}:{coords[3]}], x=[{coords[4]}:{coords[5]}]")

    data = _read_crop(input_path, in_type, coords, h5_key=h5_key)
    _write_crop(data, output_path, out_type, h5_key=h5_key, resolution=resolution)
    print("[INFO] Crop completed.")


def crop_volume(cfg):
    """Entry point from toolkit main.py dispatcher."""
    crop_cfg = cfg["crop"]
    input_path = crop_cfg["input"]
    output_path = crop_cfg["output"]
    coords = crop_cfg["coords"]
    h5_key = crop_cfg.get("h5_key", "vol0")
    resolution = crop_cfg.get("resolution", None)
    _crop_volume(input_path, output_path, coords, h5_key=h5_key, resolution=resolution)


def main():
    parser = argparse.ArgumentParser(description="Crop a region from a volume (tif/h5/precomputed).")
    parser.add_argument("--config", default="config_crop.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    crop_volume(cfg)


if __name__ == "__main__":
    main()
