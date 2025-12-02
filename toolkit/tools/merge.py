import os
import re
import h5py
import numpy as np
import tifffile as tiff
from skimage import io
import argparse
from fractions import Fraction

from magneton.toolkit.utils.config import load_config, load_global_config_path

try:
    from cloudvolume import CloudVolume
except ImportError:
    CloudVolume = None


def _merge_volume(
    chunk_path,
    save_path,
    chunk_size=[512, 512, 512],
    overlap=[64, 64, 64],
    ndim=4,
    save_as_tif=True,
    fill_missing=True,
):
    """
    Merge chunks into full 3D/4D volume (single-pass, robust).
    Auto-trims border blocks; optionally fills missing blocks with zeros.

    Args:
        chunk_path (str): Directory containing chunks.
        save_path (str): Output path for merged result.
        chunk_size (list[int]): [z, y, x] standard chunk size.
        overlap (list[int]): Overlap size.
        ndim (int): 3 for (Z,Y,X) or 4 for (C,Z,Y,X).
        save_as_tif (bool): Save merged TIFF to disk.
        fill_missing (bool): Fill missing blocks in grid with zeros.
    """
    pattern = re.compile(r"chunk_z(\d+)_y(\d+)_x(\d+)\.h5")
    chunk_files = [f for f in os.listdir(chunk_path) if f.endswith(".h5")]
    coords = []
    for f in chunk_files:
        m = pattern.match(f)
        if m:
            z, y, x = map(int, m.groups())
            coords.append((z, y, x, f))
    if not coords:
        raise RuntimeError("No valid chunk files found.")
    coords.sort()

    # Load one chunk for dtype inference
    with h5py.File(os.path.join(chunk_path, coords[0][3]), "r") as f:
        chunk = f["vol0"][:]
    dtype = chunk.dtype
    ch = chunk.shape[0] if ndim == 4 else None

    z_blocks = max(c[0] for c in coords) + 1
    y_blocks = max(c[1] for c in coords) + 1
    x_blocks = max(c[2] for c in coords) + 1
    z_size, y_size, x_size = chunk_size
    z_ov, y_ov, x_ov = overlap

    z_dim = z_blocks * (z_size - z_ov) + z_ov
    y_dim = y_blocks * (y_size - y_ov) + y_ov
    x_dim = x_blocks * (x_size - x_ov) + x_ov
    full_shape = (z_dim, y_dim, x_dim)

    print(f"[INFO] Expected grid {z_blocks}×{y_blocks}×{x_blocks}, full shape {full_shape}")

    if ndim == 3:
        full_vol = np.zeros(full_shape, dtype=dtype)
    else:
        full_vol = np.zeros((ch, *full_shape), dtype=dtype)

    # Convert coord list to dict for fast lookup
    chunk_dict = {(z, y, x): fname for (z, y, x, fname) in coords}

    # Iterate full grid (ensures missing detection)
    for zi in range(z_blocks):
        for yi in range(y_blocks):
            for xi in range(x_blocks):
                zs = zi * (z_size - z_ov)
                ys = yi * (y_size - y_ov)
                xs = xi * (x_size - x_ov)
                ze = zs + z_size
                ye = ys + y_size
                xe = xs + x_size

                ze = min(ze, full_shape[0])
                ye = min(ye, full_shape[1])
                xe = min(xe, full_shape[2])

                if (zi, yi, xi) not in chunk_dict:
                    if fill_missing:
                        print(f"[WARN] Missing chunk z{zi}_y{yi}_x{xi}, filled zeros.")
                    else:
                        continue
                    continue

                fname = chunk_dict[(zi, yi, xi)]
                with h5py.File(os.path.join(chunk_path, fname), "r") as f:
                    chunk = f["vol0"][:]
                if ndim == 4:
                    cz, cy, cx = chunk.shape[-3:]
                else:
                    cz, cy, cx = chunk.shape

                cz = min(cz, ze - zs)
                cy = min(cy, ye - ys)
                cx = min(cx, xe - xs)

                if ndim == 3:
                    full_vol[zs:zs+cz, ys:ys+cy, xs:xs+cx] = chunk[:cz, :cy, :cx]
                else:
                    full_vol[:, zs:zs+cz, ys:ys+cy, xs:xs+cx] = chunk[:, :cz, :cy, :cx]

                print(f"[INFO] Inserted chunk z{zi}_y{yi}_x{xi} → "
                      f"Z[{zs}:{zs+cz}] Y[{ys}:{ys+cy}] X[{xs}:{xs+cx}]")
    full_vol = _trim_zeros(full_vol)
    if save_as_tif:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tiff.imwrite(save_path, full_vol)
        print(f"[INFO] Saved merged volume: {save_path}")

    print("[INFO] Merge complete.")
    return full_vol

# --- Optional: remove zero padding at volume edges ---
def _trim_zeros(vol):
    """Trim zero-padding around nonzero content."""
    if vol.ndim == 4:  # (C,Z,Y,X)
        mask = np.any(vol != 0, axis=0)
    else:              # (Z,Y,X)
        mask = vol != 0

    z_mask = np.any(np.any(mask, axis=2), axis=1)
    y_mask = np.any(np.any(mask, axis=2), axis=0)
    x_mask = np.any(np.any(mask, axis=1), axis=0)

    z_min, z_max = np.where(z_mask)[0][[0, -1]]
    y_min, y_max = np.where(y_mask)[0][[0, -1]]
    x_min, x_max = np.where(x_mask)[0][[0, -1]]

    # +1 because slicing is exclusive at end
    if vol.ndim == 4:
        vol = vol[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    else:
        vol = vol[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    print(f"[INFO] Trimmed zero-padding → Z[{z_min}:{z_max+1}] "
          f"Y[{y_min}:{y_max+1}] X[{x_min}:{x_max+1}] "
          f"=> shape {vol.shape}")
    return vol

def main():
    parser = argparse.ArgumentParser(description="Split volume to chunks.")
    parser.add_argument("--config", default="config_merge.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    input = cfg["merge"]["input"]
    output = cfg["merge"]["output"]
    chunk_size = cfg["merge"]["chunk_size"]
    overlap = cfg["merge"]["overlap"]
    chunk_size = [int(Fraction(val)) for val in chunk_size]
    overlap = [int(Fraction(val)) for val in overlap]
    _merge_volume(input, output, chunk_size, overlap)


def merge_volume(cfg):
    input = cfg["merge"]["input"]
    output = cfg["merge"]["output"]
    chunk_size = cfg["merge"]["chunk_size"]
    overlap = cfg["merge"]["overlap"]
    chunk_size = [int(Fraction(val)) for val in chunk_size]
    overlap = [int(Fraction(val)) for val in overlap]
    _merge_volume(input, output, chunk_size, overlap)


if __name__=="__main__":
    main()