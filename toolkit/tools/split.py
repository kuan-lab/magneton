import os
import numpy as np
import skimage.io as io
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import argparse
from fractions import Fraction
from magneton.toolkit.utils.config import load_config, load_global_config_path

try:
    from cloudvolume import CloudVolume
except ImportError:
    CloudVolume = None


def _split_volume(path, save_path='', chunk_size=[512, 512, 512], overlap=[64, 64, 64], mip=0):
    """
    Split a 3D/4D volume (TIFF or precomputed) into smaller overlapping chunks.

    Args:
        path (str): Path to input (either .tif or precomputed dataset, e.g., file://...).
        save_path (str): Directory to save output TIFF chunks.
        chunk_size (list[int]): [z, y, x] chunk size.
        overlap (list[int]): [z, y, x] overlap in voxels.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # -------------------------------------
    # Detect input type
    # -------------------------------------
    is_precomputed = path.startswith("gs://") or path.startswith("file://") or path.startswith("precomputed://")

    # -------------------------------------
    # Load volume metadata
    # -------------------------------------
    if is_precomputed:
        if CloudVolume is None:
            raise ImportError("CloudVolume not installed. Please `pip install cloud-volume` first.")

        print(f"[INFO] Loading precomputed volume: {path}")
        vol = CloudVolume(path, mip=mip, progress=False)
        vol_shape = vol.volume_size[::-1]  # convert (x,y,z) -> (z,y,x)
        ndim = 4
        print(f"[INFO] Volume shape: {vol_shape}")

    else:
        print(f"[INFO] Loading TIFF: {path}")
        vol = io.imread(path)
        ndim = vol.ndim
        print(f"[INFO] TIFF shape: {vol.shape}")

        if ndim == 3:
            vol_shape = vol.shape
        elif ndim == 4:
            vol_shape = vol.shape[1:]  #  (C,Z,Y,X)
        else:
            raise ValueError("Only 3D or 4D TIFF supported.")

    # -------------------------------------
    # Compute chunk grid
    # -------------------------------------
    z_dim, y_dim, x_dim = vol_shape
    z_size, y_size, x_size = chunk_size
    z_ov, y_ov, x_ov = overlap

    def make_ranges(size, block, ov):
        starts = []
        start = 0
        while start < size:
            end = min(start + block, size)
            starts.append((start, end))
            if end == size:
                break
            start = end - ov
        return starts

    z_ranges = make_ranges(z_dim, z_size, z_ov)
    y_ranges = make_ranges(y_dim, y_size, y_ov)
    x_ranges = make_ranges(x_dim, x_size, x_ov)

    print(f"[INFO] Z={len(z_ranges)}, Y={len(y_ranges)}, X={len(x_ranges)} chunks total.")

    # -------------------------------------
    # Iterate over chunks
    # -------------------------------------
    chunk_idx = 0
    for zi, (zs, ze) in enumerate(z_ranges):
        for yi, (ys, ye) in enumerate(y_ranges):
            for xi, (xs, xe) in enumerate(x_ranges):
                if is_precomputed:
                    # CloudVolume expects (x,y,z)
                    bbox = np.s_[
                        xs:xe,
                        ys:ye,
                        zs:ze,
                    ]
                    chunk = np.asarray(vol[bbox])
                    # (Z,Y,X) ordering
                    # chunk = np.transpose(chunk, (2, 1, 0))
                else:
                    if ndim == 3:
                        chunk = vol[zs:ze, ys:ye, xs:xe]
                    else:  # 4D
                        chunk = vol[:, zs:ze, ys:ye, xs:xe]

                fname = f"chunk_z{zi:02d}_y{yi:02d}_x{xi:02d}.tif"
                out_path = os.path.join(save_path, fname)
                tiff.imwrite(out_path, chunk)
                chunk_idx += 1

                print(f"[INFO] Saved {fname}, shape={chunk.shape}")
    print(f"\nDone. {chunk_idx} chunks saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Split volume to chunks.")
    parser.add_argument("--config", default="config_split.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    input = cfg["split"]["input"]
    output = cfg["split"]["output"]
    chunk_size = cfg["split"]["chunk_size"]
    overlap = cfg["split"]["overlap"]
    chunk_size = [int(Fraction(val)) for val in chunk_size]
    overlap = [int(Fraction(val)) for val in overlap]
    mip = cfg["split"]["mip"]
    _split_volume(input, output, chunk_size, overlap, mip)


def split_volume(cfg):
    input = cfg["split"]["input"]
    output = cfg["split"]["output"]
    chunk_size = cfg["split"]["chunk_size"]
    overlap = cfg["split"]["overlap"]
    chunk_size = [int(Fraction(val)) for val in chunk_size]
    overlap = [int(Fraction(val)) for val in overlap]
    mip = cfg["split"]["mip"]
    _split_volume(input, output, chunk_size, overlap, mip)
    

if __name__=="__main__":
    main()
    