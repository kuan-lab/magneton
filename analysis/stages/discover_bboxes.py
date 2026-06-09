"""
Stage A — read a high-mip volume, find each mito's bbox via scipy.ndimage.find_objects,
pad halo at the discovery mip first, upscale to mip 0, write bboxes.parquet.

Run interactively (no SLURM); takes ~30 seconds on fib_c_mito_v3 at mip 2.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects

from magneton.analysis.config import load_config, get_stage_config, strip_file_prefix
from magneton.analysis.lib.precomputed_io import read_full, get_volume_specs


def discover_bboxes(cfg: dict) -> str:
    """Run stage A. Returns the path to the bboxes.parquet that was written."""
    paths = get_stage_config(cfg, "paths")
    disc  = get_stage_config(cfg, "discover")
    in_pc  = paths["input"]
    out_dir = strip_file_prefix(paths["output"])

    mip            = int(disc.get("mip", 2))
    halo_mipN      = int(disc.get("bbox_halo_mipN", 1))
    min_count_mipN = int(disc.get("min_voxel_count_mipN", 1))

    print(f"[analysis.discover] input:  {in_pc}")
    print(f"[analysis.discover] output: {out_dir}/bboxes.parquet")
    print(f"[analysis.discover] mip={mip}, halo_mipN=±{halo_mipN}, min_count_mipN={min_count_mipN}")

    # Specs at both mips so we can convert from discover-mip to mip-0 coords.
    specs_mip = get_volume_specs(in_pc, mip=mip)
    specs_0   = get_volume_specs(in_pc, mip=0)
    fX, fY, fZ = specs_mip.downsample_factor_xyz
    X0, Y0, Z0 = specs_0.shape_xyz
    print(f"[analysis.discover] mip-{mip} shape: {specs_mip.shape_xyz}, factor: ({fX},{fY},{fZ}), mip-0 shape: {specs_0.shape_xyz}")

    t0 = time.time()
    vol = read_full(in_pc, mip=mip)
    print(f"[analysis.discover] read mip-{mip} ({vol.nbytes / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

    t0 = time.time()
    # find_objects expects integer label array; returns a list indexed by (label-1)
    slices = find_objects(vol)
    print(f"[analysis.discover] find_objects: {sum(1 for s in slices if s is not None)} bboxes in {time.time() - t0:.1f}s")

    # All per-label voxel counts in ONE O(voxels) pass, instead of a per-label
    # vol[slc]==id sum (which is O(sum of bbox volumes) and dominates runtime at
    # fine mips — was the mip-0 bottleneck). bincount index = label value.
    t0 = time.time()
    counts = np.bincount(vol.reshape(-1))
    print(f"[analysis.discover] bincount counts in {time.time() - t0:.1f}s")

    rows = []
    for label_minus_1, slc in enumerate(slices):
        if slc is None:
            continue
        seg_id = label_minus_1 + 1
        # O(1) lookup — count at mip-N for this label
        count = int(counts[seg_id]) if seg_id < counts.shape[0] else 0
        if count < min_count_mipN:
            continue

        # Bbox at mip-N in XYZ
        x0_n, x1_n = int(slc[0].start), int(slc[0].stop)
        y0_n, y1_n = int(slc[1].start), int(slc[1].stop)
        z0_n, z1_n = int(slc[2].start), int(slc[2].stop)

        # 1) pad halo at the discovery mip
        x0_n -= halo_mipN; x1_n += halo_mipN
        y0_n -= halo_mipN; y1_n += halo_mipN
        z0_n -= halo_mipN; z1_n += halo_mipN
        # clip to mip-N bounds before upscale (cheaper, equivalent)
        Xn, Yn, Zn = specs_mip.shape_xyz
        x0_n = max(0, x0_n); x1_n = min(Xn, x1_n)
        y0_n = max(0, y0_n); y1_n = min(Yn, y1_n)
        z0_n = max(0, z0_n); z1_n = min(Zn, z1_n)

        # 2) upscale to mip-0
        x0 = x0_n * fX; x1 = x1_n * fX
        y0 = y0_n * fY; y1 = y1_n * fY
        z0 = z0_n * fZ; z1 = z1_n * fZ
        # 3) clip to mip-0 bounds
        x0 = max(0, x0); x1 = min(X0, x1)
        y0 = max(0, y0); y1 = min(Y0, y1)
        z0 = max(0, z0); z1 = min(Z0, z1)

        rows.append(dict(
            seg_id=int(seg_id),
            bbox_x0=int(x0), bbox_x1=int(x1),
            bbox_y0=int(y0), bbox_y1=int(y1),
            bbox_z0=int(z0), bbox_z1=int(z1),
            mipN_voxel_count=count,
        ))

    df = pd.DataFrame(rows, columns=[
        "seg_id", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "bbox_z0", "bbox_z1",
        "mipN_voxel_count",
    ])
    print(f"[analysis.discover] kept {len(df)} bboxes (filtered min_count_mipN >= {min_count_mipN})")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "bboxes.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[analysis.discover] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="analysis stage A — discover per-mito bboxes")
    ap.add_argument("--config", required=True, help="path to per-volume YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    discover_bboxes(cfg)


if __name__ == "__main__":
    main()
