"""
Stage B — per-task worker. Reads bboxes.parquet, processes a row range,
writes one task_<k>.parquet shard.

Designed to run inside a SLURM array task (one task per invocation). For
single-process debugging, pass --range manually as `start,end`.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from magneton.analysis.config import load_config, get_stage_config, strip_file_prefix
from magneton.analysis.lib.precomputed_io import read_bbox, get_volume_specs
from magneton.analysis.lib.features import compute_all_features, feature_names


def _parse_range(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"--range must be 'start,end', got {s!r}")
    return int(parts[0]), int(parts[1])


def process_range(cfg: dict, start: int, end: int, task_id: int) -> str:
    paths = get_stage_config(cfg, "paths")
    feat_cfg = get_stage_config(cfg, "features")
    in_pc   = paths["input"]
    out_dir = strip_file_prefix(paths["output"])

    bboxes_path = os.path.join(out_dir, "bboxes.parquet")
    partials_dir = os.path.join(out_dir, "stage2_partials")
    Path(partials_dir).mkdir(parents=True, exist_ok=True)

    df_bb = pd.read_parquet(bboxes_path)
    if start < 0 or end > len(df_bb) or start >= end:
        raise IndexError(f"range [{start},{end}) out of bounds for {len(df_bb)} rows")
    rows = df_bb.iloc[start:end]
    print(f"[analysis.instance] task {task_id}: processing rows [{start},{end}) = {len(rows)} mitos")

    specs = get_volume_specs(in_pc, mip=0)
    vox_nm = tuple(float(v) for v in specs.voxel_nm)
    fnames = feature_names()

    out_records = []
    t0 = time.time()
    for i, r in enumerate(rows.itertuples(index=False)):
        seg_id = int(r.seg_id)
        bbox = (int(r.bbox_x0), int(r.bbox_x1),
                int(r.bbox_y0), int(r.bbox_y1),
                int(r.bbox_z0), int(r.bbox_z1))
        crop = read_bbox(in_pc, bbox, mip=0)
        mask = (crop == seg_id)
        if not mask.any():
            print(f"[analysis.instance] WARN seg_id {seg_id} mask empty inside bbox; skipping")
            continue
        feats = compute_all_features(mask, vox_nm, feat_cfg)
        rec = {"seg_id": seg_id}
        rec.update({k: feats.get(k, 0.0) for k in fnames})
        out_records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[analysis.instance]   {i + 1}/{len(rows)} ({(time.time() - t0):.1f}s elapsed)")

    df_out = pd.DataFrame(out_records, columns=["seg_id"] + fnames)
    out_path = os.path.join(partials_dir, f"task_{task_id:04d}.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"[analysis.instance] task {task_id}: wrote {len(df_out)} rows to {out_path} in {time.time() - t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="analysis stage B — per-mito feature worker")
    ap.add_argument("--config", required=True)
    ap.add_argument("--range", required=True, help="row range into bboxes.parquet, 'start,end'")
    ap.add_argument("--task-id", type=int, default=0, help="task id used as the output filename suffix")
    args = ap.parse_args()
    start, end = _parse_range(args.range)
    cfg = load_config(args.config)
    process_range(cfg, start, end, args.task_id)


if __name__ == "__main__":
    main()
