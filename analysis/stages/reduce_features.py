"""
Stage C — concat per-task parquet shards into the final morphometrics.parquet.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from magneton.analysis.config import load_config, get_stage_config, strip_file_prefix


def reduce_features(cfg: dict) -> str:
    paths = get_stage_config(cfg, "paths")
    out_dir = strip_file_prefix(paths["output"])
    partials_dir = os.path.join(out_dir, "stage2_partials")
    shards = sorted(glob.glob(os.path.join(partials_dir, "task_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no stage2_partials/task_*.parquet shards found in {partials_dir}")
    print(f"[analysis.reduce] reading {len(shards)} shards from {partials_dir}")

    parts = [pd.read_parquet(s) for s in shards]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("seg_id").reset_index(drop=True)
    out_path = os.path.join(out_dir, "morphometrics.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[analysis.reduce] wrote {len(df)} rows to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="analysis stage C — concat task shards")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    reduce_features(cfg)


if __name__ == "__main__":
    main()
