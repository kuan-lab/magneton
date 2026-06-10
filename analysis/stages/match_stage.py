"""
Relational stage 1 — match instances across the co-registered volumes.

Reads a relational config (volumes / matching / output) and writes two tables to
the relational output dir:
    match_mito_to_bouton.parquet     (mito_seg_id, parent_bouton, sample_x/y/z)
    match_synapse_to_bouton.parquet  (synapse_seg_id, best_bouton, overlap_voxels,
                                      n_boutons_touched)

Both matchers are light (mito = two coarse-mip full reads; synapse = tiny per-
instance crops), so this runs fine on a login/interactive node.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from magneton.analysis.config import load_config, strip_file_prefix
from magneton.analysis.lib.matching import (
    match_mito_to_bouton,
    match_synapse_to_bouton,
)


def _vol(cfg: dict, name: str) -> dict:
    v = cfg.get("volumes", {}).get(name)
    if v is None:
        raise KeyError(f"relational config missing volumes.{name}")
    return v


def run_matching(cfg: dict) -> dict:
    out_dir = strip_file_prefix(cfg["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    mito   = _vol(cfg, "mito")
    bouton = _vol(cfg, "bouton")
    syn    = _vol(cfg, "synapse")
    match_cfg = cfg.get("matching", {})

    # --- mito → bouton (medoid-snapped centroid) ---
    sample_mip = int(match_cfg.get("mito_to_bouton", {}).get("sample_mip", 2))
    df_mb = match_mito_to_bouton(mito["pc"], bouton["pc"], mip=sample_mip)
    mb_path = os.path.join(out_dir, "match_mito_to_bouton.parquet")
    df_mb.to_parquet(mb_path, index=False)
    print(f"[match] wrote {mb_path}")

    # --- synapse → bouton (direct overlap, best bouton) ---
    syn_bbox_path = os.path.join(strip_file_prefix(syn["analysis_out"]), "bboxes.parquet")
    if not os.path.isfile(syn_bbox_path):
        raise FileNotFoundError(
            f"synapse bboxes.parquet not found at {syn_bbox_path}; "
            f"run the synapse per-volume analysis (discover) first"
        )
    syn_bboxes = pd.read_parquet(syn_bbox_path)
    df_sb = match_synapse_to_bouton(syn["pc"], bouton["pc"], syn_bboxes)
    sb_path = os.path.join(out_dir, "match_synapse_to_bouton.parquet")
    df_sb.to_parquet(sb_path, index=False)
    print(f"[match] wrote {sb_path}")

    return {"mito_to_bouton": mb_path, "synapse_to_bouton": sb_path}


def main():
    ap = argparse.ArgumentParser(description="relational stage 1 — cross-volume matching")
    ap.add_argument("--config", required=True, help="path to relational YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_matching(cfg)


if __name__ == "__main__":
    main()
