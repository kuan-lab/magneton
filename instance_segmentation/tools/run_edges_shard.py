# -*- coding: utf-8 -*-
"""SLURM array worker for pass 2 (region-graph edges)."""
import argparse
import gc

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.stages.edges_stage import edges_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--indices", required=True, type=str)
    ap.add_argument("--workers", default=2, type=int)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    idx = [int(x) for x in args.indices.strip().split(",") if x.strip() != ""]
    edges_blocks(cfg, get_stage_config(cfg, "edges"), indices=idx, workers=args.workers)
    gc.collect()
    print("[DONE] edges shard finished.")


if __name__ == "__main__":
    main()
