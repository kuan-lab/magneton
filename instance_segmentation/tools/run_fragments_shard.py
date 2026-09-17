# -*- coding: utf-8 -*-
"""SLURM array worker for pass 1 (fragments). Mirrors tools/run_local_shard.py."""
import argparse
import gc

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.stages.fragments_stage import fragments_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--indices", required=True, type=str,
                    help="Comma-separated core indices, e.g. 0,1,2")
    ap.add_argument("--workers", default=2, type=int)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stage_cfg = get_stage_config(cfg, "fragments")
    idx = [int(x) for x in args.indices.strip().split(",") if x.strip() != ""]
    fragments_blocks(cfg, stage_cfg, indices=idx, workers=args.workers)
    gc.collect()
    print("[DONE] fragments shard finished.")


if __name__ == "__main__":
    main()
