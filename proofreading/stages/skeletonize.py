"""
Stage A — skeletonize an instance segmentation into a WebKnossos NML.

Reads paths.seg + skeletonize_stage from the config, runs kimimaro per instance,
writes <output>/skeletons.nml. Upload it with /wk, correct in WebKnossos, then
run the expand stage on the corrected NML.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from magneton.proofreading.config import (
    load_config, get_stage_config, strip_file_prefix,
)
from magneton.proofreading.lib import skeleton_io


def skeletonize(cfg: dict):
    paths = get_stage_config(cfg, "paths")
    sk = get_stage_config(cfg, "skeletonize")
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_nml = os.path.join(out_dir, "skeletons.nml")

    t0 = time.time()
    seg, res_nm = skeleton_io.load_seg(
        paths["seg"], source=sk.get("source", "tif"),
        mip=sk.get("mip", 0), res_nm=sk.get("res_nm"),
    )
    print(f"[skeletonize] seg {seg.shape} res={res_nm}nm in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    skels = skeleton_io.skeletonize(
        seg, res_nm,
        dust_threshold=sk.get("dust_threshold", 100),
        parallel=sk.get("parallel", 8),
        parallel_chunk_size=sk.get("parallel_chunk_size", 25),
        fix_branching=sk.get("fix_branching", True),
        fix_borders=sk.get("fix_borders", True),
        teasar=sk.get("teasar") or None,
        postprocess=sk.get("postprocess"),
        downsample_nodes=sk.get("downsample_nodes", 0),
    )
    print(f"[skeletonize] {len(skels)} skeletons in {time.time()-t0:.1f}s "
          f"(parallel={sk.get('parallel', 8)})", flush=True)

    stats = skeleton_io.write_nml(skels, res_nm, out_nml,
                                  name=Path(out_dir).name)
    print(f"[skeletonize] wrote {out_nml}  "
          f"({stats['trees']} trees, {stats['nodes']} nodes)", flush=True)
    return out_nml


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading stage A — skeletonize")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    skeletonize(load_config(args.config))


if __name__ == "__main__":
    main()
