"""
Stage — supervoxel proofreading bundle for WebKnossos.

Reads an affinity ROI from precomputed, runs the SAME waterz watershed+agglomeration
as instance segmentation but keeps the pre-agglomeration SUPERVOXELS, and emits:
  - supervoxels.tif : the oversegmentation (the segmentation layer to upload)
  - agglomerate_t*.npz : per-threshold supervoxel->agglomerate mapping + adjacency
    graph (nodes/positions/edges/affinities) — the webknossos side turns each into an
    official Zarr-v3 agglomerate attachment for click-to-merge/split proofreading
  - em.tif : the matching EM image layer

Then (optionally) uploads EM + supervoxels + agglomerate attachments to WebKnossos
(headless upload needs webknossos-py >= 3.5.3, i.e. the `wk-latest` env).
Runs in the `magneton` env (waterz). Phase-1 validation on a single crop — no block
splitting / cross-block stitching.
    python -m magneton.proofreading.stages.supervoxel --config <cfg>
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np

from magneton.proofreading.config import (
    load_config, get_stage_config, strip_file_prefix,
)
from magneton.proofreading.lib import agglomerate_io
from magneton.proofreading.stages.membrane import _read_roi, _folder_id

UPLOAD_SCRIPT = "/gpfs/radev/home/yf354/magneton/upload_webknossos.py"

# PyTC affinity channel order -> spatial axis. Default (z,y,x): channel 0=z,1=y,2=x.
_CHAN = {"zyx": (2, 1, 0), "xyz": (0, 1, 2)}


def _aff_axis_order(aff_czyx, order):
    """(C,Z,Y,X) PyTC affinity -> (3,X,Y,Z) where out[d] = affinity along spatial
    axis d (0=X,1=Y,2=Z), reordering channels per `order` and transposing zyx->xyz."""
    cx, cy, cz = _CHAN[order]                 # channel index for X-, Y-, Z-affinity
    t = lambda c: np.ascontiguousarray(aff_czyx[c].transpose(2, 1, 0))   # (Z,Y,X)->(X,Y,Z)
    return np.stack([t(cx), t(cy), t(cz)], axis=0)


def supervoxel(cfg: dict):
    import tifffile
    try:
        from magneton.instance_segmentation.waterz_block import run_waterz_block
    except ImportError:
        from instance_segmentation.waterz_block import run_waterz_block

    paths = get_stage_config(cfg, "paths")
    sx = cfg.get("supervoxel_stage", {})
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    coords = paths["roi_coords"]

    t0 = time.time()
    aff = _read_roi(strip_file_prefix(paths["affinity"]), paths.get("aff_mip", 0), coords)  # (C,Z,Y,X)
    em = _read_roi(strip_file_prefix(paths["em"]), paths.get("em_mip", 0), coords)[0]        # (Z,Y,X)
    print(f"[supervoxel] aff{tuple(aff.shape)} em{tuple(em.shape)} ({time.time()-t0:.1f}s)", flush=True)

    # waterz needs increasing thresholds (in-place progressive agglomeration).
    thresholds = sorted(sx.get("thresholds", [sx.get("threshold", 0.4)]))
    t0 = time.time()
    supervox, segs = run_waterz_block(
        aff, seg_thresholds=thresholds,
        aff_thresholds=[sx.get("aff_threshold_low", 0.00001),
                        sx.get("aff_threshold_high", 0.99999)],
        merge_function=sx.get("merge_function", "aff50_his256"),
        return_fragments=True,
    )                                          # supervox (Z,Y,X); segs: one per threshold
    print(f"[supervoxel] waterz: {int(supervox.max())} supervox, "
          f"{len(thresholds)} thresholds ({time.time()-t0:.1f}s)", flush=True)

    # builder works in (X,Y,Z); supervoxels + affinity are shared across thresholds
    sv_xyz = np.ascontiguousarray(supervox.transpose(2, 1, 0))
    aff_xyz = _aff_axis_order(aff, sx.get("aff_channel_order", "zyx"))

    sv_tif = os.path.join(out_dir, "supervoxels.tif")
    em_tif = os.path.join(out_dir, "em.tif")
    tifffile.imwrite(sv_tif, supervox.astype(np.uint32))     # (Z,Y,X)
    tifffile.imwrite(em_tif, em)

    # the supervoxel RAG is shared across thresholds — compute it once.
    rag = agglomerate_io.compute_rag(sv_xyz, aff_xyz)

    agg_files = []          # (mapping_name, npz_path) — one per threshold
    for t, seg in zip(thresholds, segs):
        seg_xyz = np.ascontiguousarray(seg.transpose(2, 1, 0))
        data = agglomerate_io.agglomerate_bundle(sv_xyz, seg_xyz, aff_xyz, rag=rag)
        name = f"agglomerate_t{t:.2f}".replace(".", "_")
        npz = os.path.join(out_dir, name + ".npz")
        agglomerate_io.write_bundle_npz(data, npz)
        agg_files.append((name, npz))
        print(f"[supervoxel] t={t}: {data['_meta']} -> {name}", flush=True)
    print(f"[supervoxel] wrote {sv_tif} + {em_tif} + {len(agg_files)} agglomerate files", flush=True)

    up = sx.get("upload") or {}
    if up.get("enable", False):
        _upload(em_tif, sv_tif, agg_files, paths, up, out_dir)
    else:
        print("[supervoxel] upload disabled", flush=True)
    return sv_tif, agg_files


def _upload(em_tif, sv_tif, agg_files, paths, up, out_dir):
    vx, vy, vz = paths.get("voxel_size", [8, 8, 8])
    name = up.get("name") or Path(out_dir).name
    scratch = up.get("out_dir") or os.path.join(out_dir, "wk_upload_out")
    # env may be a name (-n) or a full prefix path (-p); the 3.5.3 `wk-latest` env
    # lives in project space, not the default envs dir, so it must be referenced by path.
    env = up.get("env", "wk-latest")
    env_flag = ["-p", env] if "/" in env else ["-n", env]
    cmd = [
        "conda", "run", *env_flag, "python", UPLOAD_SCRIPT,
        "--img-files", em_tif,
        "--seg-files", sv_tif,
        "--agglomerate-files", *[npz for _, npz in agg_files],
        "--agglomerate-names", *[nm for nm, _ in agg_files],
        "--voxel-sizes", f"({vx},{vy},{vz})",
        "--names", name,
        "--out-dir", scratch,
    ]
    folder = _folder_id(up.get("remote_folder"))
    if folder:
        cmd += ["--remote-folder", folder]
    tok = up.get("token") or os.environ.get("WEBKNOSSOS_TOKEN")
    if tok:
        cmd += ["--token", tok]
    print(f"[supervoxel] headless upload of '{name}' "
          f"({len(agg_files)} agglomerate mappings) ...", flush=True)
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading — supervoxel + agglomerate bundle")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    supervoxel(load_config(args.config))


if __name__ == "__main__":
    main()
