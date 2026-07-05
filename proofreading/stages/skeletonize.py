"""
Stage A — skeletonize an instance segmentation into a WebKnossos NML.

Reads paths.seg + skeletonize_stage from the config, runs kimimaro per instance,
writes <output>/skeletons.nml. Optionally appends a membrane "fishnet" (sparse
negative-prompt points, see lib/membrane_points) and auto-uploads EM + seg + NML
to WebKnossos. Upload it with /wk (or skeletonize_stage.upload), correct in
WebKnossos, then run the expand stage on the corrected NML.
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
from magneton.proofreading.lib import skeleton_io
from magneton.proofreading.lib.membrane_points import membrane_fishnet
from magneton.proofreading.stages.membrane import _read_roi, _to_membrane, _folder_id

UPLOAD_SCRIPT = "/gpfs/radev/home/yf354/magneton/upload_webknossos.py"


def _apply_roi(seg, paths, roi, out_dir, res_nm):
    """Crop the seg (X,Y,Z) and EM (Z,Y,X) to roi=[z1,z2,y1,y2,x1,x2], write
    seg_roi.tif + em_roi.tif to out_dir, and repoint paths['em']/['seg'] at them
    so the upload ships the cropped pair. Returns the cropped seg (X,Y,Z)."""
    import tifffile
    z1, z2, y1, y2, x1, x2 = roi
    seg_c = seg[x1:x2, y1:y2, z1:z2]
    seg_tif = os.path.join(out_dir, "seg_roi.tif")
    tifffile.imwrite(seg_tif, seg_c.transpose(2, 1, 0))      # (X,Y,Z) -> (Z,Y,X)
    em = tifffile.imread(strip_file_prefix(paths["em"]))      # (Z,Y,X)
    em_tif = os.path.join(out_dir, "em_roi.tif")
    tifffile.imwrite(em_tif, em[z1:z2, y1:y2, x1:x2])
    paths["em"], paths["seg"] = em_tif, seg_tif
    return seg_c


def _load_aligned_roi(paths, sk, out_dir):
    """Aligned-crop flow: read seg + EM from precomputed at paths['roi_coords']
    (the COMMON post-mip voxel space, [z1,z2,y1,y2,x1,x2]), write seg.tif + em.tif
    to out_dir, and repoint paths['em']/['seg'] at them for upload. The affinity is
    read separately by _membrane_net at the same coords. Returns (seg_xyz, res_nm).
    """
    import tifffile
    coords = paths["roi_coords"]
    seg = _read_roi(strip_file_prefix(paths["seg"]), paths.get("seg_mip", 0), coords)[0]
    seg = np.transpose(seg, (2, 1, 0))                           # (Z,Y,X) -> (X,Y,Z)
    em = _read_roi(strip_file_prefix(paths["em"]), paths.get("em_mip", 0), coords)[0]  # (Z,Y,X)
    seg_tif = os.path.join(out_dir, "seg.tif")
    em_tif = os.path.join(out_dir, "em.tif")
    tifffile.imwrite(seg_tif, seg.transpose(2, 1, 0))            # -> (Z,Y,X)
    tifffile.imwrite(em_tif, em)
    paths["em"], paths["seg"] = em_tif, seg_tif
    res_nm = tuple(float(v) for v in sk.get("res_nm", [8, 8, 8]))
    return seg, res_nm


def _seg_boundary_mask(seg):
    """Membrane proxy from instance boundaries: voxels adjacent (6-conn) to a
    different label. Both sides of each transition are marked (~2-voxel sheet).
    Exact and voxel-aligned with the seg crop — no affinity/coords needed."""
    m = np.zeros(seg.shape, dtype=bool)
    for ax in range(seg.ndim):
        lo = [slice(None)] * seg.ndim
        hi = [slice(None)] * seg.ndim
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        diff = seg[tuple(lo)] != seg[tuple(hi)]
        m[tuple(lo)] |= diff
        m[tuple(hi)] |= diff
    return m


def _membrane_net(cfg, sk, seg, paths):
    """Build the membrane fishnet point cloud, or None if disabled.

    `negatives.source`:
      - "affinity" (default for aligned crops): threshold an affinity ROI to a
        thick membrane. Source/coords come from paths['affinity']+paths['roi_coords']
        (the aligned-precomputed flow), or negatives.inference/coords, or
        membrane_stage. A thick membrane is what makes method="medial" meaningful.
      - "seg_boundary": membrane = instance boundaries of the seg crop (oracle, but
        only ~2 voxels thick, so points sit on cell-touching edges — no center).
    Returns (N,3) xyz voxel coords aligned with the seg/EM crop.
    """
    neg = sk.get("negatives") or {}
    if not neg.get("enable", False):
        return None
    source = neg.get("source", "affinity")
    if source == "seg_boundary":
        mask = _seg_boundary_mask(seg)
    elif source == "affinity":
        mstage = cfg.get("membrane_stage", {})
        # affinity volume + ROI: aligned-crop paths first, then negatives, then membrane_stage
        aff_path = (neg.get("inference", {}) or {}).get("path") or \
            strip_file_prefix(paths.get("affinity") or "") or \
            (mstage.get("inference", {}) or {}).get("path")
        coords = neg.get("coords") or paths.get("roi_coords") or mstage.get("coords")
        if not aff_path or not coords:
            print("[skeletonize] negatives source=affinity but no affinity path/coords "
                  "— skipping fishnet", flush=True)
            return None
        ninf = neg.get("inference") or {}
        aff = _read_roi(strip_file_prefix(aff_path), ninf.get("mip", paths.get("aff_mip", 0)),
                        coords)
        mem = _to_membrane(aff, neg.get("channel_reduce", "min"),
                           neg.get("threshold", 140), 1)         # (Z,Y,X) 0/1
        mask = np.transpose(mem, (2, 1, 0))                      # -> (X,Y,Z)
        if mask.shape != seg.shape:
            print(f"[skeletonize] WARNING membrane {mask.shape} != seg {seg.shape} "
                  "— affinity ROI not aligned to the seg crop", flush=True)
    else:
        print(f"[skeletonize] unknown negatives.source={source!r} — skipping", flush=True)
        return None
    method = neg.get("method", "medial")
    pts = membrane_fishnet(mask, spacing=neg.get("spacing", 8),
                           method=method, max_points=neg.get("max_points", 0))
    print(f"[skeletonize] membrane[{source}] {float(mask.mean())*100:.1f}% of voxels "
          f"-> {len(pts)} fishnet points (method={method}, spacing={neg.get('spacing',8)})",
          flush=True)
    return pts


def skeletonize(cfg: dict):
    paths = get_stage_config(cfg, "paths")
    sk = get_stage_config(cfg, "skeletonize")
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_nml = os.path.join(out_dir, "skeletons.nml")

    t0 = time.time()
    if sk.get("source") == "precomputed_roi":
        # Aligned crop: seg + EM + affinity all read from precomputed at the same
        # 8nm ROI, so they're voxel-aligned by construction (no GT-crop archaeology).
        seg, res_nm = _load_aligned_roi(paths, sk, out_dir)
        print(f"[skeletonize] aligned ROI {paths['roi_coords']} -> seg {seg.shape} "
              f"res={res_nm}nm, {len(np.unique(seg))} labels  ({time.time()-t0:.1f}s)",
              flush=True)
    else:
        seg, res_nm = skeleton_io.load_seg(
            paths["seg"], source=sk.get("source", "tif"),
            mip=sk.get("mip", 0), res_nm=sk.get("res_nm"),
        )
        print(f"[skeletonize] seg {seg.shape} res={res_nm}nm in {time.time()-t0:.1f}s", flush=True)
        # Optional tif sub-ROI crop. roi = [z1,z2,y1,y2,x1,x2].
        roi = paths.get("roi")
        if roi:
            seg = _apply_roi(seg, paths, roi, out_dir, res_nm)
            print(f"[skeletonize] ROI {roi} -> seg {seg.shape}, "
                  f"{len(np.unique(seg))} labels", flush=True)

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

    point_trees = None
    pts = _membrane_net(cfg, sk, seg, paths)
    if pts is not None and len(pts):
        point_trees = [{"name": "membrane_net", "points": pts, "connect": True}]

    stats = skeleton_io.write_nml(skels, res_nm, out_nml,
                                  name=Path(out_dir).name, point_trees=point_trees)
    print(f"[skeletonize] wrote {out_nml}  "
          f"({stats['trees']} trees, {stats['nodes']} nodes)", flush=True)

    up = sk.get("upload") or {}
    if up.get("enable", False):
        _upload(paths, res_nm, out_nml, up, out_dir)
    else:
        print("[skeletonize] upload disabled; upload with /wk", flush=True)
    return out_nml


def _upload(paths, res_nm, nml, up, out_dir):
    """Shell out to upload_webknossos.py (yf354 env): EM image + seg + skeleton NML."""
    token = up.get("token") or os.environ.get("WEBKNOSSOS_TOKEN")
    em_tif = strip_file_prefix(up.get("em") or paths["em"])
    seg_tif = strip_file_prefix(up.get("seg") or paths["seg"])
    vx, vy, vz = res_nm
    name = up.get("name") or Path(out_dir).name
    scratch = up.get("out_dir") or os.path.join(out_dir, "wk_upload_out")
    cmd = [
        "conda", "run", "-n", up.get("env", "yf354"), "python", UPLOAD_SCRIPT,
        "--img-files", em_tif,
        "--seg-files", seg_tif,
        "--skel-files", nml,
        "--voxel-sizes", f"({vx},{vy},{vz})",
        "--names", name,
        "--out-dir", scratch,
    ]
    folder = _folder_id(up.get("remote_folder"))
    if folder:
        cmd += ["--remote-folder", folder]
    if not token:
        print("[skeletonize] WEBKNOSSOS_TOKEN unset; NML written but NOT uploaded.\n"
              "  Set WEBKNOSSOS_TOKEN (or upload.token) and rerun, or run manually:\n  "
              + " ".join(cmd) + ' --token "$WEBKNOSSOS_TOKEN"', flush=True)
        return
    print(f"[skeletonize] uploading '{name}' via {up.get('env','yf354')} env ...", flush=True)
    subprocess.run(cmd + ["--token", token], check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading stage A — skeletonize")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    skeletonize(load_config(args.config))


if __name__ == "__main__":
    main()
