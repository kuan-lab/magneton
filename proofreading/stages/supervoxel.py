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


def _absorb_dust(sv_zyx, aff_czyx, order, max_vox):
    """
    Merge supervoxels of <= max_vox voxels into their best-affinity neighbour.

    A 1-voxel supervoxel is never a real object: it cannot be clicked reliably in
    WebKnossos and it inflates the supervoxel count the annotator has to work
    through. Each dust supervoxel is absorbed by the neighbour with the highest
    MEAN boundary affinity; dust-to-dust chains are resolved with union-find and
    the component takes its largest non-dust member (or its largest member if the
    whole chain is dust).

    Returns (sv_new, n_absorbed). Does not touch any supervoxel above max_vox.
    """
    if not max_vox:
        return sv_zyx, 0
    sizes = np.bincount(sv_zyx.ravel().astype(np.int64))
    ids = np.flatnonzero(sizes > 0)
    dust_mask = np.zeros(len(sizes), bool)
    dust_mask[ids[(sizes[ids] <= max_vox) & (ids > 0)]] = True
    if not dust_mask.any():
        return sv_zyx, 0

    cx, cy, cz = _CHAN[order]
    chan_for_axis = (cz, cy, cx)          # sv axes are (z, y, x)
    acc = {}                              # (dust_id, neighbour_id) -> [aff_sum, n]
    for d in range(3):
        lo = tuple(slice(0, -1) if i == d else slice(None) for i in range(3))
        hi = tuple(slice(1, None) if i == d else slice(None) for i in range(3))
        A, B = sv_zyx[lo], sv_zyx[hi]
        m = (A != B) & (A != 0) & (B != 0)
        if not m.any():
            continue
        a, b = A[m], B[m]
        keep = dust_mask[a] | dust_mask[b]
        if not keep.any():
            continue
        w = aff_czyx[chan_for_axis[d]][hi][m][keep]
        a, b = a[keep], b[keep]
        # orient so the dust side is first; a dust-dust face is recorded both ways
        for aa, bb, ww in zip(a.tolist(), b.tolist(), w.tolist()):
            if dust_mask[aa]:
                e = acc.setdefault((aa, bb), [0.0, 0]); e[0] += ww; e[1] += 1
            if dust_mask[bb]:
                e = acc.setdefault((bb, aa), [0.0, 0]); e[0] += ww; e[1] += 1

    best = {}
    for (dst, nbr), (tot, n) in acc.items():
        score = tot / max(n, 1)
        if dst not in best or score > best[dst][0]:
            best[dst] = (score, nbr)

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for dst, (_, nbr) in best.items():
        union(dst, nbr)

    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    lut = np.arange(len(sizes), dtype=sv_zyx.dtype)
    absorbed = 0
    for mem in groups.values():
        solid = [m for m in mem if not dust_mask[m]]
        rep = max(solid, key=lambda m: sizes[m]) if solid else max(mem, key=lambda m: sizes[m])
        for m in mem:
            if m != rep and dust_mask[m]:
                lut[m] = rep
                absorbed += 1
    orphan = int(dust_mask.sum()) - absorbed
    print(f"[supervoxel] dust (<= {max_vox} vox): {int(dust_mask.sum())} supervoxels, "
          f"{absorbed} absorbed, {orphan} left (no eligible neighbour)", flush=True)
    return lut[sv_zyx], absorbed


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
    order = sx.get("aff_channel_order", "zyx")
    # merge_function affects only the AGGLOMERATION, not the watershed, so several
    # can share one supervoxel layer and appear as separate WebKnossos mappings.
    merge_fns = sx.get("merge_functions") or [sx.get("merge_function", "aff50_his256")]
    dust_max = int(sx.get("dust_max_voxels", 0))

    runs, supervox = [], None
    for mf in merge_fns:
        t0 = time.time()
        sv, segs = run_waterz_block(
            aff, seg_thresholds=thresholds,
            aff_thresholds=[sx.get("aff_threshold_low", 0.00001),
                            sx.get("aff_threshold_high", 0.99999)],
            # MUST match instance-seg pass 1 (fragments_stage), or the proofread
            # supervoxels do not correspond to the ones production builds.
            # run_waterz_block defaults to interior_thr=0.1, which seeds inside
            # membranes; production uses 0.9.
            sv_type=sx.get("supervoxel", "3d"),
            interior_thr=sx.get("interior_threshold", 0.9),
            min_distance=sx.get("min_distance", 3),
            merge_function=mf,
            return_fragments=True,
        )                                      # sv (Z,Y,X); segs: one per threshold
        print(f"[supervoxel] {mf}: {int(sv.max())} supervox, "
              f"{len(thresholds)} thresholds ({time.time()-t0:.1f}s)", flush=True)
        if supervox is None:
            supervox = sv
        elif not np.array_equal(sv, supervox):
            raise RuntimeError(
                f"merge_function {mf} produced a different supervoxel layer; the "
                "watershed must be identical across merge functions for them to "
                "share one segmentation layer")
        runs.append((mf, segs))

    # Dust absorption rewrites the supervoxel layer, so each agglomeration is
    # rebuilt from the post-absorption labels. seg is constant within a supervoxel
    # (agglomeration only groups them, never splits), so supervoxel -> agglomerate
    # is an exact remap and the mappings stay consistent with the layer.
    if dust_max:
        sv_old = supervox
        supervox, _ = _absorb_dust(sv_old, aff, order, dust_max)
        rebuilt = []
        for mf, segs in runs:
            out = []
            for seg in segs:
                lut = np.zeros(int(sv_old.max()) + 1, dtype=seg.dtype)
                lut[sv_old.ravel()] = seg.ravel()      # constant per supervoxel
                out.append(lut[supervox])
            rebuilt.append((mf, out))
        runs = rebuilt
        n_before = len(np.unique(sv_old))
        # WebKnossos agglomerate attachments require DENSE segment ids (1..N);
        # absorption leaves a hole wherever an absorbed id used to be, so the
        # layer has to be recompacted (run_waterz_block's own output is dense).
        keep = np.flatnonzero(np.bincount(supervox.ravel().astype(np.int64)) > 0)
        keep = keep[keep > 0]
        cmap = np.zeros(int(supervox.max()) + 1, dtype=supervox.dtype)
        cmap[keep] = np.arange(1, len(keep) + 1, dtype=supervox.dtype)
        supervox = cmap[supervox]
        print(f"[supervoxel] supervoxels: {n_before} -> {len(keep)} after dust "
              f"absorption, relabelled dense 1..{len(keep)}", flush=True)

    # builder works in (X,Y,Z); supervoxels + affinity are shared across mappings
    sv_xyz = np.ascontiguousarray(supervox.transpose(2, 1, 0))
    aff_xyz = _aff_axis_order(aff, order)

    sv_tif = os.path.join(out_dir, "supervoxels.tif")
    em_tif = os.path.join(out_dir, "em.tif")
    tifffile.imwrite(sv_tif, supervox.astype(np.uint32))     # (Z,Y,X)
    tifffile.imwrite(em_tif, em)

    # the supervoxel RAG is shared across thresholds — compute it once.
    rag = agglomerate_io.compute_rag(sv_xyz, aff_xyz)

    agg_files = []          # (mapping_name, npz_path) — one per merge_fn x threshold
    multi = len(runs) > 1
    for mf, segs in runs:
        code = mf.split("_")[0]                 # e.g. aff50_his256 -> aff50
        for t, seg in zip(thresholds, segs):
            seg_xyz = np.ascontiguousarray(seg.transpose(2, 1, 0))
            data = agglomerate_io.agglomerate_bundle(sv_xyz, seg_xyz, aff_xyz, rag=rag)
            stem = f"agglomerate_{code}_t{t:.2f}" if multi else f"agglomerate_t{t:.2f}"
            name = stem.replace(".", "_")
            npz = os.path.join(out_dir, name + ".npz")
            agglomerate_io.write_bundle_npz(data, npz)
            agg_files.append((name, npz))
            print(f"[supervoxel] {code} t={t}: {data['_meta']} -> {name}", flush=True)
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
