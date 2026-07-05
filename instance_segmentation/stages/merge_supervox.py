# -*- coding: utf-8 -*-
"""
merge-supervox — stitch per-block supervoxels into a GLOBAL supervoxel layer +
a WebKnossos agglomerate graph, reusing the instance-seg block/merge machinery.

Prereq: the segmentation stage ran with `emit_supervoxels: true`, so each block
wrote (a) a supervoxel volume (`sv_path`), (b) `sv_max_id`, and (c) a within-core
RAG partial npz (`sv_rag_path`: node ids + representative positions + intra-block
adjacency + boundary affinity). The RAG is computed in the seg stage, where the
fragments and affinity are already in memory — so THIS stage never re-reads the
big affinity volume; it only reads thin seam planes. The normal merge stage must
also have produced the agglomerated global instances volume (`paths.output`).

Products (for supervoxel proofreading in WebKnossos):
  1. GLOBAL supervoxel precomputed (`paths.supervox_output`): each block's
     supervoxels offset to globally-unique ids, trimmed to its core (same offset +
     core-trim as merge_apply but WITHOUT a union rep_map — supervoxels stay atomic;
     a human merges/splits them via the graph), then compacted to dense ids 1..N.
  2. agglomerate npz bundle (`paths.supervox_agglomerate_npz`): every supervoxel is a
     node (with a representative position); intra-agglomerate adjacencies are affinity
     edges. Its CONNECTED COMPONENTS reproduce the global agglomeration. Fed to
     upload_webknossos.upload_supervoxels_official (Zarr-v3 writer).

Map-reduce, parallelized across the node's cores in ONE job (ProcessPoolExecutor):
per-block passes — write cores (A), seam edges + supervox->agglomerate (B), relabel
cores to dense ids (D) — are chunk-disjoint / independent; only the dense-id reduce
(C) is global, and it is small vectorized numpy. To promote to a SLURM array later,
wrap A/B/D with a --task-id block slice; the reduce stays a single job.
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from cloudvolume import CloudVolume
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.utils.meta_utils import load_index_meta
from magneton.instance_segmentation.utils.block_utils import compute_core_region
from magneton.instance_segmentation.stages.merge_apply import _apply_block, _ensure_output_volume
from magneton.proofreading.lib.agglomerate_io import _CHAN, write_bundle_npz


def _bridge_components(N, da, db, inst_dense, weight):
    """Import the instance-level (merge-pools) stitching into the supervoxel graph.

    The RAG only links directly-adjacent supervoxels, so an instance whose cross-block
    pieces were merged via OVERLAP CO-LOCATION (leaving no adjacency after core-trim)
    spans several RAG components. Add one bridge edge per disconnected same-instance
    component so the graph's connected components reproduce the instances (paths.output).

    N: node count (dense ids 1..N). da/db: real dense edges (1..N). inst_dense: instance
    id per dense node (index i -> dense id i+1). weight: affinity for bridge edges (they
    have no measured boundary). Returns (a,b,w) bridge arrays, a<b.
    """
    empty = (np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros(0, np.float32))
    if N == 0:
        return empty
    g = coo_matrix((np.ones(da.size, np.int8), (da - 1, db - 1)), shape=(N, N))
    _, comp = connected_components(g, directed=False)          # 0-based comp per node index
    dense_ids = np.arange(1, N + 1)
    order = np.lexsort((comp, inst_dense))                      # group by instance, then comp
    si = inst_dense[order]; sc = comp[order]; sid = dense_ids[order]
    ba, bb = [], []
    cur_inst = -1; hub = 0; prev_comp = -1
    for inst, c, node in zip(si.tolist(), sc.tolist(), sid.tolist()):
        if inst != cur_inst:                                   # new instance -> its hub
            cur_inst = inst; hub = node; prev_comp = c
        elif c != prev_comp:                                   # first node of a new comp here
            if inst > 0:                                       # skip background instance 0
                a, b = (hub, node) if hub < node else (node, hub)
                ba.append(a); bb.append(b)
            prev_comp = c
    ba = np.array(ba, np.uint32); bb = np.array(bb, np.uint32)
    return ba, bb, np.full(ba.size, np.float32(weight), np.float32)


def _compute_offsets_by_key(blocks_meta, key, start_gid=1):
    """Per-block cumulative offset so global_id = local_id + offset[index] is unique
    across blocks. Mirrors merge_pools._compute_global_offsets but keyed on `key`
    (here 'sv_max_id')."""
    offsets, cur = {}, int(start_gid)
    for b in sorted(blocks_meta, key=lambda b: b["index"]):
        offsets[int(b["index"])] = cur
        cur += int(b.get(key, 0))
    return offsets, cur


# ----------------------------------------------------------------------------
# Parallel runner
# ----------------------------------------------------------------------------
def _pool_run(fn, payloads, workers, desc, initializer=None, initargs=()):
    """Map fn over payloads across `workers` processes (serial if workers<=1)."""
    results = []
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=initializer, initargs=initargs) as ex:
            futs = [ex.submit(fn, p) for p in payloads]
            for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
                results.append(f.result())
    else:
        if initializer is not None:
            initializer(*initargs)
        for p in tqdm(payloads, desc=desc):
            results.append(fn(p))
    return results


# ----------------------------------------------------------------------------
# Pass A: write global supervoxel cores (offset + trim, no rep_map)
# ----------------------------------------------------------------------------
def _write_core_worker(payload):
    blk, sv_offsets, sv_output, core = payload
    _apply_block(dict(blk, path=blk["sv_path"]), sv_offsets,
                 rep_map=None, output_path=sv_output, core_bounds=core)
    return blk["index"]


# ----------------------------------------------------------------------------
# Pass B: seam edges (thin boundary planes) + supervoxel -> agglomerate
# ----------------------------------------------------------------------------
def _seam_agg_worker(payload):
    """For one block: cross-core seam adjacencies (with boundary affinity) on its
    +face halos, and each core supervoxel's agglomerate id (from the instances vol).
    Reads the assembled GLOBAL supervoxel volume (ids already global), so it must run
    after Pass A. Returns (seam_a, seam_b, seam_w, sv_ids, sv_aggs) — all global."""
    (blk, core, sv_output, inst_path, aff_path, aff_order, mip) = payload
    cz1, cz2, cy1, cy2, cx1, cx2 = core
    vsz_hi = blk["_vsz"]                                    # (X,Y,Z) volume size
    hx = 1 if cx2 < vsz_hi[0] else 0
    hy = 1 if cy2 < vsz_hi[1] else 0
    hz = 1 if cz2 < vsz_hi[2] else 0
    Xc, Yc, Zc = cx2 - cx1, cy2 - cy1, cz2 - cz1

    sv_vol = CloudVolume(sv_output, mip=mip, bounded=False, progress=False, fill_missing=True)
    inst_vol = CloudVolume(inst_path, mip=mip, bounded=False, progress=False, fill_missing=True)
    aff_vol = CloudVolume(aff_path, mip=mip, bounded=False, progress=False, fill_missing=True)

    sv_slab = np.ascontiguousarray(
        sv_vol[cx1:cx2 + hx, cy1:cy2 + hy, cz1:cz2 + hz][..., 0])   # (X',Y',Z') global ids
    sv_core = sv_slab[:Xc, :Yc, :Zc]

    # --- supervoxel -> agglomerate (one instances read; sample first voxel per id) ---
    inst_slab = np.ascontiguousarray(inst_vol[cx1:cx2, cy1:cy2, cz1:cz2][..., 0])
    flat_sv = sv_core.ravel()
    ids, first = np.unique(flat_sv, return_index=True)
    fg = ids > 0
    sv_ids = ids[fg].astype(np.int64)
    sv_aggs = inst_slab.ravel()[first[fg]].astype(np.int64)
    # representative GLOBAL (x,y,z) per supervoxel — the SAME first-occurrence voxel, so
    # it is guaranteed to lie in the written volume (no phantom / mis-registered positions).
    loc = np.stack(np.unravel_index(first[fg], (Xc, Yc, Zc)), axis=1).astype(np.int64)
    sv_pos = (loc + np.array([cx1, cy1, cz1], np.int64)).astype(np.int32)

    # --- seam edges: pair the two planes straddling each +face, weight = boundary aff ---
    ea, eb, ew = [], [], []

    def _seam(lo, hi, aff_plane):
        m = (lo != hi) & (lo > 0) & (hi > 0)
        if not m.any():
            return
        a = lo[m].astype(np.int64)
        b = hi[m].astype(np.int64)
        ea.append(np.minimum(a, b))
        eb.append(np.maximum(a, b))
        ew.append(aff_plane[m].astype(np.float32))

    def _aff_plane(slab_xyzc, order, axis):
        ch = _CHAN[order][axis]
        p = slab_xyzc[..., ch].astype(np.float32)
        if p.size and p.max() > 1.0:
            p /= 255.0
        return p

    if hx:  # +x face at global x=cx2 (affinity indexed at the higher voxel)
        lo = sv_slab[Xc - 1, :Yc, :Zc]
        hi = sv_slab[Xc, :Yc, :Zc]
        aff = _aff_plane(aff_vol[cx2:cx2 + 1, cy1:cy2, cz1:cz2][0], aff_order, 0)  # (Y,Z)
        _seam(lo, hi, aff)
    if hy:  # +y face at global y=cy2
        lo = sv_slab[:Xc, Yc - 1, :Zc]
        hi = sv_slab[:Xc, Yc, :Zc]
        aff = _aff_plane(aff_vol[cx1:cx2, cy2:cy2 + 1, cz1:cz2][:, 0], aff_order, 1)  # (X,Z)
        _seam(lo, hi, aff)
    if hz:  # +z face at global z=cz2
        lo = sv_slab[:Xc, :Yc, Zc - 1]
        hi = sv_slab[:Xc, :Yc, Zc]
        aff = _aff_plane(aff_vol[cx1:cx2, cy1:cy2, cz2:cz2 + 1][:, :, 0], aff_order, 2)  # (X,Y)
        _seam(lo, hi, aff)

    if ea:
        seam_a = np.concatenate(ea)
        seam_b = np.concatenate(eb)
        seam_w = np.concatenate(ew)
    else:
        seam_a = np.zeros(0, np.int64); seam_b = np.zeros(0, np.int64); seam_w = np.zeros(0, np.float32)
    return seam_a, seam_b, seam_w, sv_ids, sv_aggs, sv_pos


# ----------------------------------------------------------------------------
# Pass D: relabel written cores into the dense id space (via a shared LUT)
# ----------------------------------------------------------------------------
_LUT = None


def _init_lut(lut):
    global _LUT
    _LUT = lut


def _relabel_worker(payload):
    blk, core, sv_output, mip = payload
    cz1, cz2, cy1, cy2, cx1, cx2 = core
    # compress=False to match how the volume was created — a default (compressed)
    # write handle silently fails to persist the relabel.
    sv_vol = CloudVolume(sv_output, mip=mip, bounded=False, progress=False,
                         fill_missing=True, compress=False, non_aligned_writes=True)
    arr = np.ascontiguousarray(sv_vol[cx1:cx2, cy1:cy2, cz1:cz2][..., 0])  # (X,Y,Z) global
    dense = _LUT[arr]                                                       # global -> dense, 0 stays 0
    sv_vol[cx1:cx2, cy1:cy2, cz1:cz2] = dense[:, :, :, np.newaxis]
    return blk["index"]


# ----------------------------------------------------------------------------
def merge_supervox(global_cfg, stage_cfg):
    paths        = global_cfg["paths"]
    input_path   = paths["input"]                         # affinity (seam edge weights only)
    inst_path    = paths["output"]                        # merged global instances (agglomeration)
    sv_output    = paths["supervox_output"]               # GLOBAL supervoxel precomputed (written here)
    npz_path     = paths.get("supervox_agglomerate_npz",
                             os.path.join(os.path.dirname(sv_output.replace("file://", "")),
                                          "supervox_agglomerate.npz"))
    metadata_dir = stage_cfg.get("metadata_dir", "./metadata/local_metadata")
    mip          = stage_cfg.get("mip", 0)
    aff_order    = stage_cfg.get("aff_channel_order", "zyx")
    overlap_zyx  = tuple(global_cfg.get("block", {}).get("overlap", [0, 0, 0]))
    workers      = int(stage_cfg.get("supervox_workers", stage_cfg.get("workers", os.cpu_count() or 1)))

    # ---- blocks that emitted supervoxels + their RAG partials ----
    blocks = [b for b in load_index_meta(metadata_dir).get("blocks", [])
              if b.get("done") and b.get("sv_path") and b.get("sv_rag_path")]
    blocks.sort(key=lambda b: b["index"])
    if not blocks:
        raise SystemExit("[merge-supervox] no blocks with sv_path+sv_rag_path — run "
                         "segmentation with emit_supervoxels: true first.")
    sv_offsets, next_sv = _compute_offsets_by_key(blocks, "sv_max_id")
    print(f"[merge-supervox] {len(blocks)} blocks, {next_sv-1} raw global supervoxels, "
          f"{workers} workers")

    # volume geometry
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vsz = tuple(aff_vol.info["scales"][0]["size"])         # (X,Y,Z)
    vol_shape_zyx = (vsz[2], vsz[1], vsz[0])
    chunk_xyz = tuple(aff_vol.chunk_size)
    chunk_zyx = (chunk_xyz[2], chunk_xyz[1], chunk_xyz[0])
    cores = {b["index"]: compute_core_region(tuple(b["coords"]), overlap_zyx,
                                             vol_shape_zyx, chunk_zyx) for b in blocks}

    # ---- Pass A: write GLOBAL supervoxel cores (offset + core-trim, no rep_map) ----
    _ensure_output_volume(input_path, sv_output, mip)
    _pool_run(_write_core_worker,
              [(b, sv_offsets, sv_output, cores[b["index"]]) for b in blocks],
              workers, "supervox write cores")

    # ---- load per-block RAG partials — ONLY for within-core edge weights. Nodes,
    # positions and instances come from the ASSEMBLED volume (seam worker), so partial
    # ids never written to sv_output cannot leak in as phantom singleton nodes. ----
    be_a, be_b, be_w = [], [], []
    for b in blocks:
        off = sv_offsets[b["index"]]
        d = np.load(b["sv_rag_path"])
        if d["edge_a"].size:
            be_a.append(d["edge_a"].astype(np.int64) + off)
            be_b.append(d["edge_b"].astype(np.int64) + off)
            be_w.append(d["affinity"].astype(np.float32))

    # ---- Pass B: seam edges + supervoxel->agglomerate (reads assembled volume) ----
    for b in blocks:
        b["_vsz"] = vsz
    seam_res = _pool_run(
        _seam_agg_worker,
        [(b, cores[b["index"]], sv_output, inst_path, input_path, aff_order, mip) for b in blocks],
        workers, "supervox seam+agg")

    # ---- Pass C: global reduce (vectorized) ----
    # edges = within-core (partials) + cross-core (seams); both already global, a<b.
    seam_a = [r[0] for r in seam_res]; seam_b = [r[1] for r in seam_res]; seam_w = [r[2] for r in seam_res]
    A = np.concatenate(be_a + seam_a) if (be_a or any(x.size for x in seam_a)) else np.zeros(0, np.int64)
    B = np.concatenate(be_b + seam_b) if (be_b or any(x.size for x in seam_b)) else np.zeros(0, np.int64)
    W = np.concatenate(be_w + seam_w).astype(np.float64) if (be_w or any(x.size for x in seam_w)) else np.zeros(0, np.float64)

    # ---- nodes from the ASSEMBLED volume (seam worker): the exact supervoxels present in
    # sv_output, with valid representative voxels + instances. Cores are disjoint so ids are
    # unique; np.unique is defensive. Global ids run 1..next_sv INCLUSIVE -> size arrays +1.
    dim = next_sv + 1
    _nid = np.concatenate([r[3] for r in seam_res]) if seam_res else np.zeros(0, np.int64)
    _nag = np.concatenate([r[4] for r in seam_res]) if seam_res else np.zeros(0, np.int64)
    _npo = np.concatenate([r[5] for r in seam_res]) if seam_res else np.zeros((0, 3), np.int32)
    present, _u = np.unique(_nid, return_index=True)     # sorted unique ids actually in sv_output
    positions = _npo[_u]
    N = int(present.size)
    agg_of = np.zeros(dim, dtype=np.int64); agg_of[present] = _nag[_u]
    lut = np.zeros(dim, dtype=np.uint32); lut[present] = np.arange(1, N + 1, dtype=np.uint32)
    seg_ids = np.arange(1, N + 1, dtype=np.uint32)

    # dedup (a,b) -> mean affinity
    if A.size:
        key = A * np.int64(dim) + B
        uk, inv = np.unique(key, return_inverse=True)
        wsum = np.bincount(inv, weights=W)
        wcnt = np.bincount(inv)
        Au = uk // np.int64(dim)
        Bu = uk % np.int64(dim)
        Wu = (wsum / wcnt).astype(np.float32)
    else:
        Au = np.zeros(0, np.int64); Bu = np.zeros(0, np.int64); Wu = np.zeros(0, np.float32)

    # real intra-agglomerate RAG edges, in dense id space
    keep = (agg_of[Au] == agg_of[Bu]) & (agg_of[Au] > 0)
    da = lut[Au[keep]].astype(np.int64); db = lut[Bu[keep]].astype(np.int64)
    dw = Wu[keep].astype(np.float32)
    mreal = (da > 0) & (db > 0)
    da, db, dw = da[mreal], db[mreal], dw[mreal]

    # bridge same-instance RAG components so connected components reproduce the instances
    # (merge-pools merged some cross-block pieces via overlap that leave no adjacency).
    inst_dense = agg_of[present]                    # instance per dense node (present sorted -> 1..N)
    bw = float(np.median(dw)) if dw.size else 0.5   # inferred merges get a neutral (median) weight
    ba, bb, bwarr = _bridge_components(N, da, db, inst_dense, bw)
    n_real, n_bridge = int(da.size), int(ba.size)
    ea = np.concatenate([da.astype(np.uint32), ba])
    eb = np.concatenate([db.astype(np.uint32), bb])
    ew = np.concatenate([dw, bwarr])

    n_agg = int(np.unique(agg_of[present]).size) if N else 0
    # verify: after bridging, #components should equal #instances present
    ncc = int(connected_components(
        coo_matrix((np.ones(ea.size, np.int8), (ea - 1, eb - 1)), shape=(N, N)),
        directed=False)[0]) if N else 0
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    write_bundle_npz(
        {"seg_ids": seg_ids, "positions": positions.astype(np.int32),
         "edge_a": ea, "edge_b": eb, "affinity": ew}, npz_path)

    # ---- Pass D: relabel the written volume into dense ids (LUT shared per worker) ----
    _pool_run(_relabel_worker,
              [(b, cores[b["index"]], sv_output, mip) for b in blocks],
              workers, "supervox relabel dense",
              initializer=_init_lut, initargs=(lut,))

    print(f"[merge-supervox] wrote {sv_output}")
    print(f"[merge-supervox] bundle -> {npz_path}: {N} supervox, {n_agg} agglomerates, "
          f"{n_real} real + {n_bridge} bridge edges | graph components={ncc} "
          f"(should == {n_agg})")
    return sv_output, npz_path


def main():
    ap = argparse.ArgumentParser(description="Stitch supervoxels into a global layer + agglomerate graph.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    merge_supervox(cfg, get_stage_config(cfg, "merge"))


if __name__ == "__main__":
    main()
