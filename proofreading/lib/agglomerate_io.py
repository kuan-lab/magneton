"""
Build the inputs for a WebKnossos agglomerate mapping from a supervoxel volume +
its agglomeration.

WebKnossos supervoxel proofreading needs, for a segmentation layer whose voxels are
SUPERVOXELS (the oversegmentation), an agglomerate file that groups supervoxels into
agglomerates AND carries the supervoxel adjacency GRAPH (so the UI can split an
agglomerate with a min-cut on the edges, or merge across them). WebKnossos consumes
this as an ``AgglomerateGraph`` (networkx) written by the official Zarr-v3 writer
``AgglomerateAttachment.create``: every supervoxel is a node (with a representative
position), each retained adjacency is an affinity edge, and the graph's CONNECTED
COMPONENTS become the agglomerates. So to reproduce a threshold's agglomeration we
add every supervoxel as a node and keep only the INTRA-agglomerate adjacencies.

This module (magneton env) computes the numbers; the graph object is built on the
webknossos side (upload_webknossos.py, webknossos-py >= 3.5.3). We hand it a small,
env-portable npz bundle per threshold:

  seg_ids   (N,)    uint32   supervoxel ids present in the volume (nodes)
  positions (N,3)   int32    representative (x,y,z) voxel per supervoxel, aligned to seg_ids
  edge_a    (E,)    uint32   intra-agglomerate adjacency, endpoint A (global supervox id)
  edge_b    (E,)    uint32   endpoint B
  affinity  (E,)    float32  mean boundary affinity for the edge (min-cut weight)

Coordinates are (x,y,z) voxels in the supervoxel volume's frame. Works for any
oversegmentation (waterz or mito).

NOTE: an earlier hand-rolled HDF5 CSR file uploaded and appeared in the mapping
dropdown but DISPLAYED NOTHING (wrong container for the datastore). The official
Zarr-v3 writer is mandatory; this module now feeds that writer instead.
"""
from __future__ import annotations

import numpy as np

# PyTC affinity channel order -> channel index carrying (X, Y, Z) affinity.
# e.g. "zyx" means channel 0=Z-aff, 1=Y-aff, 2=X-aff, so X is channel 2.
_CHAN = {"zyx": (2, 1, 0), "xyz": (0, 1, 2)}


def compute_rag(supervox, aff):
    """Supervoxel region-adjacency graph with mean boundary affinity.

    supervox : (X,Y,Z) int label volume (0 = background).
    aff      : (3,X,Y,Z) float; aff[d] = affinity along spatial axis d (0=X,1=Y,2=Z).
               (Caller reorders PyTC channels to this axis convention.)
    Returns (edge_a, edge_b, affinity): 1-D arrays, edge_a<edge_b global supervox ids,
    affinity = mean of aff[d] over the shared face between the two supervoxels.
    """
    a_all, b_all, w_all = [], [], []
    for d in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[d] = slice(0, -1)
        hi[d] = slice(1, None)
        sa = supervox[tuple(lo)]
        sb = supervox[tuple(hi)]
        face = (sa != sb) & (sa > 0) & (sb > 0)
        if not face.any():
            continue
        la = sa[face].astype(np.int64)
        lb = sb[face].astype(np.int64)
        w = aff[d][tuple(hi)][face].astype(np.float64)     # affinity on the +d face
        a_all.append(np.minimum(la, lb))
        b_all.append(np.maximum(la, lb))
        w_all.append(w)
    if not a_all:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    a = np.concatenate(a_all); b = np.concatenate(b_all); w = np.concatenate(w_all)
    # collapse duplicate (a,b) faces -> mean affinity
    key = a * (int(b.max()) + 1) + b
    uk, inv = np.unique(key, return_inverse=True)
    ssum = np.bincount(inv, weights=w)
    scnt = np.bincount(inv)
    base = int(b.max()) + 1
    ea = (uk // base).astype(np.int64)
    eb = (uk % base).astype(np.int64)
    return ea, eb, (ssum / scnt).astype(np.float32)


def agglomerate_bundle(supervox, seg, aff, rag=None):
    """Assemble the npz-bundle arrays for one agglomeration (one threshold).

    supervox : (X,Y,Z) supervoxel labels (0 = background).
    seg      : (X,Y,Z) agglomerated labels at this threshold, co-registered with supervox.
    aff      : (3,X,Y,Z) affinity (only used if `rag` is not supplied).
    rag      : optional precomputed (edge_a, edge_b, affinity) from `compute_rag` — the
               RAG is shared across thresholds, so compute it once and pass it in.

    Returns a dict of arrays (seg_ids, positions, edge_a, edge_b, affinity) + `_meta`.
    Every present supervoxel is a node; only intra-agglomerate adjacencies are kept, so
    the graph's connected components reproduce this threshold's agglomeration.
    """
    supervox = np.ascontiguousarray(supervox)
    flat = supervox.ravel()
    vals, first = np.unique(flat, return_index=True)     # sorted, includes 0
    fg = vals > 0
    ids = vals[fg].astype(np.uint32)                     # present supervoxel ids (nodes)
    X, Y, Z = supervox.shape
    positions = np.stack(np.unravel_index(first[fg], (X, Y, Z)), axis=1).astype(np.int32)
    N = len(ids)

    # supervoxel -> agglomerate id (all voxels of a supervoxel share one agglomerate)
    maxid = int(supervox.max())
    seg2agg = np.zeros(maxid + 1, dtype=np.int64)
    m = flat > 0
    seg2agg[flat[m]] = seg.ravel()[m].astype(np.int64)

    ea, eb, ew = compute_rag(supervox, aff) if rag is None else rag
    same = (seg2agg[ea] == seg2agg[eb]) & (seg2agg[ea] > 0)
    ea = ea[same].astype(np.uint32)
    eb = eb[same].astype(np.uint32)
    ew = ew[same].astype(np.float32)

    n_agg = int(len(np.unique(seg2agg[ids.astype(np.int64)])))
    return {
        "seg_ids": ids,
        "positions": positions,
        "edge_a": ea,
        "edge_b": eb,
        "affinity": ew,
        "_meta": {"n_segments": N, "n_agglomerates": n_agg, "n_edges": int(len(ew))},
    }


def compute_block_sv_partial(supervox_zyx, aff_czyx, core_local, origin_xyz,
                             aff_order="zyx"):
    """Per-block supervoxel RAG partial, computed in the SEG stage where the
    fragments + affinity are already in memory (so merge never re-reads affinity).

    supervox_zyx : (Z,Y,X) local supervoxel labels for the block (0 = background).
    aff_czyx     : (C,Z,Y,X) affinity for the SAME block extent.
    core_local   : (cz1,cz2,cy1,cy2,cx1,cx2) core bounds in LOCAL block coords
                   (compute_core_region output minus the block origin).
    origin_xyz   : (x0,y0,z0) global voxel coord of the block's local (0,0,0).
    aff_order    : PyTC channel order ('zyx' or 'xyz').

    Returns a dict with LOCAL supervoxel ids (merge adds this block's sv offset):
      ids       (N,)   uint32   supervoxel ids present in the CORE (nodes)
      positions (N,3)  int32    representative GLOBAL (x,y,z) per id, aligned to ids
      edge_a    (E,)   uint32   within-core adjacency endpoint A (local id, a<b)
      edge_b    (E,)   uint32   endpoint B
      affinity  (E,)   float32  mean boundary affinity (min-cut weight)

    RAG is over the CORE sub-volume ONLY — each cross-block seam is owned by merge
    (from the two neighbouring cores), so it is not computed or double-counted here.
    """
    cz1, cz2, cy1, cy2, cx1, cx2 = core_local
    # core sub-volume -> (X,Y,Z) to match compute_rag's axis convention
    sv_xyz = np.ascontiguousarray(
        np.transpose(supervox_zyx[cz1:cz2, cy1:cy2, cx1:cx2], (2, 1, 0)))
    aff_cxyz = np.transpose(aff_czyx[:, cz1:cz2, cy1:cy2, cx1:cx2], (0, 3, 2, 1))
    cx, cy, cz = _CHAN[aff_order]                 # channel index for X/Y/Z affinity
    aff_axis = np.stack([aff_cxyz[cx], aff_cxyz[cy], aff_cxyz[cz]], axis=0).astype(np.float32)
    if aff_axis.size and aff_axis.max() > 1.0:
        aff_axis /= 255.0
    aff_axis = np.ascontiguousarray(aff_axis)

    ea, eb, ew = compute_rag(sv_xyz, aff_axis)    # local ids, a<b

    flat = sv_xyz.ravel()
    vals, first = np.unique(flat, return_index=True)     # sorted, includes 0
    fg = vals > 0
    ids = vals[fg].astype(np.uint32)
    X, Y, Z = sv_xyz.shape
    loc = np.stack(np.unravel_index(first[fg], (X, Y, Z)), axis=1).astype(np.int64)
    # local (x,y,z) within the core sub-array -> global voxel coord
    base = np.array([origin_xyz[0] + cx1, origin_xyz[1] + cy1, origin_xyz[2] + cz1],
                    dtype=np.int64)
    positions = (loc + base).astype(np.int32)
    return {
        "ids": ids,
        "positions": positions,
        "edge_a": ea.astype(np.uint32),
        "edge_b": eb.astype(np.uint32),
        "affinity": ew.astype(np.float32),
    }


def write_block_sv_partial(path, data):
    """Persist a per-block RAG partial (from compute_block_sv_partial) as npz."""
    np.savez(
        path,
        ids=data["ids"],
        positions=data["positions"],
        edge_a=data["edge_a"],
        edge_b=data["edge_b"],
        affinity=data["affinity"],
    )
    return path


def write_bundle_npz(data, path):
    """Write an agglomerate bundle as a compact npz for the webknossos-side writer."""
    np.savez(
        path,
        seg_ids=data["seg_ids"],
        positions=data["positions"],
        edge_a=data["edge_a"],
        edge_b=data["edge_b"],
        affinity=data["affinity"],
    )
    return path
