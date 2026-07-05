"""
Membrane "fishnet" — sparse, evenly-spaced points on a binary membrane mask.

The membrane (low-affinity boundary between objects, `reduce(aff) < threshold`)
is a dense set of thin sheets — millions of voxels. We reduce it to one
representative voxel per `spacing`^3 grid block, so the points trace the overall
membrane shape but stay sparse enough to:
  * seed nnInteractive NEGATIVE prompts (wall off cross-membrane flooding), and
  * be a cheap human-editable proofreading surface in WebKnossos — each point is
    one draggable node (delete a few to open a false split; add a few to wall off
    a merge), instead of repainting millions of membrane voxels.

Coordinates are (x,y,z) voxels throughout, matching skeleton_io / WebKnossos.
"""
from __future__ import annotations

import numpy as np


def membrane_fishnet(mask, spacing: int, method: str = "medial",
                     max_points: int = 0, seed: int = 0):
    """Sparse representative points on a membrane mask.

    mask    : (X, Y, Z) array; truthy = membrane.
    spacing : block side in voxels (the net's pitch). Larger = sparser.
    method  : "medial" (default) — one point per occupied block, at the voxel
              DEEPEST inside the membrane band (max distance-to-non-membrane).
              Lands points on the membrane's medial surface — its center, like a
              skeleton — instead of an edge voxel touching a cell. Needs scipy.
              "grid" — per-block membrane centroid (can sit at a band edge on a
              thin/curved membrane). "first" — the first membrane voxel per block.
    max_points : if >0 and exceeded, randomly subsample to this many (keeps WKS
              responsive; logged by the caller).

    Returns (N, 3) float xyz voxel coords.
    """
    mask = np.asarray(mask) > 0
    coords = np.argwhere(mask)                       # (M,3) integer xyz
    if coords.size == 0:
        return np.zeros((0, 3), dtype=float)
    if spacing <= 1:
        pts = coords.astype(float)
    elif method == "medial":
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(mask)    # depth into the membrane band
        keys = coords // spacing
        depth = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
        # sort by block, then deepest-first within each block; take first per block
        order = np.lexsort((-depth, keys[:, 2], keys[:, 1], keys[:, 0]))
        coords, keys = coords[order], keys[order]
        change = np.any(np.diff(keys, axis=0) != 0, axis=1)
        starts = np.concatenate(([0], np.flatnonzero(change) + 1))
        pts = coords[starts].astype(float)
    else:
        keys = coords // spacing                     # block id per voxel
        # sort voxels by block id so each block is a contiguous run
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        coords, keys = coords[order], keys[order]
        change = np.any(np.diff(keys, axis=0) != 0, axis=1)
        starts = np.concatenate(([0], np.flatnonzero(change) + 1))
        if method == "first":
            pts = coords[starts].astype(float)
        elif method == "grid":
            ends = np.concatenate((starts[1:], [len(coords)]))
            sums = np.add.reduceat(coords.astype(np.int64), starts, axis=0)
            counts = (ends - starts)[:, None]
            pts = sums / counts                      # per-block centroid (vectorized)
        else:
            raise ValueError(f"unknown method: {method!r} (want 'medial', 'grid', 'first')")
    if max_points and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[np.sort(rng.choice(len(pts), max_points, replace=False))]
    return pts


def fishnet_edges(pts, k: int = 6):
    """Edges connecting fishnet points into ONE connected tree (so WebKnossos
    doesn't split the edgeless point cloud into one tree per node on upload).

    Minimum spanning tree over a k-nearest-neighbour graph, with disconnected
    components chained together so the result always spans every point. Edges are
    short and hug the membrane (it looks like a net); they're cosmetic — the
    expand stage routes by node position, not edges. Returns (E,2) int index pairs.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

    n = len(pts)
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    k = min(k + 1, n)                                   # +1: query includes self
    d, idx = cKDTree(pts).query(pts, k=k)
    rows = np.repeat(np.arange(n), k)
    g = csr_matrix((d.ravel(), (rows, idx.ravel())), shape=(n, n))
    g = g.maximum(g.T)                                 # symmetric
    ncomp, lab = connected_components(g, directed=False)
    if ncomp > 1:                                       # chain reps so MST spans all
        reps = [int(np.flatnonzero(lab == c)[0]) for c in range(ncomp)]
        er, ec, ed = [], [], []
        for a, b in zip(reps[:-1], reps[1:]):
            er.append(a); ec.append(b); ed.append(float(np.linalg.norm(pts[a] - pts[b])))
        g = (g + csr_matrix((ed, (er, ec)), shape=(n, n))).maximum(g.T)
    return np.array(minimum_spanning_tree(g).nonzero()).T
