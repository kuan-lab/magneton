"""
Helpers for the graph-based instance-segmentation pipeline.

Design (mirrors funkelab/LSD and PyTC 2.x, see claude_notes):
  pass 1  fragments : watershed per block, cropped to a non-overlapping core,
                      ids made globally unique -> ONE fragments volume
  pass 2  edges     : per block, waterz over core+halo on those fragments ->
                      (u, v, score, contact) region-graph edges, incl. pairs
                      straddling a core face, scored by the SAME merge function
                      used inside a block
  pass 3  global    : concat edges -> threshold -> connected components -> LUT
  pass 4  relabel   : apply LUT per core

Why: the legacy pipeline labelled every overlap voxel twice and stitched by
overlap voting, which merged two neurons across a membrane strip that each
block assigned to a different cell (fib_c id 49363, 38 shared voxels, affinity
0). A region-graph edge asks "how strong is the boundary between these two
fragments" instead of "did two independent guesses overlap".
"""
import glob
import os

import numpy as np

EDGE_DTYPE = np.dtype([("u", np.uint64), ("v", np.uint64),
                       ("score", np.float32), ("contact", np.uint32)])


# ---------------------------------------------------------------- fragment ids
def fragment_id_bump(block_index, core_shape_zyx):
    """
    Offset making block-local fragment ids globally unique without any
    coordination between blocks: local ids are 1..N with N <= voxels in a core.

    uint64 on purpose -- index * voxels_per_core overflows uint32 almost
    immediately (48 blocks of 768^3 ~ 2.2e10).
    """
    voxels = np.uint64(1)
    for v in core_shape_zyx:
        voxels *= np.uint64(int(v))
    return np.uint64(int(block_index)) * voxels


def bump_ids(frag_zyx, bump):
    """Add `bump` to every non-zero id (0 stays background)."""
    out = frag_zyx.astype(np.uint64, copy=True)
    nz = out != 0
    out[nz] += np.uint64(bump)
    return out


# ---------------------------------------------------------------------- edges
def write_edges_npz(path, u, v, score, contact):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        u=np.asarray(u, np.uint64), v=np.asarray(v, np.uint64),
                        score=np.asarray(score, np.float32),
                        contact=np.asarray(contact, np.uint32))


def read_edges_dir(edges_dir, pattern="edges_*.npz"):
    """Concatenate every per-block edge file. Returns (u, v, score, contact)."""
    files = sorted(glob.glob(os.path.join(edges_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"no edge files matching {pattern} in {edges_dir}")
    us, vs, ss, cs = [], [], [], []
    for f in files:
        d = np.load(f)
        us.append(d["u"]); vs.append(d["v"]); ss.append(d["score"]); cs.append(d["contact"])
    return (np.concatenate(us), np.concatenate(vs),
            np.concatenate(ss), np.concatenate(cs))


def dedupe_edges(u, v, score, contact, how="max"):
    """
    Both neighbours of a face score the same pair. Collapse duplicates.

    how="max"  keep the WORST (largest) score  -> conservative, prefers a split
    how="min"  keep the best score             -> prefers a merge
    how="mean" average
    Contact is summed per unique pair only for "mean"; otherwise the contact of
    the kept row is used (they agree up to halo truncation).
    """
    if len(u) == 0:
        return u, v, score, contact
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    key = np.stack([a, b], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    n = len(uniq)
    if how == "min":
        best = np.full(n, np.inf, np.float64)
        np.minimum.at(best, inv, score)
    elif how == "mean":
        tot = np.zeros(n, np.float64); cnt = np.zeros(n, np.int64)
        np.add.at(tot, inv, score); np.add.at(cnt, inv, 1)
        best = tot / np.maximum(cnt, 1)
    else:
        best = np.full(n, -np.inf, np.float64)
        np.maximum.at(best, inv, score)
    cmax = np.zeros(n, np.uint64)
    np.maximum.at(cmax, inv, contact.astype(np.uint64))
    return (uniq[:, 0].astype(np.uint64), uniq[:, 1].astype(np.uint64),
            best.astype(np.float32), cmax.astype(np.uint32))


# --------------------------------------------------------------- global merge
def components_from_edges(u, v, score, contact, threshold, min_contact=0):
    """
    Keep edges with score <= threshold (waterz score = 1 - boundary affinity
    quantile, same convention as segmentation_stage thresholds) and at least
    `min_contact` contact voxels, then take connected components.

    Returns (nodes, roots): parallel uint64 arrays, root = SMALLEST fragment id
    in the component. Using a real fragment id (not a dense renumbering) makes
    the LUT idempotent -- applying it twice is a no-op, so an in-place relabel
    is safe to retry.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keep = np.asarray(score, np.float64) <= float(threshold)
    if min_contact:
        keep &= np.asarray(contact, np.uint64) >= np.uint64(min_contact)
    u, v = np.asarray(u, np.uint64)[keep], np.asarray(v, np.uint64)[keep]
    if len(u) == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint64)

    nodes = np.unique(np.concatenate([u, v]))
    iu = np.searchsorted(nodes, u)
    iv = np.searchsorted(nodes, v)
    g = coo_matrix((np.ones(len(iu), np.uint8), (iu, iv)),
                   shape=(len(nodes), len(nodes)))
    ncomp, labels = connected_components(g, directed=False)
    roots = np.full(ncomp, np.iinfo(np.uint64).max, np.uint64)
    np.minimum.at(roots, labels, nodes)
    return nodes, roots[labels]


def write_lut_npz(path, nodes, roots):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = np.argsort(nodes)
    np.savez_compressed(path, nodes=np.asarray(nodes, np.uint64)[order],
                        roots=np.asarray(roots, np.uint64)[order])


def read_lut_npz(path):
    d = np.load(path)
    return d["nodes"], d["roots"]


def apply_lut(arr, nodes, roots):
    """
    Map fragment ids to segment ids. Ids absent from the LUT (fragments with no
    surviving edge) keep their own id, which is also what makes the mapping
    idempotent.
    """
    out = np.asarray(arr, np.uint64).copy()
    nz = out != 0
    if not nz.any() or len(nodes) == 0:
        return out
    vals = out[nz]
    idx = np.clip(np.searchsorted(nodes, vals), 0, len(nodes) - 1)
    hit = nodes[idx] == vals
    out[nz] = np.where(hit, roots[idx], vals)
    return out


# ------------------------------------------------------------------ merge tree
class MergeTree:
    """
    Score at which two fragments were merged, from a waterz merge history.

    waterz reuses one of the inputs as the output id (`c` is usually `a`), so the
    history is not a parent forest as-is. Every merge therefore gets its own
    virtual node, and find_merge(u, v) is the score of the lowest common
    ancestor -- i.e. the bottleneck score on the path between u and v (a Kruskal
    tree over merges in increasing score order). Same role as
    lsd/post/merge_tree.pyx.
    """

    def __init__(self):
        self._parent = {}
        self._score = {}

    def _top(self, x):
        """Root of x's current component, walking from the leaf."""
        while x in self._parent:
            x = self._parent[x]
        return x

    def merge(self, a, b, c, score):
        ra, rb = self._top(a), self._top(b)
        node = ("m", len(self._score))
        self._score[node] = float(score)
        self._parent[ra] = node
        if rb != ra:
            self._parent[rb] = node
        if c != a and c != b and c not in self._parent:
            self._parent[c] = node          # waterz minted a fresh output id

    def find_merge(self, u, v):
        """Score at which u and v first ended up in the same region, or None."""
        seen = set()
        x = u
        while True:
            seen.add(x)
            if x not in self._parent:
                break
            x = self._parent[x]
        y = v
        while True:
            if y in seen:
                return self._score.get(y)
            if y not in self._parent:
                return None
            y = self._parent[y]


def contact_counts(frag_zyx):
    """
    Voxel-face contact area between neighbouring fragments.

    Returns dict {(min_id, max_id): n_faces}. Used for `min_contact`: a pair
    touching over a handful of voxels is not evidence of anything (the fib_c
    49363 merge rested on 38 voxels of membrane).
    """
    counts = {}
    for d in range(3):
        lo_sl = tuple(slice(0, -1) if i == d else slice(None) for i in range(3))
        hi_sl = tuple(slice(1, None) if i == d else slice(None) for i in range(3))
        a = frag_zyx[lo_sl]
        b = frag_zyx[hi_sl]
        m = (a != b) & (a != 0) & (b != 0)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.uint64)
        hi = np.maximum(a[m], b[m]).astype(np.uint64)
        key = np.stack([lo, hi], axis=1)
        uniq, cnt = np.unique(key, axis=0, return_counts=True)
        for (u, v), c in zip(uniq.tolist(), cnt.tolist()):
            k = (u, v)
            counts[k] = counts.get(k, 0) + int(c)
    return counts
