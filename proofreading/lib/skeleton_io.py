"""
Skeleton I/O for the proofreading module.

- load_seg:    read an instance segmentation (tif or precomputed) as (x,y,z).
- skeletonize: kimimaro per-instance skeletonization (+ optional postprocess).
- write_nml:   skeletons -> WebKnossos NML (voxel coords, scale = voxel size).
- parse_nml:   WebKnossos NML -> {label: (N,3) xyz} (stdlib only — safe to import
               in the nnInteractive env, which lacks kimimaro/wknml).

Coordinates are (x,y,z) voxels throughout, matching WebKnossos and the rest of
magneton. kimimaro/wknml/cloudvolume imports are deferred into the functions
that need them so this module imports cleanly in any env.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np


def _patch_kimimaro_trace():
    """kimimaro 5.8.0 + numpy>=1.24: intake.skeletonize_subset does
    `skeleton.vertices += roi.minpt` (uint32 += int64) which strict same_kind
    casting rejects. Wrap trace() to return float vertices. Idempotent."""
    import kimimaro.trace
    if getattr(kimimaro.trace.trace, "_float_patched", False):
        return
    _orig = kimimaro.trace.trace

    def _trace_float(*a, **k):
        sk = _orig(*a, **k)
        sk.vertices = sk.vertices.astype(np.float64)
        return sk

    _trace_float._float_patched = True
    kimimaro.trace.trace = _trace_float


def load_seg(path: str, source: str = "tif", mip: int = 0, res_nm=None):
    """Return (seg_xyz: np.ndarray, res_nm: tuple). `path` may carry file://."""
    if source == "tif":
        import tifffile
        p = path[len("file://"):] if path.startswith("file://") else path
        seg = tifffile.imread(p).transpose(2, 1, 0)        # (z,y,x) -> (x,y,z)
        if res_nm is None:
            raise ValueError("res_nm required for tif source")
        return seg, tuple(float(v) for v in res_nm)
    elif source == "precomputed":
        from cloudvolume import CloudVolume
        cv = CloudVolume(path, mip=mip, parallel=1, progress=False, fill_missing=True)
        seg = np.asarray(cv[:])[..., 0]                    # (x,y,z)
        rn = res_nm or tuple(float(v) for v in cv.info["scales"][mip]["resolution"])
        return seg, tuple(float(v) for v in rn)
    raise ValueError(f"unknown source: {source}")


def skeletonize(seg, res_nm, *, dust_threshold=100, parallel=8,
                parallel_chunk_size=25, fix_branching=True, fix_borders=True,
                teasar=None, postprocess=None, downsample_nodes=0):
    """Run kimimaro on every instance label. Returns {label: cloudvolume.Skeleton}.
    Masking to specific labels is the caller's job (kimimaro's object_ids path is
    buggy in 5.8.0 — the whole-volume path is what works, see _patch_kimimaro_trace)."""
    import kimimaro
    _patch_kimimaro_trace()

    kwargs = dict(anisotropy=tuple(float(v) for v in res_nm),
                  dust_threshold=dust_threshold, fix_branching=fix_branching,
                  fix_borders=fix_borders, progress=False,
                  parallel=parallel, parallel_chunk_size=parallel_chunk_size)
    if teasar:
        kwargs["teasar_params"] = dict(teasar)
    skels = kimimaro.skeletonize(seg, **kwargs)

    if postprocess and postprocess.get("enable"):
        skels = kimimaro.postprocess(
            skels,
            dust_threshold=postprocess.get("dust_threshold", 0),
            tick_threshold=postprocess.get("tick_threshold", 0),
        )
    if downsample_nodes and downsample_nodes > 1:
        for sk in skels.values():
            sk.downsample(downsample_nodes)
    return skels


def write_nml(skels, res_nm, out_path: str, name: str = "proofreading",
              point_trees=None):
    """Write {label: Skeleton} to a WebKnossos NML. Vertices (nm) -> voxel coords.

    point_trees: optional list of node clouds to append, each a dict
        {'name': str, 'points': (N,3) xyz VOXEL coords, 'color': (r,g,b,a) opt,
         'connect': bool}. Used for the membrane "fishnet" — nodes a human edits
        in WKS and the expand stage routes to negative prompts (name-prefixed,
        e.g. 'membrane_net'). Points are already in voxels, so they are NOT
        rescaled. connect=True links them into one spanning tree (WebKnossos
        splits an edgeless cloud into one tree per node on upload).
    """
    import wknml
    rx = float(res_nm[0])
    trees, nid = [], 1
    for ti, (lab, sk) in enumerate(sorted(skels.items()), start=1):
        pos = sk.vertices / np.array(res_nm)               # nm -> voxels (x,y,z)
        base = nid
        nodes = [wknml.Node(id=base + i, position=tuple(float(c) for c in p),
                            radius=float(r))
                 for i, (p, r) in enumerate(zip(pos, sk.radius))]
        edges = [wknml.Edge(source=base + int(a), target=base + int(b))
                 for a, b in sk.edges]
        nid += len(nodes)
        col = tuple(np.random.rand(3).tolist()) + (1.0,)
        trees.append(wknml.Tree(id=ti, color=col, name=f"neuron_{int(lab)}",
                                nodes=nodes, edges=edges))
    for pt in (point_trees or []):
        pts = np.asarray(pt["points"], dtype=float).reshape(-1, 3)
        base = nid
        col = pt.get("color", (1.0, 0.0, 0.0, 1.0))        # membrane net = red
        nodes = [wknml.Node(id=base + i, position=tuple(float(c) for c in p),
                            radius=1.0)
                 for i, p in enumerate(pts)]
        edges = []
        if pt.get("connect"):                               # span into one tree
            from magneton.proofreading.lib.membrane_points import fishnet_edges
            edges = [wknml.Edge(source=base + int(a), target=base + int(b))
                     for a, b in fishnet_edges(pts)]
        nid += len(nodes)
        trees.append(wknml.Tree(id=len(trees) + 1, color=col,
                                name=pt.get("name", "membrane_net"),
                                nodes=nodes, edges=edges))
    nml = wknml.NML(
        parameters=wknml.NMLParameters(name=name, scale=tuple(float(v) for v in res_nm)),
        trees=trees, branchpoints=[], comments=[], groups=[],
    )
    with open(out_path, "wb") as f:
        wknml.write_nml(f, nml)
    return dict(trees=len(trees), nodes=sum(len(t.nodes) for t in trees))


def parse_nml(path: str):
    """WebKnossos NML -> list of per-tree records (stdlib only):
        {'tree_id': int, 'name': str, 'label': int, 'points': (N,3) xyz}

    Keyed PER TREE, not per name: when a human splits a merge error in
    WebKnossos the two halves share the original name (e.g. two 'neuron_115'),
    so collapsing by name would undo the split. tree_id is unique per NML and is
    what the expand stage uses as the output label, so split neurons become two
    distinct segments — the whole point of the correction.
    """
    root = ET.parse(path).getroot()
    out = []
    for thing in root.findall("thing"):
        tid = int(thing.get("id"))
        name = thing.get("name") or f"tree_{tid}"
        try:
            lab = int(name.split("_")[1])
        except (IndexError, ValueError):
            lab = tid
        nodes = thing.find("nodes")
        if nodes is None:
            continue
        pts = [(float(n.get("x")), float(n.get("y")), float(n.get("z")))
               for n in nodes.findall("node")]
        if pts:
            out.append({"tree_id": tid, "name": name, "label": lab,
                        "points": np.asarray(pts, dtype=float)})
    return out
