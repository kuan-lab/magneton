#!/usr/bin/env python
"""Trace why two points (e.g. an axon and a dendrite) share one final instance ID.

Stage 1 (this script): cross-block stitching.
  - final ID at each point
  - block-local waterz ID (as global gid = local + offset) at each point, for every
    block whose extent contains it
  - if both points carry the SAME gid in some block -> merge happened INSIDE that
    block (waterz agglomeration or a supervoxel straddling the membrane)
  - otherwise walk unions.txt: shortest union path A -> B, plus overlap stats for every
    union edge in the merged component (recomputed exactly as select_pairs sees them)

Coordinates are mip-0 voxels of the instance volume (x,y,z).

Run from /gpfs/radev/home/yf354 in the magneton env:
  python trace_merge.py --config magneton/instance_segmentation/configs/<cfg>.yaml \
      --a X,Y,Z --b X,Y,Z [--contact X,Y,Z ...]
"""
import argparse
import bisect
import glob
import json
import os
from collections import defaultdict, deque

import numpy as np
import yaml
from cloudvolume import CloudVolume


def parse_pt(s):
    return tuple(int(round(float(v))) for v in s.replace(",", " ").split())


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--id", type=int, default=None, help="final ID: stats for every union edge in its pool (no points needed)")
ap.add_argument("--a", default=None, help="x,y,z inside segment A")
ap.add_argument("--b", default=None, help="x,y,z inside segment B")
ap.add_argument("--contact", action="append", default=[], help="x,y,z of a contact site (repeatable)")
ap.add_argument("--coord-nm", type=float, default=None,
                help="resolution (nm) of the given coords, e.g. 4 for EM voxels; converted to the volume's mip-0")
ap.add_argument("--root", default="/gpfs/radev/home/yf354", help="base for relative config paths")
ap.add_argument("--zstep", type=int, default=128, help="z slab size when reading overlaps")
ap.add_argument("--max-edges", type=int, default=1000,
                help="max union edges to recompute stats for (min cut needs all edges of the component)")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))


def rel(p):
    return p if os.path.isabs(p) else os.path.join(args.root, p)


meta_dir = rel(cfg["segmentation_stage"]["metadata_dir"])
merge_dir = rel(cfg["checkpoint"]["merge_dir"])
ms = cfg.get("merge_stage", {})
min_ov = int(ms.get("min_overlap_vox", 20))
min_fl = float(ms.get("min_frac_local", 0.7))
min_fg = float(ms.get("min_frac_global", 0.7))
min_iou = float(ms.get("min_iou", 0.7))

blocks = {}
for f in glob.glob(os.path.join(meta_dir, "block_*.json")):
    m = json.load(open(f))
    blocks[int(m["index"])] = m
offs = {int(k): int(v) for k, v in json.load(open(os.path.join(merge_dir, "global_offsets.json")))["offsets"].items()}
order = sorted((offs[i], i) for i in blocks)
starts = [o for o, _ in order]


def block_of(g):
    """gid -> (block index, local id). gids of block i are (off_i, off_i + max_id_i]."""
    k = bisect.bisect_left(starts, g) - 1
    o, i = order[k]
    return i, g - o


adj = defaultdict(set)
n_unions = 0
with open(os.path.join(merge_dir, "unions.txt")) as fh:
    for line in fh:
        p = line.split()
        if len(p) == 2:
            a, b = int(p[0]), int(p[1])
            adj[a].add(b)
            adj[b].add(a)
            n_unions += 1

_vols = {}


def vol(path):
    if path not in _vols:
        _vols[path] = CloudVolume(path, mip=0, progress=False, bounded=False, fill_missing=True)
    return _vols[path]


def val(path, p):
    x, y, z = p
    return int(np.asarray(vol(path)[x:x + 1, y:y + 1, z:z + 1]).ravel()[0])


def blocks_containing(p):
    x, y, z = p
    out = []
    for i, m in sorted(blocks.items()):
        z1, z2, y1, y2, x1, x2 = m["coords"]
        if z1 <= z < z2 and y1 <= y < y2 and x1 <= x < x2:
            out.append(i)
    return out


def gids_at(p):
    res = {}
    for i in blocks_containing(p):
        loc = val(blocks[i]["path"], p)
        res[i] = (loc, loc + offs[i] if loc else 0)
    return res


out_path = cfg["paths"]["output"]
if args.coord_nm:
    _res = vol(out_path).resolution
    _raw_parse = parse_pt

    def parse_pt(s):
        return tuple(int(v * args.coord_nm // r) for v, r in zip(_raw_parse(s), _res))

print(f"[cfg] {args.config}\n[merge] {n_unions} union pairs; thresholds min_overlap_vox={min_ov} "
      f"min_frac_local={min_fl} min_frac_global={min_fg} min_iou={min_iou} (min_iou NOT enforced by select_pairs)")
if args.id is not None:
    ga, gb, A, B = {}, {}, None, None
    if args.a and args.b:
        raise SystemExit("use either --id or --a/--b")
elif args.a and args.b:
    A, B = parse_pt(args.a), parse_pt(args.b)
    fa, fb = val(out_path, A), val(out_path, B)
    print(f"\nfinal ID at A{A} = {fa}   at B{B} = {fb}")
    if fa != fb or fa == 0:
        print("-> points do NOT share a final ID; nothing to trace (check coordinates are mip-0 voxels of this volume).")
        raise SystemExit
    ga, gb = gids_at(A), gids_at(B)
else:
    raise SystemExit("need --id or both --a and --b")

for name, p, g in (("A", A, ga), ("B", B, gb)) if A else ():
    print(f"\n{name}{p} lies in blocks {list(g)}:")
    for i, (loc, gid) in g.items():
        overlap_note = " (overlap zone)" if len(g) > 1 else ""
        print(f"   block {i:3d}: local id {loc:8d} -> gid {gid}{overlap_note}")

for c in map(parse_pt, args.contact):
    g = gids_at(c)
    zone = "OVERLAP ZONE (in %d blocks)" % len(g) if len(g) > 1 else "block core only"
    print(f"\ncontact {c}: final={val(out_path, c)}  {zone}")
    for i, (loc, gid) in g.items():
        print(f"   block {i:3d}: local {loc} -> gid {gid}")

setA = {gid for _, gid in ga.values() if gid} if A else {args.id}
setB = {gid for _, gid in gb.values() if gid} if A else set()
same = setA & setB
if same:
    print("\n==> SAME block-local segment contains both points:")
    for g in sorted(same):
        i, loc = block_of(g)
        print(f"    block {i} local id {loc} (gid {g})")
    print("    The merge happened INSIDE the block (waterz agglomeration, or a watershed supervoxel "
          "straddling the membrane). Next: re-run that block with supervoxels + merge history.")

# Connected component through unions, starting from A's gids
comp = set(setA)
q = deque(setA)
while q:
    g = q.popleft()
    for n in adj.get(g, ()):
        if n not in comp:
            comp.add(n)
            q.append(n)
by_block = defaultdict(list)
for g in comp:
    i, loc = block_of(g)
    by_block[i].append(loc)
print(f"\nunion component: {len(comp)} block-level segments across {len(by_block)} blocks")
for i in sorted(by_block):
    print(f"   block {i:3d}: local ids {sorted(by_block[i])}")

# Shortest union path A -> B
path = []
if A and not same:
    prev = {g: None for g in setA}
    q = deque(setA)
    hit = None
    while q:
        g = q.popleft()
        if g in setB:
            hit = g
            break
        for n in adj.get(g, ()):
            if n not in prev:
                prev[n] = g
                q.append(n)
    while hit is not None:
        path.append(hit)
        hit = prev[hit]
    path.reverse()
    print("\nshortest union path A -> B:")
    for g in path:
        i, loc = block_of(g)
        print(f"   gid {g}  (block {i}, local {loc})")
    if not path:
        print("   none found — A's and B's points may sit in blocks whose ids are joined only via a "
              "within-block merge elsewhere; pick A and B near the same contact.")

# Overlap stats for union edges inside the component (grouped per block pair, one slab read each)
edges = sorted({(min(a, b), max(a, b)) for a in comp for b in adj.get(a, ()) if b in comp})
all_edges = list(edges)
path_edges = {(min(a, b), max(a, b)) for a, b in zip(path, path[1:])}
if len(edges) > args.max_edges:
    print(f"\n{len(edges)} edges in component > --max-edges; computing path edges only")
    edges = sorted(path_edges)
groups = defaultdict(list)
for a, b in edges:
    ia, la = block_of(a)
    ib, lb = block_of(b)
    if ia > ib:
        (ia, la, a), (ib, lb, b) = (ib, lb, b), (ia, la, a)
    groups[(ia, ib)].append((a, la, b, lb))

rows = []
for (ia, ib), es in groups.items():
    ca, cb = blocks[ia]["coords"], blocks[ib]["coords"]
    z1, z2 = max(ca[0], cb[0]), min(ca[1], cb[1])
    y1, y2 = max(ca[2], cb[2]), min(ca[3], cb[3])
    x1, x2 = max(ca[4], cb[4]), min(ca[5], cb[5])
    va, vb = vol(blocks[ia]["path"]), vol(blocks[ib]["path"])
    st = {e: dict(c=0, ta=0, tb=0, s=np.zeros(3), lo=np.full(3, 1 << 30), hi=np.full(3, -1)) for e in es}
    for zz in range(z1, z2, args.zstep):
        ze = min(zz + args.zstep, z2)
        sa = np.asarray(va[x1:x2, y1:y2, zz:ze])[..., 0]
        sb = np.asarray(vb[x1:x2, y1:y2, zz:ze])[..., 0]
        nz = (sa != 0) & (sb != 0)
        for e in es:
            _, la, _, lb = e
            ma = (sa == la) & nz
            mb = (sb == lb) & nz
            both = ma & mb
            d = st[e]
            d["ta"] += int(ma.sum())
            d["tb"] += int(mb.sum())
            n = int(both.sum())
            if n:
                idx = np.stack(np.nonzero(both), 1) + np.array([x1, y1, zz])
                d["c"] += n
                d["s"] += idx.sum(0)
                d["lo"] = np.minimum(d["lo"], idx.min(0))
                d["hi"] = np.maximum(d["hi"], idx.max(0))
    for e in es:
        a, la, b, lb = e
        d = st[e]
        c = d["c"]
        fa_ = c / d["ta"] if d["ta"] else 0
        fb_ = c / d["tb"] if d["tb"] else 0
        iou = c / (d["ta"] + d["tb"] - c) if (d["ta"] + d["tb"] - c) else 0
        cen = (d["s"] / c).round().astype(int).tolist() if c else None
        rows.append((iou, ia, la, ib, lb, c, d["ta"], d["tb"], fa_, fb_, cen, d["lo"].tolist(), d["hi"].tolist(),
                     (min(a, b), max(a, b)) in path_edges, a, b))

print("\nunion edges (sorted by IoU; '*' = on A->B path; '!' = would FAIL min_iou or needs only ONE frac):")
print(f"{'':2}{'blkA':>5} {'locA':>8} {'blkB':>5} {'locB':>8} {'overlap':>9} {'totA':>9} {'totB':>9} "
      f"{'fracA':>6} {'fracB':>6} {'IoU':>5}  centroid_xyz  bbox_xyz")
for iou, ia, la, ib, lb, c, ta, tb, fa_, fb_, cen, lo, hi, onp, _, _ in sorted(rows, key=lambda r: r[0]):
    weak = iou < min_iou or not (fa_ >= min_fl and fb_ >= min_fg)
    flag = ("*" if onp else " ") + ("!" if weak else " ")
    print(f"{flag}{ia:5d} {la:8d} {ib:5d} {lb:8d} {c:9d} {ta:9d} {tb:9d} {fa_:6.3f} {fb_:6.3f} {iou:5.2f}  {cen}  {lo}-{hi}")

# ---- Source of the merge: IoU-weighted min cut between A's and B's block segments.
# One union anywhere is enough to merge, so the contact the user looked at need not be
# the source. The cheapest set of unions whose removal separates A from B is.
if A and not same and rows:
    if len(rows) < len(all_edges):
        print(f"\n[WARN] only {len(rows)}/{len(all_edges)} edges have stats; min cut is over a partial graph "
              f"(raise --max-edges)")
    INF = float("inf")
    cap = defaultdict(float)
    nbr = defaultdict(set)
    for r in rows:
        a, b = r[14], r[15]
        w = max(r[0], 1e-3)
        cap[(a, b)] += w
        cap[(b, a)] += w
        nbr[a].add(b)
        nbr[b].add(a)
    S, T = "S", "T"
    for g in setA:
        cap[(S, g)] = INF
        nbr[S].add(g)
        nbr[g].add(S)
    for g in setB:
        cap[(g, T)] = INF
        nbr[g].add(T)
        nbr[T].add(g)
    flow = defaultdict(float)

    def resid(u, v):
        return cap[(u, v)] - flow[(u, v)]

    def bfs_prev():
        prev = {S: None}
        q = deque([S])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if v not in prev and resid(u, v) > 1e-12:
                    prev[v] = u
                    q.append(v)
        return prev

    total = 0.0
    while True:
        prev = bfs_prev()
        if T not in prev:
            break
        bott, v = INF, T
        while prev[v] is not None:
            bott = min(bott, resid(prev[v], v))
            v = prev[v]
        v = T
        while prev[v] is not None:
            flow[(prev[v], v)] += bott
            flow[(v, prev[v])] -= bott
            v = prev[v]
        total += bott
    reach = set(bfs_prev())
    cut = sorted((r for r in rows if (r[14] in reach) != (r[15] in reach)), key=lambda r: r[0])
    scale = (vol(out_path).resolution / args.coord_nm) if args.coord_nm else None
    print(f"\n==> MERGE SOURCE (IoU-weighted min cut A|B, total weight {total:.3f}): {len(cut)} union(s) "
          f"must ALL be removed to separate A from B")
    for iou, ia, la, ib, lb, c, ta, tb, fa_, fb_, cen, lo, hi, _, _, _ in cut:
        loc = f"centroid8 {cen}"
        if scale is not None and cen:
            loc += f"  centroid{args.coord_nm:g}nm {[int(v * s) for v, s in zip(cen, scale)]}"
        kind = "WEAK stitch" if iou < min_iou else "strong match (merge likely INSIDE an adjacent block segment)"
        print(f"   {ia}/{la} <-> {ib}/{lb}  overlap {c}  fracs {fa_:.3f}/{fb_:.3f}  IoU {iou:.2f}  {loc}  [{kind}]")
