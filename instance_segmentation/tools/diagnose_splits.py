# -*- coding: utf-8 -*-
"""
For each labelled split, report the threshold at which it would actually heal.

A grid sweep tells you what a parameter set does globally; it does not tell you
whether a given split error is reachable by ANY parameter set. This does.

For a pair of fragments the answer is the *bottleneck* of the best path between
them in the region graph: the smallest value T such that keeping every edge with
score <= T connects them. That is the minimax path weight, read off a minimum
spanning forest. Three outcomes:

  finite T <= pass-2 cap   tunable -- raising the pass-3 threshold to T heals it
  T = inf (disconnected)   NOT tunable -- waterz refused every path between the
                           two fragments, so the predicted membrane between them
                           is solid. This is an affinity/model problem, and no
                           stitching parameter will fix it.

Run it on the same labels file the sweep uses:

    python -m magneton.instance_segmentation.tools.diagnose_splits \
        --config magneton/instance_segmentation/configs/config_<vol>.yaml \
        --labels magneton/instance_segmentation/configs/labels_<vol>.yaml

The keep_apart pairs get the same treatment, and their bottlenecks are the
ceiling: any threshold at or above the SMALLEST keep-apart bottleneck starts
creating merge errors. If the largest merge bottleneck you need exceeds the
smallest keep-apart bottleneck, no single threshold separates the two cases and
the graph itself has to improve.
"""
import argparse
from collections import deque

import numpy as np

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.utils.graph_utils import (
    read_edges_dir, dedupe_edges,
)
from magneton.instance_segmentation.tools.sweep_merge import load_labels


def build_forest(u, v, s, c, min_contact=0):
    """Minimum spanning forest over the finite-score edges, as an adjacency list."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    keep = np.isfinite(s)
    if min_contact:
        keep &= np.asarray(c, np.uint64) >= np.uint64(min_contact)
    u, v, s = u[keep], v[keep], s[keep].astype(np.float64)
    nodes = np.unique(np.concatenate([u, v]))
    iu = np.searchsorted(nodes, u)
    iv = np.searchsorted(nodes, v)
    # MST drops zero weights, so shift: waterz scores cluster at ~0.006 but can be 0
    w = s + 1.0
    g = coo_matrix((w, (iu, iv)), shape=(len(nodes), len(nodes)))
    mst = minimum_spanning_tree(g).tocoo()
    adj = [[] for _ in range(len(nodes))]
    for a, b, ww in zip(mst.row, mst.col, mst.data):
        adj[a].append((b, ww - 1.0))
        adj[b].append((a, ww - 1.0))
    return nodes, adj


def bottleneck(nodes, adj, a, b):
    """Max edge weight on the forest path a->b, or inf if they aren't connected."""
    ia = np.searchsorted(nodes, np.uint64(a))
    ib = np.searchsorted(nodes, np.uint64(b))
    if ia >= len(nodes) or nodes[ia] != np.uint64(a):
        return float("inf"), "u_absent"
    if ib >= len(nodes) or nodes[ib] != np.uint64(b):
        return float("inf"), "v_absent"
    if ia == ib:
        return 0.0, "same"
    seen = {int(ia): 0.0}
    q = deque([int(ia)])
    while q:
        n = q.popleft()
        for m, w in adj[n]:
            if m in seen:
                continue
            seen[m] = max(seen[n], float(w))
            if m == ib:
                return seen[m], "path"
            q.append(m)
    return float("inf"), "disconnected"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-contact", dest="min_contact", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    groups, apart = load_labels(args.labels, cfg)
    if not (groups or apart):
        raise SystemExit("labels file has no merge_groups and no keep_apart")

    edges_dir = cfg.get("edges_stage", {}).get("edges_dir", "magneton/metadata/edges")
    stage_cfg = get_stage_config(cfg, "global_merge")
    cap = float(cfg.get("edges_stage", {}).get("score_threshold", 0.45))

    u, v, s, c = read_edges_dir(edges_dir)
    u, v, s, c = dedupe_edges(u, v, s, c, how=stage_cfg.get("dedupe", "max"))
    print(f"[INFO] {len(u)} unique pairs; pass-2 cap = {cap}")
    nodes, adj = build_forest(u, v, s, c, min_contact=args.min_contact)
    print(f"[INFO] spanning forest over {len(nodes)} fragments "
          f"(min_contact={args.min_contact})")

    need = []
    print("\n--- merge_groups: threshold needed to heal each split ---")
    for gi, g in enumerate(groups):
        anchor = g[0]
        for other in g[1:]:
            t, why = bottleneck(nodes, adj, anchor, other)
            need.append(t)
            verdict = ("already merged" if t == 0 else
                       f"needs threshold >= {t:.3f}" if np.isfinite(t) else
                       f"UNREACHABLE ({why}) -- no threshold heals this")
            print(f"  group {gi}: {anchor} <-> {other}: {verdict}")

    ceil = []
    print("\n--- keep_apart: threshold at which each pair would wrongly fuse ---")
    for gi, g in enumerate(apart):
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                t, why = bottleneck(nodes, adj, g[i], g[j])
                ceil.append(t)
                verdict = ("ALREADY FUSED" if t == 0 else
                           f"fuses at threshold >= {t:.3f}" if np.isfinite(t) else
                           "safe at any threshold (disconnected)")
                print(f"  pair {gi}: {g[i]} <-> {g[j]}: {verdict}")

    print("\n--- verdict ---")
    fin = [t for t in need if np.isfinite(t) and t > 0]
    unreach = sum(1 for t in need if not np.isfinite(t))
    print(f"  labelled splits: {len(need)}  "
          f"({len(fin)} tunable, {unreach} unreachable, "
          f"{sum(1 for t in need if t == 0)} already merged)")
    if unreach:
        print(f"  {unreach} split(s) cannot be fixed by ANY pass-3 threshold: the "
              f"predicted membrane between those fragments is solid. Improve the "
              f"affinity, the watershed, or the merge function -- not the stitching.")
    if fin:
        print(f"  tunable splits need threshold >= {max(fin):.3f} "
              f"(max over {len(fin)})")
        if max(fin) > cap:
            print(f"  ... which is ABOVE the pass-2 cap {cap}: re-run pass 2 with "
                  f"score_threshold > {max(fin):.3f} before sweeping.")
    safe = [t for t in ceil if np.isfinite(t)]
    if safe:
        print(f"  first wrong fusion at threshold {min(safe):.3f}")
        if fin and max(fin) >= min(safe):
            print("  ** NO SAFE THRESHOLD: the worst split needs more merging than "
                  "the easiest false merge tolerates. The region graph has to "
                  "improve; a threshold alone cannot separate these cases. **")
        elif fin:
            lo, hi = max(fin), min(safe)
            print(f"  ** usable window: [{lo:.3f}, {hi:.3f}) -- "
                  f"suggest threshold {(lo + hi) / 2:.3f} **")
    elif fin:
        print(f"  no keep-apart pair is reachable, so anything >= {max(fin):.3f} is safe "
              f"on the labelled cases.")


if __name__ == "__main__":
    main()
