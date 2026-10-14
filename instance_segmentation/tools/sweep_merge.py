# -*- coding: utf-8 -*-
"""
Sweep pass-3 parameters (threshold x min_contact) and score each point against
hand-labelled merge/split cases.

Why this is cheap: passes 1-2 produce the fragments and the scored region graph;
pass 3 is just "threshold the graph and take connected components". So a whole
sweep touches no image data. Scoring likewise runs on the LUT alone -- no
relabel pass is needed to find the best parameters, only to render the winner.

    python -m magneton.instance_segmentation.tools.sweep_merge \
        --config magneton/instance_segmentation/configs/config_<vol>.yaml \
        --thresholds 0.3,0.4,0.5,0.6,0.7 --min-contacts 0,4,20 \
        --labels magneton/instance_segmentation/configs/labels_<vol>.yaml

Then render the winner:

    python -m magneton.instance_segmentation.main --stage relabel-hpc --yes \
        --config <same config> --lut <best lut> --wait

CEILING: pass 2 records a pair waterz REFUSED to merge as score=inf, not its raw
score, so no threshold above edges_stage.score_threshold can merge anything new.
To explore more merging (i.e. to fix split errors) re-run pass 2 once with a
higher score_threshold; pass 1 fragments stay valid and are not recomputed.
"""
import argparse
import itertools
import os

import numpy as np
import yaml

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.utils.graph_utils import (
    read_edges_dir, dedupe_edges, components_from_edges, write_lut_npz,
)


# ----------------------------------------------------------------- label input
def _resolve_point(frag_vol, point):
    """[z, y, x] -> the fragment id at that voxel."""
    z, y, x = (int(v) for v in point)
    return int(np.asarray(frag_vol[x:x + 1, y:y + 1, z:z + 1])[0, 0, 0, 0])


def _resolve_id(item, frag_vol_getter):
    """A label entry is either a fragment/segment id or a [z, y, x] point."""
    if isinstance(item, (list, tuple)):
        if len(item) != 3:
            raise ValueError(f"point must be [z, y, x], got {item}")
        return _resolve_point(frag_vol_getter(), item)
    return int(item)


def load_labels(path, cfg):
    """
    YAML:
        merge_groups:          # every id in a group must end up in ONE segment
          - [49363, 12345]     #   ids, or [z, y, x] points into the fragments volume
        keep_apart:            # these must NOT share a segment
          - [49363, 22222]

    Ids are segment ids read off an instance volume produced from the SAME
    fragments volume; a segment id is the smallest fragment id in its component,
    so it is itself a valid fragment id and needs no translation.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    cache = {}

    def frag_vol():
        if "v" not in cache:
            from cloudvolume import CloudVolume
            cache["v"] = CloudVolume(cfg["paths"]["fragments"], mip=0,
                                     bounded=False, progress=False, fill_missing=True)
        return cache["v"]

    groups = [[_resolve_id(i, frag_vol) for i in g]
              for g in data.get("merge_groups", []) or []]
    apart = [[_resolve_id(i, frag_vol) for i in g]
             for g in data.get("keep_apart", []) or []]
    return groups, apart


# --------------------------------------------------------------------- scoring
def _root_lookup(nodes, roots, ids):
    """Segment id for each fragment id; ids absent from the LUT map to themselves."""
    out = np.asarray(ids, np.uint64).copy()
    if len(nodes) == 0:
        return out
    idx = np.clip(np.searchsorted(nodes, out), 0, len(nodes) - 1)
    hit = nodes[idx] == out
    return np.where(hit, roots[idx], out)


def score_lut(nodes, roots, groups, apart):
    """Pairs joined as asked / pairs wrongly joined, over the labelled cases."""
    merge_pairs = merge_hit = 0
    groups_complete = 0
    for g in groups:
        r = _root_lookup(nodes, roots, g)
        done = True
        for a, b in itertools.combinations(range(len(g)), 2):
            merge_pairs += 1
            if r[a] == r[b]:
                merge_hit += 1
            else:
                done = False
        if done and len(g) > 1:
            groups_complete += 1

    apart_pairs = apart_ok = 0
    for g in apart:
        r = _root_lookup(nodes, roots, g)
        for a, b in itertools.combinations(range(len(g)), 2):
            apart_pairs += 1
            if r[a] != r[b]:
                apart_ok += 1
    return dict(merge_pairs=merge_pairs, merge_hit=merge_hit,
                groups_complete=groups_complete, groups_total=len(groups),
                apart_pairs=apart_pairs, apart_ok=apart_ok)


def graph_stats(nodes, roots):
    if len(nodes) == 0:
        return dict(n_merged_fragments=0, n_segments=0, largest_component=0)
    _, sizes = np.unique(roots, return_counts=True)
    return dict(n_merged_fragments=int(len(nodes)),
                n_segments=int(len(sizes)),
                largest_component=int(sizes.max()))


# ------------------------------------------------------------------------ main
def sweep(cfg, thresholds, min_contacts, groups, apart, lut_dir=None,
          write_luts=True):
    stage_cfg = get_stage_config(cfg, "global_merge")
    edges_dir = cfg.get("edges_stage", {}).get("edges_dir", "magneton/metadata/edges")
    lut_dir = lut_dir or stage_cfg.get("lut_dir", "magneton/checkpoints/lut")

    # read + dedupe ONCE; global_merge() would redo both per grid point
    u, v, s, c = read_edges_dir(edges_dir)
    print(f"[INFO] {len(u)} raw edges from {edges_dir}")
    how = stage_cfg.get("dedupe", "max")
    u, v, s, c = dedupe_edges(u, v, s, c, how=how)
    finite = np.isfinite(s)
    print(f"[INFO] {len(u)} unique pairs (dedupe='{how}'); "
          f"{int(finite.sum())} with a finite score, "
          f"{int((~finite).sum())} refused by waterz (inf)")
    if finite.any():
        qs = np.percentile(s[finite], [1, 10, 50, 90, 99]).round(3).tolist()
        print(f"[INFO] finite score percentiles (1/10/50/90/99): {qs}")
        ceiling = float(s[finite].max())
        print(f"[INFO] highest finite score = {ceiling:.3f} "
              f"(pass-2 score_threshold caps this; thresholds above it add nothing)")

    rows = []
    for t, mc in itertools.product(thresholds, min_contacts):
        nodes, roots = components_from_edges(u, v, s, c, t, mc)
        row = dict(threshold=t, min_contact=mc)
        row.update(graph_stats(nodes, roots))
        row["n_edges_kept"] = int(((s <= t) & (c >= mc)).sum())
        if groups or apart:
            row.update(score_lut(nodes, roots, groups, apart))
        if write_luts:
            lut = os.path.join(lut_dir,
                               f"lut_t{str(t).replace('.', '')}_c{mc}.npz")
            write_lut_npz(lut, nodes, roots)
            row["lut"] = lut
        rows.append(row)
        print(f"[INFO] t={t:<6} c={mc:<4} "
              f"segments={row['n_segments']:<8} "
              f"merged_frags={row['n_merged_fragments']:<9} "
              f"largest={row['largest_component']:<8} "
              + (f"merges {row['merge_hit']}/{row['merge_pairs']} "
                 f"groups {row['groups_complete']}/{row['groups_total']} "
                 f"apart {row['apart_ok']}/{row['apart_pairs']}"
                 if groups or apart else ""))
    return rows


def _report(rows, groups, apart, out_csv=None):
    import csv
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[DONE] sweep table: {out_csv}")

    if not (groups or apart):
        return
    # rank: every keep-apart pair honoured first, then most labelled merges made
    def key(r):
        apart_clean = r.get("apart_pairs", 0) == r.get("apart_ok", 0)
        return (apart_clean, r.get("merge_hit", 0), -r.get("largest_component", 0))
    best = max(rows, key=key)
    print("\n[BEST] "
          f"threshold={best['threshold']} min_contact={best['min_contact']} "
          f"-> merges {best.get('merge_hit')}/{best.get('merge_pairs')}, "
          f"groups {best.get('groups_complete')}/{best.get('groups_total')}, "
          f"apart {best.get('apart_ok')}/{best.get('apart_pairs')}, "
          f"segments {best['n_segments']}")
    if best.get("lut"):
        print(f"[BEST] apply with: --stage relabel-hpc --yes --lut {best['lut']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="instance-seg stage config")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--min-contacts", dest="min_contacts", default="0,4,20")
    ap.add_argument("--labels", default=None, help="YAML with merge_groups / keep_apart")
    ap.add_argument("--lut-dir", dest="lut_dir", default=None)
    ap.add_argument("--no-luts", action="store_true", help="score only, write nothing")
    ap.add_argument("--csv", default=None, help="write the sweep table here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    min_contacts = [int(x) for x in args.min_contacts.split(",") if x.strip()]

    groups, apart = ([], [])
    if args.labels:
        groups, apart = load_labels(args.labels, cfg)
        print(f"[INFO] labels: {len(groups)} merge groups, {len(apart)} keep-apart sets")

    rows = sweep(cfg, thresholds, min_contacts, groups, apart,
                 lut_dir=args.lut_dir, write_luts=not args.no_luts)
    _report(rows, groups, apart, out_csv=args.csv)


if __name__ == "__main__":
    main()
