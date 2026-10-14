#!/usr/bin/env python
"""Cross-volume comparison of neurite classes from analysis/neurite_filter.py.

Two questions:
  1. Are the neurites themselves different?  Per-segment radius / volume per
     class (axon, dendrite), pairwise across volumes.
  2. Is the tissue composed differently?  Each volume is tiled into sub-blocks
     (default 6 x 4 x 1 at 16 nm, ~4 x 4 x 5 um) and the voxel fraction owned by
     axons / dendrites / other labeled segments / unlabeled is measured per
     sub-block. A single whole-volume fraction has n=1 and cannot be tested;
     sub-blocks give a within-volume spread to test against. Sub-blocks are
     spatially correlated, so treat p-values as indicative and lean on effect size.

Statistics match analysis/stages/cluster_compare.py: two-sided Mann-Whitney U +
Cliff's delta, Bonferroni over (3 pairs x all features tested).

Classes come from each volume's segments.csv (--radius-cut applied here, so the
cut can be varied without re-running the filter).
"""
import argparse, itertools, os
import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume
from scipy.stats import mannwhitneyu

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"
CLASSES = ["axon", "dendrite", "other", "unlabeled"]          # code 1, 2, 3, 0
COLORS = {"fib_b": "#2a78d6", "fib_c": "#eb6834", "fib_f": "#1baf7a"}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def load_segments(vol, suffix, mip, cut):
    d = pd.read_csv(os.path.join(FILT, f"{vol}_{suffix}_mip{mip}", "segments.csv"))
    d["cls"] = np.where(~d.neurite, "other",
                        np.where(d.radius_nm < cut, "axon", "dendrite"))
    d["vol"] = vol
    return d


def subblock_fractions(vol, suffix, mip, seg, grid):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{vol}_{suffix}", mip=mip,
                     progress=False, fill_missing=True)
    arr = np.asarray(cv[:, :, :][..., 0])
    code = {int(l): (1 if c == "axon" else 2 if c == "dendrite" else 3)
            for l, c in zip(seg.label, seg.cls)}
    code[0] = 0
    arr = fastremap.remap(arr, code, preserve_missing_labels=True).astype(np.uint8)
    rows = []
    edges = [np.linspace(0, s, g + 1).astype(int) for s, g in zip(arr.shape, grid)]
    for i, j, k in itertools.product(*(range(g) for g in grid)):
        blk = arr[edges[0][i]:edges[0][i+1], edges[1][j]:edges[1][j+1],
                  edges[2][k]:edges[2][k+1]]
        n = np.bincount(blk.ravel(), minlength=4)[:4] / blk.size
        rows.append({"vol": vol, "block": f"{i}_{j}_{k}", "axon": n[1],
                     "dendrite": n[2], "other": n[3], "unlabeled": n[0]})
    return pd.DataFrame(rows)


def pairwise(df, group_col, feats, vols):
    out = []
    for f in feats:
        for a, b in itertools.combinations(vols, 2):
            x = df.loc[df[group_col] == a, f].dropna()
            y = df.loc[df[group_col] == b, f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            out.append({"feature": f, "pair": f"{a} vs {b}", "n_a": len(x), "n_b": len(y),
                        "median_a": x.median(), "median_b": y.median(),
                        "p": p, "cliffs_delta": cliffs_delta(x, y)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vols", nargs="+", default=["fib_b", "fib_c", "fib_f"])
    ap.add_argument("--suffix", default="neuron_instances_graph_v1")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-cut", type=float, default=150.0)
    ap.add_argument("--grid", type=int, nargs=3, default=[6, 4, 1],
                    help="sub-blocks along x y z")
    ap.add_argument("--out", default=os.path.join(FILT, "compare_graph_v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    segs = {v: load_segments(v, args.suffix, args.mip, args.radius_cut) for v in args.vols}
    allseg = pd.concat(segs.values(), ignore_index=True)

    # ---- whole-volume composition (descriptive, n=1 per volume) ----
    summ = []
    for v, d in segs.items():
        tot = d.voxels.sum()
        cv = CloudVolume(f"precomputed://file://{ROOT}/{v}_{args.suffix}", mip=args.mip)
        box = int(np.prod(cv.volume_size))
        vum3 = box * np.prod(cv.resolution) * 1e-9
        row = {"vol": v, "tissue_um3": vum3, "labeled_frac": tot / box}
        for c in ("axon", "dendrite", "other"):
            dc = d[d.cls == c]
            row[f"{c}_n"] = len(dc)
            row[f"{c}_frac"] = dc.voxels.sum() / box
            row[f"{c}_n_per_1000um3"] = len(dc) / vum3 * 1000
        # Length density: a tube of radius r and volume V has length V / (pi r^2).
        # Uses the face-count radius (biased low ~1.3-1.5x), so absolute values are
        # inflated; ratios across volumes are what to read.
        for c in ("axon", "dendrite"):
            dc = d[d.cls == c]
            L = (dc.volume_um3 / (np.pi * (dc.radius_nm * 1e-3) ** 2)).sum()
            row[f"{c}_length_um_per_um3"] = L / vum3
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out, "composition_whole_volume.csv"), index=False)

    # ---- sub-block composition ----
    blocks = pd.concat([subblock_fractions(v, args.suffix, args.mip, segs[v], args.grid)
                        for v in args.vols], ignore_index=True)
    blocks["neurite"] = blocks.axon + blocks.dendrite
    blocks.to_csv(os.path.join(args.out, "composition_subblocks.csv"), index=False)

    # ---- tests ----
    tests = []
    for c in ("axon", "dendrite"):
        sub = allseg[allseg.cls == c].copy()
        for t in pairwise(sub, "vol", ["radius_nm", "volume_um3"], args.vols):
            t["feature"] = f"{c}:{t['feature']}"
            tests.append(t)
    for t in pairwise(blocks, "vol", ["axon", "dendrite", "neurite", "other", "unlabeled"],
                      args.vols):
        t["feature"] = f"block_frac:{t['feature']}"
        tests.append(t)
    tests = pd.DataFrame(tests)
    tests["p_bonf"] = np.minimum(tests.p * len(tests), 1.0)
    tests["sig"] = pd.cut(tests.p_bonf, [-1, 1e-3, 1e-2, 5e-2, 2],
                          labels=["***", "**", "*", "ns"])
    tests.to_csv(os.path.join(args.out, "pairwise_tests.csv"), index=False)

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    for ax, c in zip(axs, ["axon", "dendrite", "neurite", "other", "unlabeled"]):
        data = [blocks.loc[blocks.vol == v, c] * 100 for v in args.vols]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, v in zip(bp["boxes"], args.vols):
            patch.set_facecolor(COLORS.get(v, "0.6")); patch.set_alpha(0.35)
        for i, (v, y) in enumerate(zip(args.vols, data), 1):
            ax.scatter(np.full(len(y), i) + np.random.uniform(-0.12, 0.12, len(y)), y,
                       s=10, color=COLORS.get(v, "0.3"), zorder=3)
        ax.set_xticks(range(1, len(args.vols) + 1), args.vols)
        ax.set_title(f"{c} (% of sub-block)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "composition_subblocks.png"), dpi=130)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, c in zip(axs, ("axon", "dendrite")):
        sub = allseg[allseg.cls == c]
        bins = np.logspace(np.log10(sub.radius_nm.min()), np.log10(sub.radius_nm.max()), 40)
        for v in args.vols:
            ax.hist(sub.loc[sub.vol == v, "radius_nm"], bins=bins, histtype="step", lw=2,
                    density=True, color=COLORS.get(v, "0.3"),
                    label=f"{v} (n={(sub.vol == v).sum():,})")
        ax.set_xscale("log"); ax.set_title(f"{c} radius")
        ax.set_xlabel("radius = 2V/SA (nm, face-count)"); ax.set_ylabel("density")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "radius_by_class.png"), dpi=130)

    pd.set_option("display.width", 220)
    print(summ.round(4).T.to_string(header=False))
    print(tests.round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
