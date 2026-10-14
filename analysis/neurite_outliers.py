#!/usr/bin/env python
"""Which volume is the odd one out? Cross-volume outlier screen on the classified
neurites (analysis/neurite_classify.py must have run on every volume).

1. Odd-one-out screen (univariate). For each class x feature, volume X is flagged
   when it differs from BOTH others in the same direction (Bonferroni p < 0.05,
   same sign of Cliff's delta) and the other two differ from each other by less
   than X does. Heatmap value = mean delta of X vs the other two.
2. Multivariate separability. Per class, a cross-validated logistic regression on
   standardized features tells each pair of volumes apart; AUC 0.5 = the neurites
   are indistinguishable. Run with all features and without the split-off counts,
   which depend on each volume's segmentation split rate, not only on anatomy.
3. Tissue level, per sub-block (--grid, default 6 x 4 x 1): voxel fraction of
   axon / dendrite / glia / short / other, and length density (um of shaft path
   per um^3) of axons and dendrites from the skeleton main paths. Sub-blocks are
   spatially correlated, so read p-values as indicative and lean on effect size.

Outputs in --out: odd_one_out.csv / .png, separability.csv / .png,
subblocks.csv, tissue.png.
"""
import argparse, itertools, os, pickle
import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume
from scipy.stats import mannwhitneyu

from magneton.analysis.lib import neurite_geometry as ng

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"
COLORS = {"fib_b": "#2a78d6", "fib_c": "#eb6834", "fib_f": "#1baf7a"}
NAMES = {"fib_b": "fib_b (Basil, AU)", "fib_c": "fib_c (Cappa, YA)", "fib_f": "fib_f (Fennel, AI)"}
FEATURES = ["r_median_nm", "r_inscribed_nm", "ellipticity", "circularity", "aspect_ratio",
            "tortuosity", "straightness", "r_cv", "varicosity_per_um", "varicosity_peak_ratio",
            "bead_period_um", "protrusions_per_um", "split_neighbors_per_um", "spines_per_um",
            "surface_excess"]
SPLIT_CONFOUNDED = {"split_neighbors_per_um", "spines_per_um"}
MV_FEATURES = ["r_median_nm", "r_inscribed_nm", "ellipticity", "circularity", "aspect_ratio",
               "tortuosity", "straightness", "r_cv", "varicosity_per_um",
               "protrusions_per_um", "split_neighbors_per_um", "surface_excess"]
CODES = {"axon": 1, "dendrite": 2, "glia": 3, "short": 4, "other": 5}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def odd_one_out(tests, vols, groups):
    """tests: rows with group, feature, a, b, delta (a vs b), p_bonf."""
    rows = []
    for (g, f), t in tests.groupby(["group", "feature"]):
        get = {}
        for _, r in t.iterrows():
            get[(r.a, r.b)] = (r.delta, r.p_bonf)
            get[(r.b, r.a)] = (-r.delta, r.p_bonf)
        for x in vols:
            y, z = [v for v in vols if v != x]
            if (x, y) not in get or (x, z) not in get or (y, z) not in get:
                continue
            (dxy, pxy), (dxz, pxz), (dyz, _) = get[(x, y)], get[(x, z)], get[(y, z)]
            outlier = (pxy < 0.05 and pxz < 0.05 and np.sign(dxy) == np.sign(dxz)
                       and abs(dyz) < min(abs(dxy), abs(dxz)))
            rows.append({"group": g, "feature": f, "vol": x, "mean_delta_vs_others": (dxy + dxz) / 2,
                         "delta_other_pair": dyz, "outlier": outlier})
    return pd.DataFrame(rows)


def pairwise(df, group, feature, vols):
    out = []
    for a, b in itertools.combinations(vols, 2):
        x = df.loc[df.vol == a, feature].dropna()
        y = df.loc[df.vol == b, feature].dropna()
        if len(x) < 5 or len(y) < 5:
            continue
        out.append({"group": group, "feature": feature, "a": a, "b": b, "n_a": len(x), "n_b": len(y),
                    "median_a": x.median(), "median_b": y.median(),
                    "p": mannwhitneyu(x, y, alternative="two-sided").pvalue,
                    "delta": cliffs_delta(x, y)})
    return out


def separability(df, vols, feats, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    out = []
    X_all = df[feats].copy()
    for f in feats:                                      # tame heavy right tails
        if (X_all[f].dropna() > 0).all() and X_all[f].skew() > 2:
            X_all[f] = np.log(X_all[f])
    for a, b in itertools.combinations(vols, 2):
        m = df.vol.isin([a, b]) & X_all.notna().all(axis=1)
        X, y = X_all[m].values, (df.vol[m] == b).astype(int).values
        if min(np.bincount(y, minlength=2)) < 10:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        clf.fit(X, y)
        coef = clf[-1].coef_[0]
        top = sorted(zip(feats, coef), key=lambda t: -abs(t[1]))[:3]
        out.append({"a": a, "b": b, "n_a": int((y == 0).sum()), "n_b": int((y == 1).sum()),
                    "auc": roc_auc_score(y, p),
                    "top_features": "; ".join(f"{f} ({'+' if c > 0 else '-'}{b[-1]})" for f, c in top)})
    return out


def tissue_subblocks(vol, suffix, mip, d_morph, seg_df, grid):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{vol}_{suffix}", mip=mip,
                     progress=False, fill_missing=True, parallel=1)
    res = np.array(cv.resolution, dtype=float)
    arr = np.asarray(cv[:, :, :][..., 0])
    code = {int(l): CODES["other"] for l in seg_df.label}
    code.update({int(l): CODES[c] for l, c in zip(d_morph.label, d_morph.cls)})
    code[0] = 0
    arr = fastremap.remap(arr, code, preserve_missing_labels=True).astype(np.uint8)
    edges = [np.linspace(0, s, g + 1).astype(int) for s, g in zip(arr.shape, grid)]

    # skeleton main paths -> length per class per block
    with open(os.path.join(FILT, f"{vol}_{suffix}_mip{mip}", "morphometrics",
                           f"skeletons_mip{mip}.pkl"), "rb") as f:
        skels = pickle.load(f)
    bounds = np.array(arr.shape) * res
    length = {c: np.zeros(grid) for c in ("axon", "dendrite")}
    for l, c in zip(d_morph.label, d_morph.cls):
        if c not in length or int(l) not in skels:
            continue
        s = skels[int(l)]
        v = ng.kimimaro_to_nm(s.vertices, res)
        path = ng.main_path(v, s.edges, bounds_nm=bounds, radius_nm=np.asarray(s.radius, float))
        _, xyz, _ = ng.resample_path(v[path], 80.0, 250.0)
        if not len(xyz):
            continue
        idx = [np.clip(np.searchsorted(e[1:-1] * r, xyz[:, k], side="right"), 0, g - 1)
               for k, (e, r, g) in enumerate(zip(edges, res, grid))]
        np.add.at(length[c], tuple(idx), 0.080)          # um per sample

    rows = []
    for i, j, k in itertools.product(*(range(g) for g in grid)):
        blk = arr[edges[0][i]:edges[0][i + 1], edges[1][j]:edges[1][j + 1], edges[2][k]:edges[2][k + 1]]
        n = np.bincount(blk.ravel(), minlength=6)[:6] / blk.size
        vol_um3 = blk.size * np.prod(res) * 1e-9
        row = {"vol": vol, "block": f"{i}_{j}_{k}"}
        row.update({f"frac_{c}": n[code_] for c, code_ in CODES.items()})
        row["frac_unlabeled"] = n[0]
        row["axon_length_density"] = length["axon"][i, j, k] / vol_um3
        row["dendrite_length_density"] = length["dendrite"][i, j, k] / vol_um3
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vols", nargs="+", default=["fib_b", "fib_c", "fib_f"])
    ap.add_argument("--suffix", default="neuron_instances_graph_v1")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--min-profile-um", type=float, default=2.0)
    ap.add_argument("--grid", type=int, nargs=3, default=[6, 4, 1])
    ap.add_argument("--out", default=os.path.join(FILT, "compare_outliers_graph_v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vols = args.vols

    morph, segs = {}, {}
    for v in vols:
        base = os.path.join(FILT, f"{v}_{args.suffix}_mip{args.mip}")
        d = pd.read_csv(os.path.join(base, "morphometrics", "neurites.csv"))
        if "surface_excess" not in d:
            raise SystemExit(f"{v}: run analysis/neurite_classify.py first")
        d["vol"] = v
        d["spines_per_um"] = d.protrusions_per_um + d.split_neighbors_per_um
        morph[v] = d
        segs[v] = pd.read_csv(os.path.join(base, "segments.csv"))
    df = pd.concat(morph.values(), ignore_index=True)
    prof = df[df.profile_length_um >= args.min_profile_um]

    # ---- 1. univariate tests (per-neurite + per-sub-block), one Bonferroni family ----
    blocks = pd.concat([tissue_subblocks(v, args.suffix, args.mip, morph[v], segs[v], args.grid)
                        for v in vols], ignore_index=True)
    blocks.to_csv(os.path.join(args.out, "subblocks.csv"), index=False)
    tests = []
    for cls in ("axon", "dendrite"):
        for f in FEATURES:
            tests += pairwise(prof[prof.cls == cls], cls, f, vols)
    tissue_feats = ["frac_axon", "frac_dendrite", "frac_glia", "frac_short", "frac_other",
                    "axon_length_density", "dendrite_length_density"]
    for f in tissue_feats:
        tests += pairwise(blocks, "tissue", f, vols)
    tests = pd.DataFrame(tests)
    tests["p_bonf"] = np.minimum(tests.p * len(tests), 1.0)
    odd = odd_one_out(tests, vols, None)
    odd.to_csv(os.path.join(args.out, "odd_one_out.csv"), index=False)
    tests.to_csv(os.path.join(args.out, "pairwise_tests.csv"), index=False)

    # ---- 2. multivariate separability ----
    sep = []
    for cls in ("axon", "dendrite"):
        sub = prof[prof.cls == cls].reset_index(drop=True)
        for tag, feats in (("all features", MV_FEATURES),
                           ("no split-off counts", [f for f in MV_FEATURES if f not in SPLIT_CONFOUNDED])):
            for r in separability(sub, vols, feats):
                r.update({"class": cls, "features": tag})
                sep.append(r)
    sep = pd.DataFrame(sep)
    sep.to_csv(os.path.join(args.out, "separability.csv"), index=False)

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    piv = odd.pivot_table(index=["group", "feature"], columns="vol", values="mean_delta_vs_others")
    flag = odd.pivot_table(index=["group", "feature"], columns="vol", values="outlier", aggfunc="max")
    piv, flag = piv[vols], flag[vols]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(piv) + 1.8))
    lim = max(0.3, np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, ("● " if flag.values[i, j] else "") + f"{val:+.2f}", ha="center",
                        va="center", fontsize=7.5, color="white" if abs(val) > 0.6 * lim else "black")
    ax.set_xticks(range(len(vols)), [NAMES.get(v, v) for v in vols], fontsize=8)
    ax.set_yticks(range(len(piv)), [f"{g}: {f}" for g, f in piv.index], fontsize=7.5)
    ax.set_title("How each volume differs from the other two (mean Cliff's δ)\n"
                 "● = odd one out: differs from both, same direction, Bonferroni p < 0.05", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.5, label="δ vs other two (+ = higher)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "odd_one_out.png"), dpi=130)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, cls in zip(axs, ("axon", "dendrite")):
        s = sep[sep["class"] == cls]
        pairs = [f"{a[-1]} vs {b[-1]}" for a, b in itertools.combinations(vols, 2)]
        for k, (tag, c) in enumerate((("all features", "0.3"), ("no split-off counts", "#2a78d6"))):
            t = s[s.features == tag]
            vals = [t[(t.a == a) & (t.b == b)].auc.iloc[0] if len(t[(t.a == a) & (t.b == b)]) else np.nan
                    for a, b in itertools.combinations(vols, 2)]
            ax.bar(np.arange(len(pairs)) + (k - 0.5) * 0.38, vals, 0.36, color=c, label=tag)
        ax.axhline(0.5, color="0.5", ls=":", lw=1)
        ax.set_ylim(0.4, 1.0); ax.set_xticks(range(len(pairs)), pairs)
        ax.set_ylabel("cross-validated AUC"); ax.set_title(f"{cls}s: how separable are the volumes?")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "separability.png"), dpi=130)

    fig, axs = plt.subplots(1, len(tissue_feats), figsize=(2.6 * len(tissue_feats), 3.8))
    rng = np.random.default_rng(0)
    for ax, f in zip(axs, tissue_feats):
        data = [blocks.loc[blocks.vol == v, f] * (100 if f.startswith("frac") else 1) for v in vols]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, v in zip(bp["boxes"], vols):
            patch.set_facecolor(COLORS[v]); patch.set_alpha(0.3)
        for i, (v, y) in enumerate(zip(vols, data), 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=9, color=COLORS[v], zorder=3)
        ax.set_xticks(range(1, len(vols) + 1), [v[-1] for v in vols])
        ax.set_title(f.replace("frac_", "% ").replace("_length_density", " length\n(um / um^3)"), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tissue.png"), dpi=130)

    pd.set_option("display.width", 220)
    print("ODD ONE OUT:")
    print(odd[odd.outlier].sort_values(["vol", "group", "feature"]).round(3).to_string(index=False))
    print("\nSEPARABILITY:")
    print(sep.round(3).to_string(index=False))
    print("\nTISSUE (sub-block medians):")
    print(blocks.groupby("vol")[tissue_feats].median().round(4).T)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
