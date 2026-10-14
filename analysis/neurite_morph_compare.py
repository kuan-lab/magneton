#!/usr/bin/env python
"""Cross-volume comparison of skeleton morphometrics (analysis/neurite_morphometrics.py).

Per class (axon / dendrite), each feature is compared pairwise across volumes
with two-sided Mann-Whitney U + Cliff's delta, Bonferroni over every
(class x feature x pair) test -- the same statistics as cluster_compare.py and
neurite_compare.py.

Outputs (in --out):
  pairwise_tests.csv       one row per class x feature x pair
  features_<class>.png     per-feature box + strip plots
  orientation.png          principal-axis directions (azimuth vs elevation)
"""
import argparse, itertools, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"
COLORS = {"fib_b": "#2a78d6", "fib_c": "#eb6834", "fib_f": "#1baf7a"}
FEATURES = [
    ("volume_um3", "segment volume (um^3)", True),
    ("path_length_um", "shaft length inside volume (um)", True),
    ("vol_per_um", "volume per length (um^3/um)", True),
    ("surface_per_um", "surface per length (um^2/um)", True),
    ("r_median_nm", "median radius (nm)", True),
    ("r_inscribed_nm", "inscribed radius (nm)", True),
    ("ellipticity", "cross-section minor/major", False),
    ("aspect_ratio", "length / diameter", True),
    ("tortuosity", "path / chord", True),
    ("straightness", "PC1 variance share", False),
    ("r_cv", "radius CV along length", False),
    ("varicosity_per_um", "varicosities per um", False),
    ("varicosity_peak_ratio", "varicosity peak / shaft", False),
    ("bead_period_um", "bead period (um)", False),
    ("protrusions_per_um", "attached protrusions per um", False),
    ("split_neighbors_per_um", "split-off neighbours per um", False),
    ("spines_per_um", "attached + split-off per um", False),
]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vols", nargs="+", default=["fib_b", "fib_c", "fib_f"])
    ap.add_argument("--suffix", default="neuron_instances_graph_v1")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--min-profile-um", type=float, default=2.0,
                    help="drop neurites with less valid radius profile than this")
    ap.add_argument("--out", default=os.path.join(FILT, "compare_morph_graph_v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = []
    for v in args.vols:
        d = pd.read_csv(os.path.join(FILT, f"{v}_{args.suffix}_mip{args.mip}",
                                     "morphometrics", "neurites.csv"))
        d["vol"] = v
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # volume-related per-length measures; SA = 2V / r_2V/SA (face-count surface)
    df["vol_per_um"] = df.volume_um3 / df.path_length_um
    df["surface_per_um"] = 2 * df.volume_um3 / (df.r_2v_sa_nm * 1e-3) / df.path_length_um
    df["spines_per_um"] = df.protrusions_per_um + df.split_neighbors_per_um
    n0 = len(df)
    df = df[df.profile_length_um >= args.min_profile_um]
    print(f"{len(df):,}/{n0:,} neurites with >= {args.min_profile_um} um of valid profile")
    print(df.groupby(["vol", "cls"]).size().unstack())

    tests = []
    for cls in ("axon", "dendrite"):
        sub = df[df.cls == cls]
        for f, _, _ in FEATURES:
            for a, b in itertools.combinations(args.vols, 2):
                x = sub.loc[sub.vol == a, f].dropna()
                y = sub.loc[sub.vol == b, f].dropna()
                if len(x) < 5 or len(y) < 5:
                    continue
                tests.append({"class": cls, "feature": f, "pair": f"{a} vs {b}",
                              "n_a": len(x), "n_b": len(y),
                              "median_a": x.median(), "median_b": y.median(),
                              "p": mannwhitneyu(x, y, alternative="two-sided").pvalue,
                              "cliffs_delta": cliffs_delta(x, y)})
    tests = pd.DataFrame(tests)
    tests["p_bonf"] = np.minimum(tests.p * len(tests), 1.0)
    tests["sig"] = pd.cut(tests.p_bonf, [-1, 1e-3, 1e-2, 5e-2, 2],
                          labels=["***", "**", "*", "ns"])
    tests.to_csv(os.path.join(args.out, "pairwise_tests.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    for cls in ("axon", "dendrite"):
        sub = df[df.cls == cls]
        ncol = 5
        nrow = int(np.ceil(len(FEATURES) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
        for ax in axs.ravel()[len(FEATURES):]:
            ax.axis("off")
        for ax, (f, label, logy) in zip(axs.ravel(), FEATURES):
            data = [sub.loc[sub.vol == v, f].dropna() for v in args.vols]
            bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
            for patch, v in zip(bp["boxes"], args.vols):
                patch.set_facecolor(COLORS.get(v, "0.6")); patch.set_alpha(0.3)
            for i, (v, y) in enumerate(zip(args.vols, data), 1):
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=3, alpha=0.35,
                           color=COLORS.get(v, "0.3"), zorder=3)
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(range(1, len(args.vols) + 1), args.vols)
            t = tests[(tests["class"] == cls) & (tests.feature == f)]
            ax.set_title(label + "\n" + "  ".join(f"{p.split(' vs ')[0][-1]}-{p.split(' vs ')[1][-1]}:{s}"
                                                  for p, s in zip(t.pair, t.sig)), fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        names = {"fib_b": "fib_b Basil AU", "fib_c": "fib_c Cappa YA", "fib_f": "fib_f Fennel AI"}
        fig.suptitle(f"{cls}s — median + IQR, Mann-Whitney U, Bonferroni (n = "
                     + ", ".join(f"{names.get(v, v)} {(sub.vol == v).sum()}" for v in args.vols) + ")")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"features_{cls}.png"), dpi=120)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, cls in zip(axs, ("axon", "dendrite")):
        for v in args.vols:
            s = df[(df.cls == cls) & (df.vol == v)]
            ax.scatter(s.axis_azimuth_deg, s.axis_elevation_deg, s=6, alpha=0.5,
                       color=COLORS.get(v, "0.3"), label=f"{v} (n={len(s)})")
        ax.set_xlim(0, 180); ax.set_ylim(0, 90)
        ax.set_xlabel("azimuth in xy (deg, 0 = +x)"); ax.set_title(f"{cls} principal axis")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("elevation from xy plane (deg, 90 = along z)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "orientation.png"), dpi=120)

    pd.set_option("display.width", 220)
    show = tests[tests.sig != "ns"].sort_values(["class", "feature", "pair"])
    print(show.round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
