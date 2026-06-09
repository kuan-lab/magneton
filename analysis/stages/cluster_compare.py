"""
Pairwise cross-volume comparison: bar plots with significance stars.

Reads per-volume `morphometrics.parquet` (+ sibling `bboxes.parquet` to
recover tissue dimensions) and produces:

  <out>/tissue_summary.png            — count / volume fraction / density per volume
  <out>/features_size.png             — V, SA, max diameter, hull SA
  <out>/features_shape.png            — sphericity, PC1/2/3 length
  <out>/features_inertia_symmetry.png — PC1/2/3 inertia, PC1/2/3 symmetry
  <out>/features_cross_section.png    — PC1/2/3 CS area + circum
  <out>/pairwise_tests.csv            — Mann-Whitney U + Cliff's δ per feature × pair

Each panel: one bar per source (median + IQR error bars) with significance
brackets between pairs. Mann-Whitney U is the test (robust, no normality
assumption); stars use Bonferroni correction over (3 pairs × N features).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PALETTE_DEFAULT = {"fib_b": "#1f77b4", "fib_c": "#ff7f0e", "fib_f": "#2ca02c"}

FEATURES_SIZE   = ["volume_nm3", "surface_area_nm2", "max_diameter_nm", "convex_hull_sa_nm2"]
FEATURES_SHAPE  = ["sphericity", "PC1_length_nm", "PC2_length_nm", "PC3_length_nm"]
FEATURES_IS     = ["PC1_inertia", "PC2_inertia", "PC3_inertia",
                   "PC1_symmetry", "PC2_symmetry", "PC3_symmetry"]
FEATURES_CS     = ["PC1_cs_area_nm2", "PC2_cs_area_nm2", "PC3_cs_area_nm2",
                   "PC1_cs_circum_nm", "PC2_cs_circum_nm", "PC3_cs_circum_nm"]

ALL_FEATURES    = FEATURES_SIZE + FEATURES_SHAPE + FEATURES_IS + FEATURES_CS

# Log-scale these for clearer bar comparison (right-skewed)
LOG_Y = set(FEATURES_SIZE + ["PC1_length_nm", "PC2_length_nm", "PC3_length_nm",
                             "PC1_inertia", "PC2_inertia", "PC3_inertia",
                             "PC1_cs_area_nm2", "PC2_cs_area_nm2", "PC3_cs_area_nm2",
                             "PC1_cs_circum_nm", "PC2_cs_circum_nm", "PC3_cs_circum_nm"])


# ----------------------------- IO ---------------------------------------------

def load_pooled(source_paths: dict, voxel_nm=(4.0, 4.0, 4.0)):
    """Returns (pooled DataFrame with 'source' col, tissue_dims_vox_per_source)."""
    frames = []
    tissue = {}
    for name, p in source_paths.items():
        df = pd.read_parquet(p)
        df = df.copy()
        df["source"] = name
        frames.append(df)
        # Recover tissue dims from the sibling bboxes.parquet (max bbox bounds)
        bboxes_path = os.path.join(os.path.dirname(p), "bboxes.parquet")
        if os.path.isfile(bboxes_path):
            b = pd.read_parquet(bboxes_path)
            tissue[name] = (int(b.bbox_x1.max()), int(b.bbox_y1.max()), int(b.bbox_z1.max()))
        else:
            tissue[name] = None
    pooled = pd.concat(frames, ignore_index=True)
    return pooled, tissue


# ----------------------------- stats ------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ ∈ [-1, 1]. ≥0.474 large, ≥0.33 medium, ≥0.147 small effect."""
    a = np.sort(a); b = np.sort(b)
    n_a, n_b = len(a), len(b)
    # Vectorized: for each a, count how many b are below (and equal)
    less = np.searchsorted(b, a, side="left")
    equal_or_less = np.searchsorted(b, a, side="right")
    greater = n_b - equal_or_less
    return float((less.sum() - greater.sum()) / (n_a * n_b))


def pairwise_table(pooled: pd.DataFrame, features, n_tests: int) -> pd.DataFrame:
    sources = sorted(pooled["source"].unique())
    pairs = [(sources[i], sources[j]) for i in range(len(sources)) for j in range(i+1, len(sources))]
    bonf = max(1, n_tests * len(pairs))

    rows = []
    for feat in features:
        for a, b in pairs:
            x = pooled.loc[pooled["source"] == a, feat].dropna().values
            y = pooled.loc[pooled["source"] == b, feat].dropna().values
            if len(x) < 5 or len(y) < 5:
                continue
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            d = cliffs_delta(x, y)
            rows.append({
                "feature": feat, "src_a": a, "src_b": b,
                "n_a": len(x), "n_b": len(y),
                "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                "median_ratio": float(np.median(x) / np.median(y)) if np.median(y) > 0 else np.nan,
                "U": float(u), "p_value": float(p),
                "p_bonf": float(min(1.0, p * bonf)),
                "cliffs_delta": d,
            })
    return pd.DataFrame(rows)


def stars(p_bonf: float) -> str:
    if p_bonf < 1e-4: return "***"
    if p_bonf < 1e-3: return "**"
    if p_bonf < 5e-2: return "*"
    return "ns"


# ----------------------------- panel plotting ---------------------------------

def _ordered_pairs(sources):
    return [(sources[i], sources[j]) for i in range(len(sources)) for j in range(i+1, len(sources))]


def _short(name: str) -> str:
    """Strip fib_ prefix for compact pair annotations."""
    return name.replace("fib_", "")


def _plot_feature_panel(ax, pooled, feature, sources, palette, pair_pvalues, log_y=False):
    # Per-source values; log-transform if requested so the violin shape isn't smushed
    data = []
    medians = []; q1s = []; q3s = []
    for s in sources:
        v = pooled.loc[pooled["source"] == s, feature].dropna().values
        if log_y:
            v = np.log10(np.clip(v, 1e-9, None))
        data.append(v)
        medians.append(float(np.median(v)))
        q1s.append(float(np.percentile(v, 25)))
        q3s.append(float(np.percentile(v, 75)))

    positions = np.arange(len(sources))
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.8)
    for body, s in zip(parts["bodies"], sources):
        body.set_facecolor(palette[s])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
        body.set_linewidth(0.4)

    # Overlay median dot + IQR box (vertical line + caps)
    for pos, m, q1, q3 in zip(positions, medians, q1s, q3s):
        ax.plot([pos, pos], [q1, q3], color="black", lw=2.0)
        ax.plot([pos - 0.08, pos + 0.08], [q1, q1], color="black", lw=1.0)
        ax.plot([pos - 0.08, pos + 0.08], [q3, q3], color="black", lw=1.0)
        ax.scatter([pos], [m], color="white", edgecolor="black",
                   s=24, zorder=10, linewidth=1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(sources, fontsize=8)
    if log_y:
        ax.set_title(f"{feature}  (log10)", fontsize=9)
    else:
        ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)

    # Inline pair-test annotation across the bottom of the panel.
    # Format example: "b-c: *  |  b-f: ns  |  c-f: ***"
    pairs = _ordered_pairs(sources)
    chunks = []
    for a, b in pairs:
        p = pair_pvalues.get((a, b, feature))
        tag = stars(p) if p is not None else "?"
        chunks.append(f"{_short(a)}-{_short(b)}: {tag}")
    annotation = "   ".join(chunks)
    ax.text(0.5, -0.22, annotation, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, family="monospace",
            color="#333")


def plot_feature_grid(pooled, features, sources, palette, pair_pvalues, out_path,
                      n_cols=4, title=None):
    n = len(features)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    if title:
        fig.suptitle(title, fontsize=11, y=1.00)
    axes = np.atleast_2d(axes).ravel()
    for i, feat in enumerate(features):
        _plot_feature_panel(axes[i], pooled, feat, sources, palette, pair_pvalues,
                            log_y=(feat in LOG_Y))
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tissue_summary(pooled, tissue_dims, sources, palette, voxel_nm, out_path,
                        pair_pvalues=None):
    """Three panels: mito count, volume fraction, mito density (per μm³)."""
    voxel_volume_nm3 = float(np.prod(voxel_nm))
    rows = []
    for s in sources:
        sub = pooled[pooled["source"] == s]
        dims = tissue_dims.get(s)
        tissue_nm3 = dims[0] * dims[1] * dims[2] * voxel_volume_nm3 if dims else np.nan
        tissue_um3 = tissue_nm3 / 1e9 if tissue_nm3 else np.nan
        mito_vol_sum_nm3 = float(sub["volume_nm3"].sum())
        rows.append({
            "source": s,
            "count": len(sub),
            "tissue_um3": tissue_um3,
            "volume_fraction": float(mito_vol_sum_nm3 / tissue_nm3) if tissue_nm3 else np.nan,
            "density_per_um3": len(sub) / tissue_um3 if tissue_um3 else np.nan,
        })
    summary = pd.DataFrame(rows).set_index("source")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metrics = [
        ("count",            "Mito count",                 False, 1.0),
        ("volume_fraction",  "Volume fraction (mito/tissue)", False, 100.0),  # display as %
        ("density_per_um3",  "Density (mitos / μm³)",      False, 1.0),
    ]
    colors = [palette[s] for s in sources]
    for ax, (col, title, log_y, scale) in zip(axes, metrics):
        vals = summary.loc[sources, col].values * scale
        bars = ax.bar(np.arange(len(sources)), vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(sources)))
        ax.set_xticklabels(sources, fontsize=9)
        ax.set_title(title + (" (%)" if col == "volume_fraction" else ""), fontsize=10)
        if log_y:
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=8)
        # leave headroom for annotation
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("Cross-volume tissue summary", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


# ----------------------------- main -------------------------------------------

def run_compare(source_paths: dict, out_dir: str, voxel_nm=(4.0, 4.0, 4.0)):
    out_p = Path(out_dir); out_p.mkdir(parents=True, exist_ok=True)
    pooled, tissue_dims = load_pooled(source_paths)
    sources = sorted(pooled["source"].unique())
    palette = {s: PALETTE_DEFAULT.get(s, plt.cm.tab10(i)) for i, s in enumerate(sources)}

    print(f"[compare] {len(pooled)} mitos across {len(sources)} sources: " +
          ", ".join(f"{s} (n={int((pooled.source == s).sum())})" for s in sources))
    for s, dims in tissue_dims.items():
        print(f"[compare] {s} tissue dims (vox): {dims}")

    # Stats — Bonferroni over all (feature, pair) combinations
    table = pairwise_table(pooled, ALL_FEATURES, n_tests=len(ALL_FEATURES))
    table.to_csv(out_p / "pairwise_tests.csv", index=False)
    print(f"[compare] wrote {out_p / 'pairwise_tests.csv'}")

    # Map for plotting lookup: (src_a, src_b, feat) -> bonf p
    pair_pvalues = {(r.src_a, r.src_b, r.feature): r.p_bonf for r in table.itertuples()}

    # Plots
    summary = plot_tissue_summary(pooled, tissue_dims, sources, palette, voxel_nm,
                                  out_p / "tissue_summary.png")
    summary.to_csv(out_p / "tissue_summary.csv")
    print(f"[compare] tissue summary:")
    print(summary.to_string())

    plot_feature_grid(pooled, FEATURES_SIZE,  sources, palette, pair_pvalues,
                      out_p / "features_size.png", n_cols=4,
                      title="Size features (Mann-Whitney U, Bonferroni-corrected)")
    plot_feature_grid(pooled, FEATURES_SHAPE, sources, palette, pair_pvalues,
                      out_p / "features_shape.png", n_cols=4,
                      title="Shape features")
    plot_feature_grid(pooled, FEATURES_IS,    sources, palette, pair_pvalues,
                      out_p / "features_inertia_symmetry.png", n_cols=3,
                      title="Inertia + symmetry along PCs")
    plot_feature_grid(pooled, FEATURES_CS,    sources, palette, pair_pvalues,
                      out_p / "features_cross_section.png", n_cols=3,
                      title="Cross-section through PC plane")
    print(f"[compare] plots -> {out_p}")

    # Quick textual headline: significant feature count per pair
    print("\n[compare] significance summary (Bonferroni-corrected, * = p_bonf < 0.05):")
    for (a, b), grp in table.groupby(["src_a", "src_b"]):
        sig = (grp["p_bonf"] < 0.05).sum()
        large = (grp["cliffs_delta"].abs() >= 0.474).sum()
        med = ((grp["cliffs_delta"].abs() >= 0.33) & (grp["cliffs_delta"].abs() < 0.474)).sum()
        small = ((grp["cliffs_delta"].abs() >= 0.147) & (grp["cliffs_delta"].abs() < 0.33)).sum()
        print(f"  {a} vs {b}: {sig}/{len(grp)} features significant; "
              f"effect sizes — large:{large} medium:{med} small:{small}")
    return out_p


def _parse_sources(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--source expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"--source {name}: file not found: {path}")
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser(description="Pairwise cross-volume morphometric comparison")
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=PATH (repeatable) — per-volume morphometrics.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel-nm", default="4,4,4",
                    help="voxel size in nm, comma-separated (default 4,4,4)")
    args = ap.parse_args()
    voxel_nm = tuple(float(v) for v in args.voxel_nm.split(","))
    run_compare(_parse_sources(args.source), args.out, voxel_nm=voxel_nm)


if __name__ == "__main__":
    main()
