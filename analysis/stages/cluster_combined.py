"""
Combined cross-volume embedding + per-source comparison.

Reads multiple `morphometrics.parquet` files (one per source volume / animal),
tags each row with a `source` label, z-scores features on the POOLED matrix
(so all volumes share a single scale), runs PCA + UMAP jointly, plus a
classifier two-sample test (RF, 5-fold CV) to quantify whether the per-source
distributions are distinguishable from morphology alone. Outputs:

  <combined_out>/morphometrics_combined_embedded.parquet
  <combined_out>/classifier_two_sample.json    — RF F1 + per-class + top features
  <combined_out>/plots/combined_umap_by_source.png      — scatter colored by source
  <combined_out>/plots/combined_umap_kde_by_source.png  — per-source KDE contours (paper-style)
  <combined_out>/plots/combined_umap_by_volume.png      — scatter colored by mito volume
  <combined_out>/plots/combined_umap_by_pca1.png        — scatter colored by PCA1 score
  <combined_out>/plots/combined_pca_loadings.png
  <combined_out>/plots/combined_explained_variance.png

Inputs are passed on the CLI as `--source NAME=PATH` pairs (repeatable),
where PATH is a per-volume `morphometrics.parquet`. Output dir via `--out`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from magneton.analysis.stages.cluster import (
    FEATURE_COLS,
    LOG_FEATURES,
    plot_explained_variance,
    plot_pca_loadings,
)


def prepare_pooled(df: pd.DataFrame, log_transform: bool = True):
    X = df[FEATURE_COLS].copy()
    if log_transform:
        for col in LOG_FEATURES:
            X[col] = np.log1p(X[col].clip(lower=0))
    Xz = StandardScaler().fit_transform(X.values)
    return Xz, FEATURE_COLS


def plot_umap_by_source(emb: np.ndarray, sources: pd.Series, out_path: Path,
                        palette=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    cats = sorted(sources.unique())
    if palette is None:
        # tab10-style palette; fixed assignment so colors are stable across runs
        default = {"fib_b": "#1f77b4", "fib_c": "#ff7f0e", "fib_f": "#2ca02c"}
        palette = {c: default.get(c, plt.cm.tab10(i)) for i, c in enumerate(cats)}
    for c in cats:
        mask = (sources == c).values
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5,
                   color=palette[c], label=f"{c} (n={int(mask.sum())})", linewidths=0)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Combined UMAP — {len(sources)} mitos across {len(cats)} volumes")
    ax.legend(markerscale=2, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_kde_by_source(emb: np.ndarray, sources: pd.Series, out_path: Path,
                            palette=None, grid_size: int = 120,
                            density_percentiles=(15, 40, 70, 90)):
    """
    Per-source KDE contour overlay on shared UMAP axes (paper-style figure for
    "where does each animal concentrate in morphometric space").

    `density_percentiles` are percentiles of the per-source KDE density evaluated
    at the source's own data points — so contours go from low-density outer
    shell to high-density inner core. Drops sources with fewer than 10 points.
    """
    cats = sorted(sources.unique())
    if palette is None:
        default = {"fib_b": "#1f77b4", "fib_c": "#ff7f0e", "fib_f": "#2ca02c"}
        palette = {c: default.get(c, plt.cm.tab10(i)) for i, c in enumerate(cats)}

    x_min, x_max = emb[:, 0].min(), emb[:, 0].max()
    y_min, y_max = emb[:, 1].min(), emb[:, 1].max()
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, grid_size),
        np.linspace(y_min - pad_y, y_max + pad_y, grid_size),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(8, 7))
    # Light scatter underneath so dense regions are still visible.
    ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.08, color="gray", linewidths=0)

    for c in cats:
        mask = (sources == c).values
        if mask.sum() < 10:
            continue
        kde = gaussian_kde(emb[mask].T)
        z = kde(grid).reshape(xx.shape)
        # Set contour thresholds as percentiles of the KDE evaluated at own points
        own_density = kde(emb[mask].T)
        levels = np.percentile(own_density, list(density_percentiles))
        levels = np.unique(np.sort(levels))   # in case ties
        ax.contour(xx, yy, z, levels=levels,
                   colors=[palette[c]], linewidths=1.3, alpha=0.9)
        # Proxy line for legend
        ax.plot([], [], color=palette[c], lw=1.5,
                label=f"{c} (n={int(mask.sum())})")

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Pooled UMAP — per-source KDE contours (outer 15% → inner 90%)")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def classifier_two_sample_test(Xz: np.ndarray, sources: pd.Series,
                               feature_names, n_splits: int = 5,
                               n_estimators: int = 200, seed: int = 0):
    """
    5-fold CV random-forest classifier predicting source from morphometrics.
    Returns a dict with macro_f1, per-class F1, confusion matrix, top features.

    macro_f1 ≈ chance (1/n_classes) → distributions are indistinguishable from
    morphology (good biological news: morphometry is animal-independent).
    macro_f1 ≫ chance → strong animal signal (could be biology or technical
    differences between specimens).
    """
    y = sources.values
    labels = sorted(set(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                random_state=seed)
    y_pred = cross_val_predict(rf, Xz, y, cv=skf, n_jobs=1)

    macro = float(f1_score(y, y_pred, average="macro"))
    per_class = f1_score(y, y_pred, average=None, labels=labels)
    cm = confusion_matrix(y, y_pred, labels=labels)
    chance = 1.0 / len(labels)

    # Refit on all data to extract feature importances (consistent w/ paper)
    rf.fit(Xz, y)
    imp = list(zip(feature_names, rf.feature_importances_))
    imp.sort(key=lambda x: -x[1])

    return {
        "n_classes":          len(labels),
        "labels":             list(labels),
        "macro_f1":           macro,
        "chance_f1":          float(chance),
        "per_class_f1":       {str(c): float(f) for c, f in zip(labels, per_class)},
        "confusion_matrix":   cm.tolist(),
        "confusion_labels":   list(labels),
        "top_features":       [(name, float(score)) for name, score in imp[:10]],
        "cv_n_splits":        n_splits,
        "rf_n_estimators":    n_estimators,
        "random_seed":        seed,
    }


def plot_umap_by_continuous(emb: np.ndarray, values: np.ndarray, label: str,
                            out_path: Path, log_color: bool = False,
                            cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(8, 7))
    c = np.log10(np.clip(values, a_min=1e-12, a_max=None)) if log_color else values
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=6, alpha=0.6, cmap=cmap, linewidths=0)
    plt.colorbar(sc, ax=ax, label=(f"log10 {label}" if log_color else label))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Combined UMAP — color: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_combined(sources_paths: dict, out_dir: str,
                 log_transform: bool = True,
                 n_pcs: int = 10,
                 umap_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 seed: int = 0) -> str:
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir_p / "plots"; plot_dir.mkdir(exist_ok=True)

    frames = []
    for name, p in sources_paths.items():
        df = pd.read_parquet(p)
        df = df.copy()
        df["source"] = name
        frames.append(df)
        print(f"[cluster_combined] {name}: {len(df)} mitos from {p}")
    pooled = pd.concat(frames, ignore_index=True)
    print(f"[cluster_combined] pooled: {len(pooled)} mitos across {len(sources_paths)} volumes")

    Xz, names = prepare_pooled(pooled, log_transform=log_transform)
    pca = PCA(n_components=min(n_pcs, Xz.shape[1]))
    scores = pca.fit_transform(Xz)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"[cluster_combined] PCA per-PC variance: "
          f"{(pca.explained_variance_ratio_ * 100).round(1).tolist()}%")
    print(f"[cluster_combined] PCA cumulative:      {(cumvar * 100).round(1).tolist()}%")

    print(f"[cluster_combined] running UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})")
    import umap
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                        random_state=seed)
    emb = reducer.fit_transform(Xz)
    print(f"[cluster_combined] UMAP shape: {emb.shape}")

    for i in range(min(3, scores.shape[1])):
        pooled[f"PCA{i+1}"] = scores[:, i]
    pooled["UMAP1"] = emb[:, 0]
    pooled["UMAP2"] = emb[:, 1]

    out_parquet = out_dir_p / "morphometrics_combined_embedded.parquet"
    pooled.to_parquet(out_parquet, index=False)
    print(f"[cluster_combined] wrote {out_parquet}")

    plot_explained_variance(pca, plot_dir / "combined_explained_variance.png")
    plot_pca_loadings(pca, names, plot_dir / "combined_pca_loadings.png", n_pcs=3)
    plot_umap_by_source(emb, pooled["source"], plot_dir / "combined_umap_by_source.png")
    plot_umap_kde_by_source(emb, pooled["source"], plot_dir / "combined_umap_kde_by_source.png")
    plot_umap_by_continuous(emb, pooled["volume_nm3"].values, "volume_nm3",
                            plot_dir / "combined_umap_by_volume.png", log_color=True)
    plot_umap_by_continuous(emb, scores[:, 0], "PCA1 score",
                            plot_dir / "combined_umap_by_pca1.png", cmap="coolwarm")
    print(f"[cluster_combined] plots -> {plot_dir}")

    # Classifier two-sample test: how distinguishable are the source distributions?
    print(f"[cluster_combined] running RF two-sample test ({len(pooled['source'].unique())} classes, "
          f"5-fold CV, 200 trees)...")
    rf_report = classifier_two_sample_test(Xz, pooled["source"], names,
                                           n_splits=5, n_estimators=200, seed=seed)
    rf_path = out_dir_p / "classifier_two_sample.json"
    with open(rf_path, "w") as f:
        json.dump(rf_report, f, indent=2)
    print(f"[cluster_combined] classifier macro F1 = {rf_report['macro_f1']:.3f}  "
          f"(chance = {rf_report['chance_f1']:.3f})")
    print(f"[cluster_combined]   per-class F1: " +
          ", ".join(f"{k}={v:.3f}" for k, v in rf_report["per_class_f1"].items()))
    print(f"[cluster_combined]   top features: " +
          ", ".join(f"{n}({s:.2f})" for n, s in rf_report["top_features"][:5]))
    print(f"[cluster_combined] wrote {rf_path}")
    return str(out_parquet)


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
    ap = argparse.ArgumentParser(description="Combined cross-volume morphometric embedding")
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=PATH (repeatable) — per-volume morphometrics.parquet")
    ap.add_argument("--out", required=True, help="output dir for combined parquet + plots")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sources = _parse_sources(args.source)
    run_combined(sources, args.out,
                 log_transform=not args.no_log,
                 n_pcs=args.n_pcs,
                 umap_neighbors=args.umap_neighbors,
                 umap_min_dist=args.umap_min_dist,
                 seed=args.seed)


if __name__ == "__main__":
    main()
