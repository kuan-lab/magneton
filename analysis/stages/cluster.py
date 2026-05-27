"""
Stage D — embed & describe morphometric space.

Reads `<output>/morphometrics.parquet`, z-scores the 20 features (with optional
log-transform on right-skewed columns), runs PCA + UMAP, and writes:

  <output>/morphometrics_embedded.parquet   — original rows + PCA1..3 + UMAP1..2
  <output>/plots/explained_variance.png
  <output>/plots/pca_loadings.png           — top-3 PC loadings, sorted by |weight|
  <output>/plots/umap_by_volume.png         — UMAP scatter colored by log10(volume)
  <output>/plots/umap_by_sphericity.png     — UMAP scatter colored by sphericity
  <output>/plots/umap_by_pca1.png           — UMAP scatter colored by PCA1 score

Not unsupervised clustering (yet); this is the paper's visualization step plus a
per-mito PCA that shows the dominant axes of morphological variance in the data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from magneton.analysis.config import load_config, get_stage_config, strip_file_prefix


# The 20 morphometric columns, in the same order as features.feature_names()
FEATURE_COLS = [
    "volume_nm3", "surface_area_nm2", "sphericity",
    "convex_hull_sa_nm2", "max_diameter_nm",
    "PC1_length_nm", "PC2_length_nm", "PC3_length_nm",
    "PC1_inertia",   "PC2_inertia",   "PC3_inertia",
    "PC1_symmetry",  "PC2_symmetry",  "PC3_symmetry",
    "PC1_cs_area_nm2",  "PC2_cs_area_nm2",  "PC3_cs_area_nm2",
    "PC1_cs_circum_nm", "PC2_cs_circum_nm", "PC3_cs_circum_nm",
]

# Right-skewed features benefit from log1p before z-scoring. Sphericity (0..1) and
# the three symmetries (0..1, paper-faithful "nearest" convention) stay linear.
LOG_FEATURES = {
    "volume_nm3", "surface_area_nm2", "convex_hull_sa_nm2", "max_diameter_nm",
    "PC1_length_nm", "PC2_length_nm", "PC3_length_nm",
    "PC1_inertia",   "PC2_inertia",   "PC3_inertia",
    "PC1_cs_area_nm2",  "PC2_cs_area_nm2",  "PC3_cs_area_nm2",
    "PC1_cs_circum_nm", "PC2_cs_circum_nm", "PC3_cs_circum_nm",
}


def prepare_features(df: pd.DataFrame, log_transform: bool = True):
    X = df[FEATURE_COLS].copy()
    if log_transform:
        for col in LOG_FEATURES:
            X[col] = np.log1p(X[col].clip(lower=0))
    Xz = StandardScaler().fit_transform(X.values)
    return Xz, FEATURE_COLS


def run_pca(Xz: np.ndarray, n_components: int = 10):
    n = min(n_components, Xz.shape[1])
    pca = PCA(n_components=n)
    scores = pca.fit_transform(Xz)
    return pca, scores


def run_umap(Xz: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(Xz)


# ----------------------------- plotting ---------------------------------------

def plot_explained_variance(pca: PCA, out_path: Path):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax.bar(x, pca.explained_variance_ratio_, color="#4a7", alpha=0.7, label="per PC")
    ax.plot(x, cumvar, color="#a44", marker="o", label="cumulative")
    ax.set_xlabel("PC"); ax.set_ylabel("explained variance")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_loadings(pca: PCA, feature_names, out_path: Path, n_pcs: int = 3):
    n_pcs = min(n_pcs, pca.components_.shape[0])
    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 7), sharex=False)
    if n_pcs == 1:
        axes = [axes]
    for i in range(n_pcs):
        loadings = pca.components_[i]
        order = np.argsort(np.abs(loadings))[::-1]
        colors = ["#4a7" if v >= 0 else "#a44" for v in loadings[order]]
        axes[i].barh(range(len(loadings)), loadings[order], color=colors)
        axes[i].set_yticks(range(len(loadings)))
        axes[i].set_yticklabels([feature_names[j] for j in order], fontsize=8)
        axes[i].axvline(0, color="k", lw=0.5)
        axes[i].set_title(f"PC{i+1}  ({pca.explained_variance_ratio_[i]*100:.1f}%)")
        axes[i].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap(emb: np.ndarray, color_vals: np.ndarray, color_label: str,
              out_path: Path, log_color: bool = False, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    c = np.log10(np.clip(color_vals, a_min=1e-12, a_max=None)) if log_color else color_vals
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=6, alpha=0.6, cmap=cmap, linewidths=0)
    plt.colorbar(sc, ax=ax, label=(f"log10 {color_label}" if log_color else color_label))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP — color: {color_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- main -------------------------------------------

def cluster(cfg: dict,
            log_transform: bool = True,
            n_pcs: int = 10,
            umap_neighbors: int = 15,
            umap_min_dist: float = 0.1,
            seed: int = 0) -> str:
    paths = get_stage_config(cfg, "paths")
    out_dir = strip_file_prefix(paths["output"])
    in_path = os.path.join(out_dir, "morphometrics.parquet")
    out_path = os.path.join(out_dir, "morphometrics_embedded.parquet")
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    print(f"[analysis.cluster] {len(df)} mitos × {len(FEATURE_COLS)} features")

    Xz, names = prepare_features(df, log_transform=log_transform)
    pca, scores = run_pca(Xz, n_components=n_pcs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"[analysis.cluster] PCA explained variance per PC: "
          f"{(pca.explained_variance_ratio_ * 100).round(1).tolist()}%")
    print(f"[analysis.cluster] PCA cumulative: {(cumvar * 100).round(1).tolist()}%")

    print(f"[analysis.cluster] running UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})")
    emb = run_umap(Xz, n_neighbors=umap_neighbors, min_dist=umap_min_dist, seed=seed)
    print(f"[analysis.cluster] UMAP embedding shape: {emb.shape}")

    # Attach scores and embedding to df
    for i in range(min(3, scores.shape[1])):
        df[f"PCA{i+1}"] = scores[:, i]
    df["UMAP1"] = emb[:, 0]
    df["UMAP2"] = emb[:, 1]
    df.to_parquet(out_path, index=False)
    print(f"[analysis.cluster] wrote {out_path}")

    # Plots
    plot_explained_variance(pca, plot_dir / "explained_variance.png")
    plot_pca_loadings(pca, names, plot_dir / "pca_loadings.png", n_pcs=3)
    plot_umap(emb, df["volume_nm3"].values,    "volume_nm3",  plot_dir / "umap_by_volume.png", log_color=True)
    plot_umap(emb, df["sphericity"].values,    "sphericity",  plot_dir / "umap_by_sphericity.png")
    plot_umap(emb, scores[:, 0],               "PCA1 score",  plot_dir / "umap_by_pca1.png", cmap="coolwarm")
    print(f"[analysis.cluster] plots → {plot_dir}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="analysis stage D — PCA + UMAP on morphometrics")
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-log", action="store_true", help="skip log1p on right-skewed features")
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cluster(cfg,
            log_transform=not args.no_log,
            n_pcs=args.n_pcs,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            seed=args.seed)


if __name__ == "__main__":
    main()
