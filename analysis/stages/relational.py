"""
Relational stage 2 — statistics + plots on top of the cross-volume matches.

Joins the two match tables (from match_stage) with the three per-volume
morphometrics.parquet tables and answers the relational questions:

  - which mitos live inside a bouton (vs orphan), and how their morphology
    differs from the whole mito population;
  - per bouton: how many mitos it contains, their total/mean volume, how many
    synapses it touches;
  - how many boutons contain ≥1 mito / touch ≥1 synapse (and how many do not);
  - how many synapses touch a bouton (and how many do not);
  - correlations between bouton size and its mito / synapse load.

Writes:
  <out>/bouton_relations.parquet   per-bouton relational table + bouton morphometrics
  <out>/summary.json               headline counts + fractions + correlations
  <out>/plots/*.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from magneton.analysis.config import load_config, strip_file_prefix


def _morph_path(vol_cfg: dict) -> str:
    return os.path.join(strip_file_prefix(vol_cfg["analysis_out"]), "morphometrics.parquet")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ------------------------------------------------------------------ plots -----

def _plot_in_vs_all(mito: pd.DataFrame, plot_dir: Path):
    """Overlaid distributions: mitos inside a bouton vs all mitos."""
    in_b = mito[mito["in_bouton"]]
    panels = [
        ("volume_nm3", "volume (nm³)", True),
        ("sphericity", "sphericity", False),
        ("elongation", "elongation (PC1/PC2)", True),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (col, label, logx) in zip(axes, panels):
        all_v = mito[col].replace([np.inf, -np.inf], np.nan).dropna().values
        in_v  = in_b[col].replace([np.inf, -np.inf], np.nan).dropna().values
        if logx:
            all_v = np.log10(np.clip(all_v, 1e-12, None))
            in_v  = np.log10(np.clip(in_v, 1e-12, None))
            label = f"log10 {label}"
        bins = np.histogram_bin_edges(all_v, bins=40) if len(all_v) else 40
        ax.hist(all_v, bins=bins, density=True, alpha=0.45, color="#888", label=f"all mitos (n={len(all_v)})")
        ax.hist(in_v,  bins=bins, density=True, alpha=0.55, color="#4a7", label=f"in bouton (n={len(in_v)})")
        ax.set_xlabel(label); ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("Mito morphology: inside a bouton vs all")
    fig.tight_layout()
    fig.savefig(plot_dir / "mito_in_bouton_vs_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, x, y, xlabel, ylabel, logx=False, logy=False):
    r = _pearson(np.asarray(x, float), np.asarray(y, float))
    ax.scatter(x, y, s=8, alpha=0.4, color="#46c", linewidths=0)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"r = {r:.3f}")
    return r


def _plot_bouton_vs_mito(bdf: pd.DataFrame, plot_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    r1 = _scatter(axes[0], bdf["volume_nm3"], bdf["n_mitos"],
                  "bouton volume (nm³)", "# mitos contained", logx=True)
    has = bdf[bdf["n_mitos"] > 0]
    r2 = _scatter(axes[1], has["volume_nm3"], has["total_mito_volume_nm3"],
                  "bouton volume (nm³)", "total mito volume (nm³)", logx=True, logy=True)
    fig.suptitle("Bouton size vs mito load")
    fig.tight_layout()
    fig.savefig(plot_dir / "bouton_vs_mito.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"r_boutonvol_vs_nmitos": r1, "r_boutonvol_vs_totalmitovol": r2}


def _plot_bouton_vs_synapse(bdf: pd.DataFrame, plot_dir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = _scatter(ax, bdf["volume_nm3"], bdf["n_synapses"],
                 "bouton volume (nm³)", "# synapses touching", logx=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "bouton_vs_synapse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"r_boutonvol_vs_nsynapses": r}


def _plot_fractions(summary: dict, plot_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("boutons", "boutons", summary["boutons_with_mito"], summary["boutons_without_mito"],
         "has mito", "no mito"),
        ("boutons2", "boutons", summary["boutons_with_synapse"], summary["boutons_without_synapse"],
         "has synapse", "no synapse"),
        ("synapses", "synapses", summary["synapses_touching_bouton"], summary["synapses_not_touching"],
         "touch bouton", "no bouton"),
    ]
    for ax, (_, title, a, b, la, lb) in zip(axes, specs):
        ax.bar([la, lb], [a, b], color=["#4a7", "#a44"])
        ax.set_ylabel("count"); ax.set_title(title)
        for i, v in enumerate([a, b]):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Paired-group fractions")
    fig.tight_layout()
    fig.savefig(plot_dir / "paired_fractions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_synapse_overlap(sdf: pd.DataFrame, plot_dir: Path):
    matched = sdf[sdf["best_bouton"] > 0]
    if len(matched) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(np.clip(matched["overlap_voxels"].values, 1, None)), bins=40, color="#46c", alpha=0.7)
    ax.set_xlabel("log10 overlap voxels (synapse∩bouton)")
    ax.set_ylabel("# synapses")
    ax.set_title(f"Synapse→bouton contact size (n={len(matched)})")
    fig.tight_layout()
    fig.savefig(plot_dir / "synapse_overlap_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main ------

def relational(cfg: dict) -> str:
    out_dir = strip_file_prefix(cfg["output"])
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    vols = cfg["volumes"]
    mito_m   = pd.read_parquet(_morph_path(vols["mito"]))
    bouton_m = pd.read_parquet(_morph_path(vols["bouton"]))
    syn_m    = pd.read_parquet(_morph_path(vols["synapse"]))
    mb = pd.read_parquet(os.path.join(out_dir, "match_mito_to_bouton.parquet"))
    sb = pd.read_parquet(os.path.join(out_dir, "match_synapse_to_bouton.parquet"))

    # --- mito table: morphometrics + parent bouton ---
    mito = mito_m.merge(mb, left_on="seg_id", right_on="mito_seg_id", how="left")
    mito["parent_bouton"] = mito["parent_bouton"].fillna(0).astype(np.int64)
    mito["in_bouton"] = mito["parent_bouton"] > 0
    mito["elongation"] = mito["PC1_length_nm"] / mito["PC2_length_nm"].replace(0, np.nan)

    # --- per-bouton aggregation of mito + synapse load ---
    mito_in = mito[mito["in_bouton"]]
    g_mito = mito_in.groupby("parent_bouton")["volume_nm3"].agg(
        n_mitos="count", total_mito_volume_nm3="sum", mean_mito_volume_nm3="mean")
    syn_matched = sb[sb["best_bouton"] > 0]
    g_syn = syn_matched.groupby("best_bouton")["synapse_seg_id"].agg(n_synapses="count")

    bdf = bouton_m[["seg_id", "volume_nm3", "sphericity"]].copy()
    bdf = bdf.merge(g_mito, left_on="seg_id", right_index=True, how="left")
    bdf = bdf.merge(g_syn, left_on="seg_id", right_index=True, how="left")
    for c, fill in [("n_mitos", 0), ("total_mito_volume_nm3", 0.0),
                    ("mean_mito_volume_nm3", 0.0), ("n_synapses", 0)]:
        bdf[c] = bdf[c].fillna(fill)
    bdf["n_mitos"] = bdf["n_mitos"].astype(np.int64)
    bdf["n_synapses"] = bdf["n_synapses"].astype(np.int64)
    bdf["has_mito"] = bdf["n_mitos"] > 0
    bdf["has_synapse"] = bdf["n_synapses"] > 0
    bdf.to_parquet(os.path.join(out_dir, "bouton_relations.parquet"), index=False)

    # --- summary counts ---
    n_mito = len(mito); n_mito_in = int(mito["in_bouton"].sum())
    n_syn = len(sb); n_syn_touch = int((sb["best_bouton"] > 0).sum())
    n_bouton = len(bdf)
    summary = {
        "n_mitos": n_mito,
        "mitos_in_bouton": n_mito_in,
        "mitos_orphan": n_mito - n_mito_in,
        "frac_mitos_in_bouton": n_mito_in / max(1, n_mito),
        "n_synapses": n_syn,
        "synapses_touching_bouton": n_syn_touch,
        "synapses_not_touching": n_syn - n_syn_touch,
        "frac_synapses_touching": n_syn_touch / max(1, n_syn),
        "n_boutons": n_bouton,
        "boutons_with_mito": int(bdf["has_mito"].sum()),
        "boutons_without_mito": int((~bdf["has_mito"]).sum()),
        "boutons_with_synapse": int(bdf["has_synapse"].sum()),
        "boutons_without_synapse": int((~bdf["has_synapse"]).sum()),
        "boutons_with_both": int((bdf["has_mito"] & bdf["has_synapse"]).sum()),
        "frac_boutons_with_mito": int(bdf["has_mito"].sum()) / max(1, n_bouton),
        "frac_boutons_with_synapse": int(bdf["has_synapse"].sum()) / max(1, n_bouton),
        "max_mitos_in_one_bouton": int(bdf["n_mitos"].max()) if n_bouton else 0,
        "max_synapses_on_one_bouton": int(bdf["n_synapses"].max()) if n_bouton else 0,
    }

    # --- plots (correlations folded into summary) ---
    _plot_in_vs_all(mito, plot_dir)
    summary.update(_plot_bouton_vs_mito(bdf, plot_dir))
    summary.update(_plot_bouton_vs_synapse(bdf, plot_dir))
    _plot_fractions(summary, plot_dir)
    _plot_synapse_overlap(sb, plot_dir)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # console digest
    print("[relational] ---- summary ----")
    print(f"[relational] mitos:    {n_mito_in}/{n_mito} inside a bouton "
          f"({summary['frac_mitos_in_bouton']*100:.1f}%)")
    print(f"[relational] synapses: {n_syn_touch}/{n_syn} touch a bouton "
          f"({summary['frac_synapses_touching']*100:.1f}%)")
    print(f"[relational] boutons:  {summary['boutons_with_mito']}/{n_bouton} contain a mito, "
          f"{summary['boutons_with_synapse']}/{n_bouton} touch a synapse")
    print(f"[relational] corr bouton-vol vs #mitos: {summary['r_boutonvol_vs_nmitos']:.3f}; "
          f"vs #synapses: {summary['r_boutonvol_vs_nsynapses']:.3f}")
    print(f"[relational] wrote {summary_path} + plots/ + bouton_relations.parquet")
    return summary_path


def main():
    ap = argparse.ArgumentParser(description="relational stage 2 — stats + plots")
    ap.add_argument("--config", required=True, help="path to relational YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    relational(cfg)


if __name__ == "__main__":
    main()
