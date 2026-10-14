#!/usr/bin/env python
"""Independent cross-volume check of bouton and synapse counts and sizes, read
straight from the instance volumes (not from merge metadata, which overcounts:
fib_b boutons 19.3k claimed vs 2,608 real).

Per object type x volume, at --mip (default 1 = 8 nm):
  * count and density per 1000 um^3, for all labels and for labels >= --min-um3
  * size distribution of INTERIOR objects only (bbox not touching the volume
    boundary; clipped objects read small)
  * density per sub-block (--grid, by bbox centre) so density gets a real test
Statistics: Mann-Whitney U + Cliff's delta, Bonferroni over every test run.

Outputs: analysis_outputs/bouton_synapse_compare/{objects.csv, subblocks.csv,
pairwise_tests.csv, summary.csv, sizes.png, density.png}
"""
import argparse, itertools, os, time
import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume
from scipy import ndimage
from scipy.stats import mannwhitneyu

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
OUT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/bouton_synapse_compare"
COLORS = {"fib_b": "#2a78d6", "fib_c": "#eb6834", "fib_f": "#1baf7a"}
NAMES = {"fib_b": "fib_b (Basil, AU)", "fib_c": "fib_c (Cappa, YA)", "fib_f": "fib_f (Fennel, AI)"}
OBJECTS = {"bouton": "bouton_instances_v1", "synapse": "synapse_instances_v3"}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) * len(b) > 4e7:                       # subsample huge pairs; delta is a proportion
        rng = np.random.default_rng(0)
        a = rng.choice(a, min(len(a), 6000), replace=False)
        b = rng.choice(b, min(len(b), 6000), replace=False)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def objects(vol, obj, mip):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{vol}_{OBJECTS[obj]}", mip=mip,
                     progress=False, fill_missing=True, parallel=1)
    res = np.array(cv.resolution, dtype=float)
    a = np.asarray(cv[:, :, :][..., 0])
    shape = np.array(a.shape)
    a, remap = fastremap.renumber(a, preserve_zero=True, in_place=True)
    inv = np.zeros(max(remap.values()) + 1, dtype=np.uint64)
    for old, new in remap.items():
        inv[new] = old
    vox = np.bincount(a.ravel())
    sl = ndimage.find_objects(a)
    rows = []
    for i, s in enumerate(sl, start=1):
        if s is None:
            continue
        lo = np.array([t.start for t in s]); hi = np.array([t.stop for t in s])
        rows.append((int(inv[i]), int(vox[i]), *((lo + hi) / 2 * res / 1000.0),
                     bool((lo == 0).any() or (hi == shape).any())))
    d = pd.DataFrame(rows, columns=["label", "voxels", "cx_um", "cy_um", "cz_um", "touches_boundary"])
    d["volume_um3"] = d.voxels * np.prod(res) * 1e-9
    d["vol"], d["object"] = vol, obj
    tissue_um3 = float(np.prod(shape * res) * 1e-9)
    return d, tissue_um3, shape * res / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vols", nargs="+", default=["fib_b", "fib_c", "fib_f"])
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--min-um3", type=float, nargs=2, default=[0.02, 0.001],
                    metavar=("BOUTON", "SYNAPSE"),
                    help="size floor for the filtered counts (bouton 0.02 um^3 ~ the fib_b "
                         "sizefilt331k cut at 4 nm)")
    ap.add_argument("--grid", type=int, nargs=3, default=[6, 4, 1])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    floor = dict(zip(OBJECTS, args.min_um3))

    frames, summ, blocks = [], [], []
    for obj in OBJECTS:
        for v in args.vols:
            t0 = time.time()
            d, tissue, ext = objects(v, obj, args.mip)
            frames.append(d)
            big = d[d.volume_um3 >= floor[obj]]
            inner = big[~big.touches_boundary]
            summ.append({"object": obj, "vol": v, "tissue_um3": tissue, "n_all": len(d),
                         "n_ge_floor": len(big), "density_all_per_1000um3": len(d) / tissue * 1000,
                         "density_per_1000um3": len(big) / tissue * 1000,
                         "n_interior": len(inner),
                         "median_volume_um3_interior": inner.volume_um3.median(),
                         "volume_fraction": d.volume_um3.sum() / tissue})
            edges = [np.linspace(0, e, g + 1) for e, g in zip(ext, args.grid)]
            idx = [np.clip(np.searchsorted(e[1:-1], big[c], side="right"), 0, g - 1)
                   for e, c, g in zip(edges, ("cx_um", "cy_um", "cz_um"), args.grid)]
            cnt = np.zeros(args.grid)
            np.add.at(cnt, tuple(idx), 1)
            vfrac = np.zeros(args.grid)
            np.add.at(vfrac, tuple(idx), big.volume_um3.values)
            bvol = tissue / np.prod(args.grid)
            for i, j, k in itertools.product(*(range(g) for g in args.grid)):
                blocks.append({"object": obj, "vol": v, "block": f"{i}_{j}_{k}",
                               "density_per_1000um3": cnt[i, j, k] / bvol * 1000,
                               "volume_fraction": vfrac[i, j, k] / bvol})
            print(f"{obj} {v}: {len(d):,} labels, {len(big):,} >= {floor[obj]} um3 "
                  f"({time.time()-t0:.0f}s)", flush=True)
    D = pd.concat(frames, ignore_index=True)
    S = pd.DataFrame(summ)
    B = pd.DataFrame(blocks)
    D.to_csv(os.path.join(OUT, "objects.csv"), index=False)
    B.to_csv(os.path.join(OUT, "subblocks.csv"), index=False)

    tests = []
    for obj in OBJECTS:
        inner = D[(D.object == obj) & (D.volume_um3 >= floor[obj]) & ~D.touches_boundary]
        bo = B[B.object == obj]
        for what, df_, col in (("object volume (interior)", inner, "volume_um3"),
                               ("density per sub-block", bo, "density_per_1000um3"),
                               ("volume fraction per sub-block", bo, "volume_fraction")):
            for a, b in itertools.combinations(args.vols, 2):
                x, y = df_.loc[df_.vol == a, col], df_.loc[df_.vol == b, col]
                tests.append({"object": obj, "measure": what, "pair": f"{a} vs {b}",
                              "median_a": x.median(), "median_b": y.median(),
                              "p": mannwhitneyu(x, y, alternative="two-sided").pvalue,
                              "cliffs_delta": cliffs_delta(x, y)})
    T = pd.DataFrame(tests)
    T["p_bonf"] = np.minimum(T.p * len(T), 1.0)
    T["sig"] = pd.cut(T.p_bonf, [-1, 1e-3, 1e-2, 5e-2, 2], labels=["***", "**", "*", "ns"])
    T.to_csv(os.path.join(OUT, "pairwise_tests.csv"), index=False)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, obj in zip(axs, OBJECTS):
        inner = D[(D.object == obj) & (D.volume_um3 >= floor[obj]) & ~D.touches_boundary]
        bins = np.logspace(np.log10(inner.volume_um3.min()), np.log10(inner.volume_um3.max()), 50)
        for v in args.vols:
            x = inner.loc[inner.vol == v, "volume_um3"]
            ax.hist(x, bins=bins, histtype="step", lw=2, density=True, color=COLORS[v],
                    label=f"{NAMES[v]}: n={len(x):,}, median {x.median():.3g} µm³")
            ax.axvline(x.median(), color=COLORS[v], ls=":", lw=1.2)
        t = T[(T.object == obj) & (T.measure == "object volume (interior)")]
        ax.set_title(f"{obj} volume (interior, ≥ {floor[obj]} µm³)\n"
                     + "   ".join(f"{p.split(' vs ')[0][-1]}–{p.split(' vs ')[1][-1]}: {s}"
                                  for p, s in zip(t.pair, t.sig)), fontsize=10)
        ax.set_xscale("log"); ax.set_xlabel("volume (µm³)"); ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sizes.png"), dpi=130)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    rng = np.random.default_rng(0)
    for ax, (obj, col, lab) in zip(axs, [(o, c, l) for o in OBJECTS for c, l in
                                          (("density_per_1000um3", "per 1000 µm³"),
                                           ("volume_fraction", "volume fraction"))]):
        data = [B.loc[(B.object == obj) & (B.vol == v), col] for v in args.vols]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, v in zip(bp["boxes"], args.vols):
            patch.set_facecolor(COLORS[v]); patch.set_alpha(0.3)
        for i, (v, y) in enumerate(zip(args.vols, data), 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=9, color=COLORS[v], zorder=3)
        meas = "density per sub-block" if col.startswith("density") else "volume fraction per sub-block"
        t = T[(T.object == obj) & (T.measure == meas)]
        ax.set_title(f"{obj} {lab}\n" + "  ".join(f"{p.split(' vs ')[0][-1]}–{p.split(' vs ')[1][-1]}:{s}"
                                                   for p, s in zip(t.pair, t.sig)), fontsize=9)
        ax.set_xticks(range(1, len(args.vols) + 1), [v[-1] for v in args.vols])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "density.png"), dpi=130)

    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False))
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
