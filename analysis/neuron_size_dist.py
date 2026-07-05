#!/usr/bin/env python
"""Compare neuron segment size distributions across the 3 FIB-SEM animals.

Reads each fib_X_neuron_instances_v2 precomputed volume at the 16 nm mip,
computes a true per-label voxel count via bincount, converts to physical
volume (um^3), and writes per-animal size arrays + a summary table + plots.
"""
import os, json, time
import numpy as np
from cloudvolume import CloudVolume

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
OUT = "/gpfs/radev/home/yf354/magneton/analysis/output/neuron_size_dist"
os.makedirs(OUT, exist_ok=True)

ANIMALS = {"fib_b": "Basil", "fib_c": "Cappa", "fib_f": "Fennel"}
MIP_IDX = 1                   # scale index: 0=8nm, 1=16nm, 2=32nm
MIP_KEY = "16_16_16"          # 16 nm isotropic (matches MIP_IDX)
VOX_UM3 = (16e-3) ** 3        # um^3 per 16 nm voxel = 4.096e-6


def voxel_volume_um3(scale_key):
    nm = int(scale_key.split("_")[0])
    return (nm * 1e-3) ** 3


def sizes_for(code):
    path = f"precomputed://file://{ROOT}/{code}_neuron_instances_v2"
    cv = CloudVolume(path, mip=MIP_IDX, progress=False, fill_missing=True)
    t0 = time.time()
    vol = cv[:, :, :][..., 0]            # x,y,z uint32
    counts = np.bincount(vol.reshape(-1))
    nvox_total = vol.size
    del vol
    labels = np.nonzero(counts)[0]
    labels = labels[labels != 0]         # drop background
    sizes = counts[labels].astype(np.int64)
    print(f"  {code}: read+count {time.time()-t0:.1f}s  "
          f"shape voxels={nvox_total:,}  segments={len(sizes):,}")
    return labels.astype(np.int64), sizes


def summarize(name, sizes_vox, vum3):
    s_um3 = sizes_vox * vum3
    pct = lambda q: np.percentile(s_um3, q)
    return {
        "animal": name,
        "n_segments": int(len(sizes_vox)),
        "total_vol_um3": float(s_um3.sum()),
        "min_um3": float(s_um3.min()),
        "p25_um3": float(pct(25)),
        "median_um3": float(pct(50)),
        "mean_um3": float(s_um3.mean()),
        "p75_um3": float(pct(75)),
        "p95_um3": float(pct(95)),
        "p99_um3": float(pct(99)),
        "max_um3": float(s_um3.max()),
        # fraction of total volume held by the single largest segment (mega-merge indicator)
        "top1_frac": float(np.sort(s_um3)[-1] / s_um3.sum()),
        "top10_frac": float(np.sort(s_um3)[-10:].sum() / s_um3.sum()),
    }


def main():
    vum3 = voxel_volume_um3(MIP_KEY)
    print(f"voxel volume @ {MIP_KEY}: {vum3:.3e} um^3")
    all_sizes = {}
    rows = []
    for code, name in ANIMALS.items():
        print(f"[{name}] ({code})")
        labels, sizes = sizes_for(code)
        np.save(os.path.join(OUT, f"{code}_sizes_vox.npy"), sizes)
        np.save(os.path.join(OUT, f"{code}_labels.npy"), labels)
        all_sizes[code] = sizes
        rows.append(summarize(name, sizes, vum3))

    # write summary table
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"mip": MIP_KEY, "vox_um3": vum3, "rows": rows}, f, indent=2)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== SUMMARY (segment volume in um^3) ===")
    print(df.to_string(index=False))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fib_b": "#1f77b4", "fib_c": "#ff7f0e", "fib_f": "#2ca02c"}
    codes = list(ANIMALS.keys())
    names = [ANIMALS[c] for c in codes]

    # side-by-side violin plots of log10(segment volume) for the 3 animals
    data = [np.log10(all_sizes[c] * vum3) for c in codes]
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = np.arange(1, len(codes) + 1)
    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=True, widths=0.8)
    for pc, c in zip(parts["bodies"], codes):
        pc.set_facecolor(colors[c]); pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}\n(n={len(d):,})" for n, d in zip(names, data)])
    ax.set_ylabel("segment volume (um^3, log10 scale)")
    # relabel log10 ticks as actual um^3 values
    ymin = int(np.floor(min(d.min() for d in data)))
    ymax = int(np.ceil(max(d.max() for d in data)))
    ticks = list(range(ymin, ymax + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_title(f"Neuron segment size distribution @ {MIP_KEY} nm")
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "neuron_size_violin.png"), dpi=130)
    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
