#!/usr/bin/env python
"""Is a volume's flattening biological or a preparation artifact (compression
along one axis)? Two independent tests per volume.

neurites  Cross-sections perpendicular to each axon / dendrite shaft (every
          --every-nm along the skeleton main path, 8 nm labels). For each section:
          ellipticity and the 3D directions of its long and short axes.
            * minor_wins_<k>: fraction of flattened sections whose SHORT axis is
              closer to global axis k than their long axis is. Both axes lie in
              the plane perpendicular to the neurite, so without a preferred
              direction this is 50% for every k; compression along k pushes it up.
            * ellipticity vs |tangent . e_k|: compression along k cannot flatten a
              neurite that runs along k, so under compression sections of neurites
              parallel to k stay round and perpendicular ones flatten.
mito      Each mitochondrion's shape tensor from voxel second moments (one pass,
          16 nm). Flatness = sqrt(l3/l2); shortest-axis direction e3. Under
          compression along k, mean |e3_k| rises above 0.5 (its isotropic value)
          and mitochondria are flatter overall.

Outputs: analysis_outputs/compression_check/{sections,mito}_<vol>.csv, summary.csv, *.png
"""
import argparse, os, pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume

from magneton.analysis.lib import neurite_geometry as ng

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"
OUT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/compression_check"
COLORS = {"fib_b": "#2a78d6", "fib_c": "#eb6834", "fib_f": "#1baf7a"}
AX = "xyz"

_LABELS = None
_CFG = None


def _sections(job):
    label, cls, verts, edges, radius = job
    c = _CFG
    v = ng.kimimaro_to_nm(verts, c["skel_res"])
    path = ng.main_path(v, edges, bounds_nm=c["bounds_nm"], radius_nm=radius)
    s, xyz, tan = ng.resample_path(v[path], c["every_nm"], 250.0)
    rows = []
    for i in range(len(s)):
        if s[i] < 200 or s[i] > s[-1] - 200:
            continue
        r_dbf = np.interp(s[i], np.r_[0, np.cumsum(np.linalg.norm(np.diff(v[path], axis=0), axis=1))],
                          radius[path])
        cs = ng.cross_section(_LABELS, c["sample_res"], label, xyz[i], tan[i],
                              float(np.clip(4 * r_dbf + 150, 400, 3000)))
        if not np.isfinite(cs["area_nm2"]) or cs["oob"] or cs["edge"] or cs["major_nm"] <= 0:
            continue
        mj = cs["major_xyz"] / np.linalg.norm(cs["major_xyz"])
        mn = np.cross(tan[i], mj); mn /= np.linalg.norm(mn)
        rows.append([label, cls, cs["minor_nm"] / cs["major_nm"], *tan[i], *mj, *mn])
    return rows


def neurite_sections(vol, suffix, mip, every_nm, workers):
    global _LABELS, _CFG
    base = os.path.join(FILT, f"{vol}_{suffix}_mip{mip}", "morphometrics")
    d = pd.read_csv(os.path.join(base, "neurites.csv"))
    d = d[d.cls.isin(["axon", "dendrite"]) & (d.profile_length_um >= 2)]
    with open(os.path.join(base, f"skeletons_mip{mip}.pkl"), "rb") as f:
        sk = pickle.load(f)
    cv = CloudVolume(f"precomputed://file://{ROOT}/{vol}_{suffix}", mip=0, progress=False,
                     fill_missing=True, parallel=8)
    _LABELS = np.asarray(cv[:, :, :][..., 0])
    res = tuple(float(r) for r in cv.resolution)
    skres = tuple(float(r) for r in CloudVolume(f"precomputed://file://{ROOT}/{vol}_{suffix}",
                                                mip=mip).resolution)
    _CFG = dict(skel_res=skres, sample_res=res, every_nm=every_nm,
                bounds_nm=np.array(_LABELS.shape) * np.array(res))
    jobs = [(int(l), c, sk[int(l)].vertices, sk[int(l)].edges, np.asarray(sk[int(l)].radius, float))
            for l, c in zip(d.label, d.cls) if int(l) in sk]
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(_sections, jobs, chunksize=4):
            rows += r
    _LABELS = None
    cols = ["label", "cls", "ellipticity", "tx", "ty", "tz", "Mx", "My", "Mz", "mx", "my", "mz"]
    out = pd.DataFrame(rows, columns=cols)
    out["vol"] = vol
    return out


def mito_shapes(vol, seg, mip, min_voxels):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{seg}", mip=mip, progress=False,
                     fill_missing=True, parallel=1)
    res = np.array(cv.resolution, dtype=float)
    a = np.asarray(cv[:, :, :][..., 0])
    a, _ = fastremap.renumber(a, preserve_zero=True, in_place=True)
    n = int(a.max()) + 1
    S = {k: np.zeros(n) for k in ("n", "x", "y", "z", "xx", "yy", "zz", "xy", "xz", "yz")}
    for z0 in range(0, a.shape[2], 32):                      # slabs keep memory flat
        sl = a[:, :, z0:z0 + 32]
        lab = sl.ravel(order="F")
        m = lab > 0
        lab = lab[m]
        X, Y, Z = np.meshgrid(np.arange(sl.shape[0]), np.arange(sl.shape[1]),
                              np.arange(z0, z0 + sl.shape[2]), indexing="ij")
        x, y, z = (q.ravel(order="F")[m] * r for q, r in zip((X, Y, Z), res))
        for k, w in (("n", None), ("x", x), ("y", y), ("z", z), ("xx", x * x), ("yy", y * y),
                     ("zz", z * z), ("xy", x * y), ("xz", x * z), ("yz", y * z)):
            S[k] += np.bincount(lab, weights=w, minlength=n)
    rows = []
    keep = np.flatnonzero(S["n"] >= min_voxels)
    for i in keep:
        N = S["n"][i]
        mu = np.array([S["x"][i], S["y"][i], S["z"][i]]) / N
        C = np.array([[S["xx"][i], S["xy"][i], S["xz"][i]],
                      [S["xy"][i], S["yy"][i], S["yz"][i]],
                      [S["xz"][i], S["yz"][i], S["zz"][i]]]) / N - np.outer(mu, mu)
        w, V = np.linalg.eigh(C)                              # ascending
        l3, l2, l1 = np.maximum(w, 1e-9)
        e3 = V[:, 0]
        rows.append([vol, int(i), N * np.prod(res) * 1e-9, np.sqrt(l3 / l2), np.sqrt(l2 / l1),
                     *np.abs(e3)])
    return pd.DataFrame(rows, columns=["vol", "id", "volume_um3", "flatness", "elongation",
                                       "e3x", "e3y", "e3z"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vols", nargs="+", default=["fib_b", "fib_c", "fib_f"])
    ap.add_argument("--suffix", default="neuron_instances_graph_v1")
    ap.add_argument("--mito-suffix", default="mito_instances_lucchi_v1_e10_hf")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--every-nm", type=float, default=320.0)
    ap.add_argument("--mito-min-voxels", type=int, default=100, help="at the 16 nm mip")
    ap.add_argument("--part", choices=["neurites", "mito", "both"], default="both")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for v in args.vols:
        if args.part in ("neurites", "both"):
            p = os.path.join(OUT, f"sections_{v}.csv")
            if not os.path.exists(p):
                neurite_sections(v, args.suffix, args.mip, args.every_nm, args.workers).to_csv(p, index=False)
                print(f"{v}: sections done", flush=True)
        if args.part in ("mito", "both"):
            p = os.path.join(OUT, f"mito_{v}.csv")
            if not os.path.exists(p):
                mito_shapes(v, f"{v}_{args.mito_suffix}", args.mip, args.mito_min_voxels).to_csv(p, index=False)
                print(f"{v}: mito done", flush=True)

    summary = []
    secs = [pd.read_csv(os.path.join(OUT, f"sections_{v}.csv")) for v in args.vols
            if os.path.exists(os.path.join(OUT, f"sections_{v}.csv"))]
    mitos = [pd.read_csv(os.path.join(OUT, f"mito_{v}.csv")) for v in args.vols
             if os.path.exists(os.path.join(OUT, f"mito_{v}.csv"))]
    S = pd.concat(secs, ignore_index=True) if secs else pd.DataFrame()
    M = pd.concat(mitos, ignore_index=True) if mitos else pd.DataFrame()

    for v in args.vols:
        for cls in ("axon", "dendrite"):
            if S.empty:
                break
            s = S[(S.vol == v) & (S.cls == cls)]
            flat = s[s.ellipticity < 0.8]
            row = {"vol": v, "object": cls, "n": len(s), "median_ellipticity": s.ellipticity.median()}
            for k in AX:
                row[f"minor_wins_{k}"] = float((flat[f"m{k}"].abs() > flat[f"M{k}"].abs()).mean())
            summary.append(row)
        if not M.empty:
            m = M[M.vol == v]
            row = {"vol": v, "object": "mito", "n": len(m), "median_flatness": m.flatness.median()}
            for k in AX:
                row[f"mean_abs_e3_{k}"] = float(m[f"e3{k}"].mean())
            summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not S.empty:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        bins = np.linspace(0, 1, 6)
        for r, cls in enumerate(("axon", "dendrite")):
            for j, k in enumerate(AX):
                ax = axs[r, j]
                for v in args.vols:
                    s = S[(S.vol == v) & (S.cls == cls)]
                    b = np.digitize(s[f"t{k}"].abs(), bins[1:-1])
                    med = [s.ellipticity[b == i].median() for i in range(5)]
                    ax.plot((bins[:-1] + bins[1:]) / 2, med, "o-", color=COLORS.get(v, "0.3"), label=v)
                ax.set_title(f"{cls}s: ellipticity vs alignment with {k}")
                ax.set_xlabel(f"|neurite direction · {k}|  (1 = runs along {k})")
                ax.spines[["top", "right"]].set_visible(False)
            axs[r, 0].set_ylabel("median cross-section minor/major")
        axs[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "sections_vs_direction.png"), dpi=120)
    if not M.empty:
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        for v in args.vols:
            m = M[M.vol == v]
            axs[0].hist(m.flatness, bins=40, range=(0, 1), histtype="step", lw=2, density=True,
                        color=COLORS.get(v, "0.3"), label=f"{v} (n={len(m):,})")
            for j, k in enumerate(AX, 1):
                w = 1 - m.flatness                             # weight by how flat it is
                axs[j].hist(m[f"e3{k}"], bins=20, range=(0, 1), weights=w, histtype="step", lw=2,
                            density=True, color=COLORS.get(v, "0.3"))
        axs[0].set_title("mito flatness sqrt(l3/l2)"); axs[0].legend(frameon=False, fontsize=8)
        for j, k in enumerate(AX, 1):
            axs[j].axhline(1, color="0.5", ls=":")
            axs[j].set_title(f"mito shortest axis: |e3 · {k}|\n(flat = isotropic; weighted by flatness)")
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "mito_shortest_axis.png"), dpi=120)

    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
