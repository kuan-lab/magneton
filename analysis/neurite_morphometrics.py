#!/usr/bin/env python
"""Skeleton-based morphometrics for the neurite candidates found by
analysis/neurite_filter.py (run that first; this reads its segments.csv).

Per neurite (one row in neurites.csv):
  length       path_length_um (main skeleton path), chord_um, tortuosity
  caliber      r_median_nm / r_mean_nm / r_p10 / r_p90 from cross-sections
               perpendicular to the skeleton every --step-nm, sampled at
               --sample-mip (8 nm); r_inscribed_nm = kimimaro's distance-to-
               boundary radius on the main path; ellipticity (minor/major) and
               circularity (4 pi A / P^2) medians
  shape        aspect_ratio = path length / median diameter; principal axis
               (unit vector, elevation from the xy plane, azimuth in xy) and
               straightness (PC1 variance share)
  bumpiness    r_cv, swollen_frac, varicosity peaks (per um, peak/baseline
               ratio, spacing), bead_period_um (autocorrelation), attached
               protrusions per um (skeleton side branches >= 0.3 um beyond the
               shaft), split_neighbors_per_um (small fully-interior segments
               touching the neurite: split-off spine heads / boutons)

Cost on fib_c (1,798 neurites, 8 cores): skeletonization ~19 min at 16 nm
(cached to skeletons_mip<N>.pkl and reused), cross-sections a few minutes.
"""
import argparse, os, pickle, time
import multiprocessing as mp

import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume

from magneton.analysis.lib import neurite_geometry as ng

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"

_LABELS = None          # sample-mip label volume, inherited by forked workers
_CFG = None


def _read(seg, mip, parallel=8):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{seg}", mip=mip,
                     progress=False, fill_missing=True, parallel=parallel)
    return np.asarray(cv[:, :, :][..., 0]), tuple(float(r) for r in cv.resolution)


def skeletonize(vol, res, ids, teasar, parallel):
    import kimimaro
    from magneton.proofreading.lib.skeleton_io import _patch_kimimaro_trace
    _patch_kimimaro_trace()
    # CloudVolume's parallel read sets the start method to spawn; spawned kimimaro
    # workers re-import kimimaro without the dtype patch and die on uint32 += int64.
    import multiprocess
    multiprocess.set_start_method("fork", force=True)
    mp.set_start_method("fork", force=True)
    vol = fastremap.mask_except(vol, list(ids))
    return kimimaro.skeletonize(vol, anisotropy=res, teasar_params=teasar,
                                dust_threshold=0, fix_branching=True, fix_borders=True,
                                progress=False, parallel=parallel,
                                parallel_chunk_size=25)


def split_neighbors(vol, seg_df, ids):
    """{neurite id: # small fully-interior segments touching it} from face
    adjacency. 'Small' = 0.005-0.3 um^3 and touching no volume face."""
    vol, remap = fastremap.renumber(vol, preserve_zero=True, in_place=False)
    inv = np.zeros(max(remap.values()) + 1, dtype=np.uint64)
    for old, new in remap.items():
        inv[new] = old
    N = np.uint64(len(inv))
    keys = []
    for ax in range(3):
        lo = [slice(None)] * 3; lo[ax] = slice(None, -1)
        hi = [slice(None)] * 3; hi[ax] = slice(1, None)
        a, b = vol[tuple(lo)], vol[tuple(hi)]
        m = (a != b) & (a > 0) & (b > 0)
        x, y = a[m].astype(np.uint64), b[m].astype(np.uint64)
        keys.append(np.unique(np.minimum(x, y) * N + np.maximum(x, y)))
    k = np.unique(np.concatenate(keys))
    pa, pb = inv[(k // N).astype(np.int64)], inv[(k % N).astype(np.int64)]
    small = set(seg_df.loc[(~seg_df.neurite) & (seg_df.n_faces == 0)
                           & seg_df.volume_um3.between(0.005, 0.3), "label"].astype(np.uint64))
    idset = set(np.uint64(i) for i in ids)
    count = dict.fromkeys(idset, 0)
    for x, y in zip(pa, pb):
        if x in idset and y in small:
            count[x] += 1
        elif y in idset and x in small:
            count[y] += 1
    return count


def measure_one(args):
    label, sk_vertices, sk_edges, sk_radius = args
    c = _CFG
    try:
        v = ng.kimimaro_to_nm(sk_vertices, c["skel_res"])
        path = ng.main_path(v, sk_edges, bounds_nm=c["bounds_nm"], radius_nm=sk_radius)
        pts = v[path]
        s, xyz, tan = ng.resample_path(pts, c["step_nm"], c["smooth_nm"])
        row = {"label": int(label)}
        if len(s) < 3:
            return row, None
        L = float(s[-1] + c["step_nm"])
        row["path_length_um"] = L / 1000.0
        row["chord_um"] = float(np.linalg.norm(pts[-1] - pts[0]) / 1000.0)
        row["tortuosity"] = row["path_length_um"] / max(row["chord_um"], 1e-6)
        row["r_inscribed_nm"] = float(np.median(sk_radius[path]))
        row.update(ng.principal_axis(xyz))

        # per-sample plane half-width from the DBF radius along the path
        s_v = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        r_dbf = np.interp(s, s_v, sk_radius[path])
        r_eq = np.full(len(s), np.nan)
        ell, circ = [], []
        for i in range(len(s)):
            if s[i] < c["end_trim_nm"] or s[i] > L - c["end_trim_nm"]:
                continue
            half = float(np.clip(4 * r_dbf[i] + 150, 400, 3000))
            cs = ng.cross_section(_LABELS, c["sample_res"], label, xyz[i], tan[i], half)
            if cs.get("edge") and half < 3000:
                cs = ng.cross_section(_LABELS, c["sample_res"], label, xyz[i], tan[i],
                                      min(half * 2, 3000))
            if not np.isfinite(cs["area_nm2"]) or cs["oob"] or cs["edge"]:
                continue
            r_eq[i] = np.sqrt(cs["area_nm2"] / np.pi)
            if cs["major_nm"] > 0:
                ell.append(cs["minor_nm"] / cs["major_nm"])
            if cs["perimeter_nm"] > 0:
                circ.append(4 * np.pi * cs["area_nm2"] / cs["perimeter_nm"] ** 2)
        row.update(ng.profile_features(s, r_eq, c["step_nm"]))
        row["ellipticity"] = float(np.median(ell)) if ell else np.nan
        row["circularity"] = float(np.median(circ)) if circ else np.nan
        if "r_median_nm" in row:
            row["aspect_ratio"] = row["path_length_um"] * 1000.0 / (2 * row["r_median_nm"])
        sb = ng.side_branches(v, sk_edges, path, sk_radius)
        row["n_protrusions"] = sb["n_protrusions"]
        row["protrusions_per_um"] = sb["n_protrusions"] / row["path_length_um"]
        row["protrusion_len_median_um"] = (float(np.median(sb["protrusion_lengths_nm"])) / 1000.0
                                           if sb["protrusion_lengths_nm"] else np.nan)
        return row, (s.astype(np.float32), r_eq.astype(np.float32))
    except Exception as e:                                   # keep the batch alive
        return {"label": int(label), "error": repr(e)}, None


def main():
    global _LABELS, _CFG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", default="fib_c_neuron_instances_graph_v1")
    ap.add_argument("--filter-mip", type=int, default=1, help="mip neurite_filter ran at")
    ap.add_argument("--skel-mip", type=int, default=1)
    ap.add_argument("--sample-mip", type=int, default=0)
    ap.add_argument("--step-nm", type=float, default=80.0)
    ap.add_argument("--smooth-nm", type=float, default=250.0)
    ap.add_argument("--end-trim-nm", type=float, default=200.0)
    ap.add_argument("--teasar-scale", type=float, default=1.2)
    ap.add_argument("--teasar-const", type=float, default=100.0)
    ap.add_argument("--radius-cut", type=float, default=150.0,
                    help="neurite_filter 2V/SA cut used for the class column")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--skel-parallel", type=int, default=4,
                    help="kimimaro workers; each needs ~6-10 GB at 16 nm, so 8 overran a 64 GB job")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N neurites only")
    args = ap.parse_args()

    base = os.path.join(FILT, f"{args.seg}_mip{args.filter_mip}")
    out = os.path.join(base, "morphometrics")
    os.makedirs(out, exist_ok=True)
    seg_df = pd.read_csv(os.path.join(base, "segments.csv"))
    cand = seg_df[seg_df.neurite].copy()
    cand["cls"] = np.where(cand.radius_nm < args.radius_cut, "axon", "dendrite")
    if args.limit:
        cand = cand.groupby("cls", group_keys=False).head(args.limit)
    ids = cand.label.astype(np.uint64).values

    t0 = time.time()
    # parallel=1: CloudVolume's parallel read switches multiprocessing to spawn,
    # and spawned kimimaro workers lose the dtype patch (see skeletonize).
    vol16, skel_res = _read(args.seg, args.skel_mip, parallel=1)
    skel_path = os.path.join(out, f"skeletons_mip{args.skel_mip}.pkl")
    if os.path.exists(skel_path):
        with open(skel_path, "rb") as f:
            skels = pickle.load(f)
        print(f"loaded {len(skels):,} cached skeletons")
    else:
        skels = skeletonize(vol16.copy(), skel_res, ids,
                            dict(scale=args.teasar_scale, const=args.teasar_const),
                            args.skel_parallel)
        with open(skel_path, "wb") as f:
            pickle.dump(skels, f)
        print(f"skeletonized {len(skels):,} ({time.time()-t0:.0f}s)")
    nbr = split_neighbors(vol16, seg_df, ids)
    del vol16
    print(f"adjacency done ({time.time()-t0:.0f}s)")

    _LABELS, sample_res = _read(args.seg, args.sample_mip)
    _CFG = dict(skel_res=skel_res, sample_res=sample_res, step_nm=args.step_nm,
                bounds_nm=np.array(_LABELS.shape) * np.array(sample_res),
                smooth_nm=args.smooth_nm, end_trim_nm=args.end_trim_nm)
    print(f"loaded mip{args.sample_mip} {_LABELS.shape} ({time.time()-t0:.0f}s)")

    jobs = [(int(i), skels[int(i)].vertices, skels[int(i)].edges,
             np.asarray(skels[int(i)].radius, dtype=float))
            for i in ids if int(i) in skels]
    rows, profiles = [], {}
    # CloudVolume switches the default start method to spawn; force fork so the
    # workers inherit _LABELS instead of starting with it unset.
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (row, prof) in enumerate(pool.imap_unordered(measure_one, jobs, chunksize=4)):
            rows.append(row)
            if prof is not None:
                profiles[f"{row['label']}_s"], profiles[f"{row['label']}_r"] = prof
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df = cand[["label", "cls", "volume_um3", "radius_nm", "n_faces"]].rename(
        columns={"radius_nm": "r_2v_sa_nm"}).merge(df, on="label", how="left")
    df["split_neighbors"] = [nbr.get(np.uint64(l), 0) for l in df.label]
    df["split_neighbors_per_um"] = df.split_neighbors / df.path_length_um
    df.to_csv(os.path.join(out, "neurites.csv"), index=False)
    np.savez_compressed(os.path.join(out, "profiles.npz"), **profiles)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"wrote {out}/neurites.csv ({len(df):,} rows, {n_err} errors, "
          f"{df.r_median_nm.notna().sum():,} with a radius profile) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
