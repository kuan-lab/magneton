#!/usr/bin/env python
"""Filter through-going neurites out of a neuron segmentation and estimate
each one's caliber, to separate thin axons from thick dendrites.

One pass over the whole volume at a single mip (default 16 nm):
  * volume      — voxel count per label
  * surface     — voxel faces shared with a different label (incl. background).
                  Faces on the volume boundary are NOT counted: a neurite's
                  cut ends are not real membrane, so SA is lateral-only.
  * faces       — which of the 6 volume faces each label touches
  * radius      — r = 2V / SA (exact for a cylinder: V = pi r^2 L, SA = 2 pi r L),
                  so no length estimate is needed.

Caveat: face-counted SA overestimates true SA on oblique surfaces (x4/pi for
an axis-aligned cylinder, ~x1.5 averaged over orientations), so radius_nm is
biased low by a similar factor. The bias is roughly uniform, so it shifts the
histogram without merging the axon/dendrite peaks. Spines and varicosities
add surface too, so bumpy neurites read slightly thin.

A label is a neurite candidate if it touches >= --min-faces volume faces and
is at least --min-vol-um3.

Usage:
  python analysis/neurite_filter.py                       # fib_c graph_v1, 16 nm
  python analysis/neurite_filter.py --seg fib_c_neuron_instances_v2 --mip 0
"""
import argparse, os, time
import numpy as np
import pandas as pd
import fastremap
from cloudvolume import CloudVolume

ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs"
OUT_ROOT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"
FACE_NAMES = ["x0", "x1", "y0", "y1", "z0", "z1"]


def load(seg, mip):
    cv = CloudVolume(f"precomputed://file://{ROOT}/{seg}", mip=mip,
                     progress=False, fill_missing=True)
    res = np.array(cv.resolution, dtype=float)
    t0 = time.time()
    vol = np.asarray(cv[:, :, :][..., 0])
    print(f"read {seg} mip{mip} shape={vol.shape} res={res.tolist()} "
          f"({time.time()-t0:.1f}s)")
    # Compact IDs so bincount stays small (graph_v1 labels are uint64).
    vol, remap = fastremap.renumber(vol, preserve_zero=True, in_place=True)
    orig_ids = np.zeros(max(remap.values()) + 1, dtype=np.uint64)
    for old, new in remap.items():
        orig_ids[new] = old
    return vol, res, orig_ids


def measure(vol, res):
    n = int(vol.max()) + 1
    voxels = np.bincount(vol.ravel(), minlength=n)

    # Lateral surface: count faces between differing labels along each axis.
    sa_nm2 = np.zeros(n, dtype=np.float64)
    for ax in range(3):
        face_area = np.prod(np.delete(res, ax))
        lo = [slice(None)] * 3; lo[ax] = slice(None, -1)
        hi = [slice(None)] * 3; hi[ax] = slice(1, None)
        a, b = vol[tuple(lo)], vol[tuple(hi)]
        diff = a != b
        sa_nm2 += face_area * np.bincount(a[diff], minlength=n)
        sa_nm2 += face_area * np.bincount(b[diff], minlength=n)
        del diff, a, b

    faces = np.zeros((n, 6), dtype=bool)
    for ax in range(3):
        for side, idx in enumerate((0, -1)):
            sl = [slice(None)] * 3; sl[ax] = idx
            faces[np.unique(vol[tuple(sl)]), 2 * ax + side] = True
    return voxels, sa_nm2, faces


def tag_segment_properties(seg, df, radius_cut=None):
    """Add class tags + radius/volume numbers to the volume's existing
    Neuroglancer segment-property list (written by toolkit/tools/segment_props.py).
    Classes come from df.cls when present (analysis/neurite_classify.py passes
    axon / dendrite / short), else from radius_nm < radius_cut.
    Idempotent: previous classes/numbers from this script are replaced, other
    tags (size, debris) are kept. The original file is backed up once."""
    import json, shutil
    path = os.path.join(ROOT, seg, "segment_properties", "info")
    if not os.path.exists(path):
        print(f"[WARN] {path} missing; run toolkit segment_props first. Not tagging.")
        return
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    with open(path + ".orig") as f:            # always rebuild from the original
        sp = json.load(f)
    ids = sp["inline"]["ids"]
    props = {p["id"]: p for p in sp["inline"]["properties"]}

    neur = df[df.neurite].set_index(df[df.neurite].label.astype(str))
    if "cls" in neur:
        cls = dict(zip(neur.index, neur.cls))
    else:
        cls = dict(zip(neur.index, np.where(neur.radius_nm < radius_cut, "axon", "dendrite")))

    tagp = props["tags"]
    names = [n for n in ("axon", "dendrite", "glia", "short") if n in set(cls.values())]
    tagp["tags"] += names
    tag_idx = {name: tagp["tags"].index(name) for name in names}
    radius, volume = [], []
    for k, sid in enumerate(ids):
        c = cls.get(sid)
        if c is not None:
            tagp["values"][k] = sorted(tagp["values"][k] + [tag_idx[c]])
            r = float(neur.at[sid, "radius_nm"])
            props["label"]["values"][k] += (f" · {c} r={r:.0f}nm" if np.isfinite(r) else f" · {c}")
        radius.append(round(float(neur.at[sid, "radius_nm"]), 1)
                      if c and np.isfinite(neur.at[sid, "radius_nm"]) else 0.0)
        volume.append(round(float(neur.at[sid, "volume_um3"]), 4) if c else 0.0)
    sp["inline"]["properties"] += [
        {"id": "radius_nm", "type": "number", "data_type": "float32", "values": radius},
        {"id": "volume_um3", "type": "number", "data_type": "float32", "values": volume}]

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sp, f)
    os.replace(tmp, path)
    n_tag = sum(1 for s in ids if s in cls)
    print(f"tagged {n_tag:,}/{len(cls):,} neurites in {path} "
          f"({sum(1 for s in ids if cls.get(s) == 'axon'):,} axon, "
          f"{sum(1 for s in ids if cls.get(s) == 'dendrite'):,} dendrite, "
          f"{sum(1 for s in ids if cls.get(s) == 'glia'):,} glia, "
          f"{sum(1 for s in ids if cls.get(s) == 'short'):,} short)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", default="fib_c_neuron_instances_graph_v1",
                    help="precomputed dir name under precomputed_outputs/")
    ap.add_argument("--mip", type=int, default=1, help="scale index (1 = 16 nm)")
    ap.add_argument("--min-faces", type=int, default=2)
    ap.add_argument("--min-vol-um3", type=float, default=0.01)
    ap.add_argument("--radius-cut", type=float, default=150.0,
                    help="radius (nm, face-count) splitting axon (thin) vs dendrite (thick)")
    ap.add_argument("--tag", action="store_true",
                    help="write axon/dendrite tags into the volume's segment_properties")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, f"{args.seg}_mip{args.mip}")
    os.makedirs(out_dir, exist_ok=True)

    vol, res, orig_ids = load(args.seg, args.mip)
    t0 = time.time()
    voxels, sa_nm2, faces = measure(vol, res)
    print(f"measured {len(voxels)-1:,} labels ({time.time()-t0:.1f}s)")
    del vol

    vol_nm3 = voxels * np.prod(res)
    df = pd.DataFrame({
        "label": orig_ids,
        "voxels": voxels,
        "volume_um3": vol_nm3 * 1e-9,
        "surface_um2": sa_nm2 * 1e-6,
        "radius_nm": np.divide(2 * vol_nm3, sa_nm2, out=np.zeros_like(sa_nm2),
                               where=sa_nm2 > 0),
        "n_faces": faces.sum(1),
    })
    for i, name in enumerate(FACE_NAMES):
        df[f"touch_{name}"] = faces[:, i]
    df = df.iloc[1:]                                  # drop background
    df["neurite"] = (df.n_faces >= args.min_faces) & (df.volume_um3 >= args.min_vol_um3)

    df.to_csv(os.path.join(out_dir, "segments.csv"), index=False)
    neur = df[df.neurite]
    print(f"{len(df):,} labels -> {len(neur):,} neurite candidates "
          f"(>= {args.min_faces} faces, >= {args.min_vol_um3} um^3)")
    print(neur.radius_nm.describe(percentiles=[.05, .25, .5, .75, .95]).round(1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.logspace(np.log10(max(neur.radius_nm.min(), 1)),
                       np.log10(neur.radius_nm.max()), 80)
    axs[0].hist(neur.radius_nm, bins=bins, color="0.3")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("radius = 2V/SA (nm, face-count SA)")
    axs[0].set_ylabel("neurite count")
    axs[0].set_title(f"{args.seg}  (n={len(neur):,})")
    axs[1].scatter(neur.radius_nm, neur.volume_um3, s=3, alpha=0.4, color="0.3")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("radius (nm)"); axs[1].set_ylabel("volume (um^3)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radius_hist.png"), dpi=130)

    # Volume distributions: what the neurite filter kept vs dropped, and the
    # kept neurites split at --radius-cut (thin = axon?, thick = dendrite?).
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.logspace(np.log10(df.volume_um3[df.volume_um3 > 0].min()),
                       np.log10(df.volume_um3.max()), 80)
    axs[0].hist(df.volume_um3[~df.neurite], bins=bins, histtype="step", lw=2,
                color="#eb6834", label=f"dropped (n={(~df.neurite).sum():,})")
    axs[0].hist(neur.volume_um3, bins=bins, histtype="step", lw=2,
                color="#2a78d6", label=f"neurite candidates (n={len(neur):,})")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("volume (um^3)"); axs[0].set_ylabel("segment count")
    axs[0].set_title("all segments")
    axs[0].legend(frameon=False)

    thin = neur.radius_nm < args.radius_cut
    bins = np.logspace(np.log10(neur.volume_um3.min()),
                       np.log10(neur.volume_um3.max()), 50)
    axs[1].hist(neur.volume_um3[thin], bins=bins, histtype="step", lw=2,
                color="#2a78d6", label=f"r < {args.radius_cut:g} nm (n={thin.sum():,})")
    axs[1].hist(neur.volume_um3[~thin], bins=bins, histtype="step", lw=2,
                color="#eb6834", label=f"r >= {args.radius_cut:g} nm (n={(~thin).sum():,})")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("volume (um^3)"); axs[1].set_ylabel("neurite count")
    axs[1].set_title("neurite candidates by radius")
    axs[1].legend(frameon=False)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "volume_hist.png"), dpi=130)
    print(f"wrote {out_dir}/segments.csv, radius_hist.png, volume_hist.png")

    if args.tag:
        tag_segment_properties(args.seg, df, args.radius_cut)


if __name__ == "__main__":
    main()
