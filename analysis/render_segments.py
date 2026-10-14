#!/usr/bin/env python
"""Render individual segment meshes to PNG, one figure per object, several views.

One object per figure so a single giant (possibly merged) segment cannot hide the
rest. Views: top (xy), two sides (xz, yz), one oblique; equal µm scale on all axes.
Meshes are read from the precomputed seg's mesh dir via CloudVolume (the same
files Neuroglancer uses); rendering is matplotlib Poly3DCollection with simple
Lambert shading, so it works headless on a compute node.

Selection: sizes come from the relabel stage's per-core counts (counts_*.npz),
summed per segment id; objects are picked by rank / size band.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cloudvolume import CloudVolume

VIEWS = [("top (xy)", 90, -90), ("side (xz)", 0, -90), ("side (yz)", 0, 0), ("oblique", 25, -55)]


def segment_sizes(counts_dir):
    ids, cnt = [], []
    for f in sorted(glob.glob(os.path.join(counts_dir, "counts_*.npz"))):
        d = np.load(f)
        ids.append(d["ids"]); cnt.append(d["counts"])
    ids = np.concatenate(ids); cnt = np.concatenate(cnt).astype(np.int64)
    u, inv = np.unique(ids, return_inverse=True)
    tot = np.bincount(inv, weights=cnt).astype(np.int64)
    keep = u != 0
    return u[keep], tot[keep]


def pick(ids, sizes, n_top, bands, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(sizes)[::-1]
    out = [(int(ids[i]), int(sizes[i]), f"rank {r + 1}") for r, i in enumerate(order[:n_top])]
    for lo, hi, n in bands:
        cand = np.where((sizes >= lo) & (sizes < hi))[0]
        for i in rng.choice(cand, min(n, len(cand)), replace=False):
            out.append((int(ids[i]), int(sizes[i]), f"{lo:.0e}-{hi:.0e} vox"))
    return out


def shade(verts, faces, color):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, 0.5, 0.75]); light /= np.linalg.norm(light)
    lam = 0.35 + 0.65 * np.abs(n @ light)          # two-sided Lambert + ambient
    return np.clip(np.asarray(color)[None, :3] * lam[:, None], 0, 1)


def render(sid, size, label, cv, res_nm, out_dir, max_faces, color):
    m = cv.mesh.get(sid)
    m = m[sid] if isinstance(m, dict) else m
    v = np.asarray(m.vertices, np.float64) / 1000.0          # nm -> µm
    f = np.asarray(m.faces, np.int64)
    if len(f) > max_faces:                                    # keep figures tractable
        # quadric edge-collapse keeps a closed surface; dropping random faces
        # leaves holes (speckled, see-through renders). Needs fast_simplification.
        try:
            import trimesh
            tm = trimesh.Trimesh(v, f, process=False).simplify_quadric_decimation(face_count=max_faces)
            v, f = np.asarray(tm.vertices), np.asarray(tm.faces)
        except Exception as e:
            print(f"[WARN] {sid}: quadric decimation unavailable ({e!r:.60}); random face subsample")
            f = f[np.random.default_rng(0).choice(len(f), max_faces, replace=False)]
    lo, hi = v.min(0), v.max(0)
    ext = hi - lo
    cols = shade(v, f, color)
    fig = plt.figure(figsize=(16, 4.6), dpi=130)
    for k, (name, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, len(VIEWS), k + 1, projection="3d")
        pc = Poly3DCollection(v[f], facecolors=cols, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        c = (lo + hi) / 2; r = ext.max() / 2 * 1.05
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (µm)", fontsize=8); ax.set_ylabel("y (µm)", fontsize=8); ax.set_zlabel("z (µm)", fontsize=8)
        ax.tick_params(labelsize=6)
    vol_um3 = size * np.prod(res_nm) / 1e9
    fig.suptitle(f"segment {sid}  ·  {label}  ·  {size:,} vox ({vol_um3:.1f} µm³)  ·  "
                 f"extent {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} µm", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"seg_{sid}.png")
    fig.savefig(path); plt.close(fig)
    return path, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True, help="precomputed seg URL (with meshes)")
    ap.add_argument("--counts-dir", required=True, help="relabel checkpoint dir with counts_*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--max-faces", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids, sizes = segment_sizes(args.counts_dir)
    print(f"[INFO] {len(ids)} segments")
    chosen = pick(ids, sizes, args.n_top, [(1e6, 1e7, 3), (1e5, 1e6, 3)], args.seed)
    cv = CloudVolume(args.seg, progress=False)
    res = np.asarray(cv.resolution, float)
    palette = plt.get_cmap("tab10")
    for j, (sid, size, label) in enumerate(chosen):
        try:
            path, ext = render(sid, size, label, cv, res, args.out, args.max_faces, palette(j % 10))
            print(f"[OK] {sid} {label} {size:,} vox extent {ext.round(1).tolist()} µm -> {path}", flush=True)
        except Exception as e:
            print(f"[SKIP] {sid} {label}: {e!r}"[:200], flush=True)


if __name__ == "__main__":
    main()
