"""
Skeleton-based geometry for unbranched neurite segments (axon / dendrite pieces).

All coordinates here are physical nm, (x, y, z) order, voxel CENTERS. kimimaro
returns vertices at voxel-corner indices * resolution, so callers convert with
`kimimaro_to_nm` before using anything else in this module.

Pipeline per segment:
  main_path          longest path through the skeleton (tree diameter)
  resample_path      uniform arc-length samples + smoothed unit tangents
  cross_sections     plane perpendicular to the tangent at each sample, read
                     from the full label volume -> area, ellipse, perimeter
  profile_features   radius profile -> caliber, variability, varicosity peaks
  side_branches      skeleton vertices off the main path -> attached protrusions
  principal_axis     PCA of the resampled main path -> direction + straightness
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, binary_fill_holes
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from skimage import measure


def kimimaro_to_nm(vertices: np.ndarray, res_nm) -> np.ndarray:
    """kimimaro vertex = voxel index * res (a voxel corner); shift to its center."""
    return vertices + np.asarray(res_nm, dtype=float) / 2.0


# ------------------------------------------------------------------ skeleton graph

def _graph(vertices: np.ndarray, edges: np.ndarray):
    n = len(vertices)
    w = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    g = coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]],
                                  np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    return g


def _walk(pred_row, a, b):
    path = [b]
    while path[-1] != a:
        path.append(pred_row[path[-1]])
    return np.array(path[::-1])


def main_path(vertices: np.ndarray, edges: np.ndarray, bounds_nm=None,
              radius_nm: np.ndarray = None, margin_nm: float = 100.0,
              exit_probe_nm: float = 800.0, min_exit_frac: float = 0.5,
              tail_frac: float = 0.4) -> np.ndarray:
    """Vertex indices of the neurite's shaft path in the largest component.

    With `bounds_nm` (volume extent, xyz nm): the path joins two places where the
    skeleton exits the volume, since a through-going neurite enters and leaves
    through faces, and the shortest path between exits follows the shaft.
    Spiny dendrites also exit through spines (10-30 exits seen on fib_c), so with
    `radius_nm` each exit is scored by the largest inscribed radius within
    `exit_probe_nm` of it (the radius AT a face is ~0, the boundary counts as
    edge) and only exits >= `min_exit_frac` of the thickest one are eligible.
    Remaining terminal stretches thinner than `tail_frac` x the shaft median are
    trimmed. Without bounds (or < 2 exits): the tree diameter, which on a spiny
    dendrite can run spine tip to spine tip."""
    g = _graph(vertices, edges)
    ncomp, comp = connected_components(g, directed=False)
    if ncomp > 1:
        tot = np.zeros(ncomp)
        np.add.at(tot, comp[edges[:, 0]],
                  np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1))
        keep = np.flatnonzero(comp == np.argmax(tot))
    else:
        keep = np.arange(len(vertices))

    path = None
    if bounds_nm is not None:
        dist = np.full(len(vertices), np.inf)
        p = vertices[keep]
        dist[keep] = np.minimum(p, np.asarray(bounds_nm, dtype=float) - p).min(axis=1)
        near = np.flatnonzero(dist < margin_nm)
        if len(near) >= 2:
            nc, lab = connected_components(g[near][:, near], directed=False)
            if nc >= 2:
                reps = np.array([near[lab == j][np.argmin(dist[near[lab == j]])]
                                 for j in range(nc)])
                if radius_nm is not None:
                    Dl = dijkstra(g, indices=reps, limit=exit_probe_nm)
                    score = np.array([radius_nm[np.isfinite(row)].max() for row in Dl])
                    ok = score >= min_exit_frac * score.max()
                    if ok.sum() >= 2:
                        reps = reps[ok]
                if len(reps) >= 2:
                    D, P = dijkstra(g, indices=reps, return_predecessors=True)
                    Dr = D[:, reps]
                    Dr[~np.isfinite(Dr)] = -1
                    i, j = np.unravel_index(np.argmax(Dr), Dr.shape)
                    path = _walk(P[i], reps[i], reps[j])

    if path is None:
        d0 = dijkstra(g, indices=keep[0])
        a = keep[np.argmax(np.where(np.isfinite(d0[keep]), d0[keep], -1))]
        da, pred = dijkstra(g, indices=a, return_predecessors=True)
        b = keep[np.argmax(np.where(np.isfinite(da[keep]), da[keep], -1))]
        path = _walk(pred, a, b)

    if radius_nm is not None and len(path) > 10:
        r = radius_nm[path]
        n = len(path)
        shaft = np.median(r[n // 4: 3 * n // 4])
        thick = np.flatnonzero(r >= tail_frac * shaft)
        if len(thick) >= 2:
            # DBF also falls to ~0 within about one shaft radius of a volume face
            # (the boundary counts as edge), so only trim a thin end that runs
            # well past that: a spine neck, not the shaft meeting the face.
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vertices[path], axis=0), axis=1))]
            allow = max(2.0 * shaft, 300.0)
            lo, hi = thick[0], thick[-1]
            if s[lo] > allow:
                path, s, hi = path[lo:], s[lo:] - s[lo], hi - lo
            if s[-1] - s[hi] > allow:
                path = path[:hi + 1]
    return path


def resample_path(points_nm: np.ndarray, step_nm: float, smooth_nm: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform samples along the polyline. Returns (s, xyz, unit tangent).
    Tangents come from a Gaussian-smoothed copy so voxel stair-steps in the
    skeleton don't tilt the cross-section planes."""
    seg = np.linalg.norm(np.diff(points_nm, axis=0), axis=1)
    s_v = np.r_[0.0, np.cumsum(seg)]
    if s_v[-1] < 2 * step_nm:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
    s = np.arange(0.0, s_v[-1], step_nm)
    xyz = np.stack([np.interp(s, s_v, points_nm[:, k]) for k in range(3)], axis=1)
    sm = gaussian_filter1d(xyz, sigma=max(smooth_nm / step_nm, 1e-6), axis=0, mode="nearest")
    tan = np.gradient(sm, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return s, xyz, tan


# ------------------------------------------------------------------ cross-sections

def _plane_basis(t: np.ndarray):
    helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, helper); u /= np.linalg.norm(u)
    v = np.cross(t, u); v /= np.linalg.norm(v)
    return u, v


def cross_section(labels: np.ndarray, res_nm, label: int, p_nm: np.ndarray,
                  t: np.ndarray, half_nm: float) -> Dict[str, float]:
    """Sample the plane through p_nm normal to t on a grid of one voxel pitch.
    Keeps the component at (or nearest) the center. Flags:
      edge  - the component touches the grid border (plane too small / spine)
      oob   - part of the plane left the volume (neurite cut by the boundary)"""
    res = np.asarray(res_nm, dtype=float)
    pitch = float(res.min())
    n = int(np.ceil(half_nm / pitch))
    u, v = _plane_basis(t)
    g = np.arange(-n, n + 1) * pitch
    gu, gv = np.meshgrid(g, g, indexing="ij")
    pts = p_nm[None, None, :] + gu[..., None] * u + gv[..., None] * v
    idx = np.rint((pts - res / 2.0) / res).astype(np.int64)
    shape = np.array(labels.shape)
    inb = ((idx >= 0) & (idx < shape)).all(axis=-1)
    vals = np.zeros(inb.shape, dtype=bool)
    ii = idx[inb]
    vals[inb] = labels[ii[:, 0], ii[:, 1], ii[:, 2]] == label
    if not vals.any():
        return {"area_nm2": np.nan}
    lbl = measure.label(binary_fill_holes(vals), connectivity=2)
    c = lbl[n, n]
    if c == 0:
        fg = np.argwhere(lbl > 0)
        k = np.argmin(((fg - n) ** 2).sum(1))
        if np.hypot(*(fg[k] - n)) * pitch > 150:         # nothing of ours near the center
            return {"area_nm2": np.nan}
        c = lbl[tuple(fg[k])]
    comp = lbl == c
    area = comp.sum() * pitch ** 2
    rp = measure.regionprops(comp.astype(np.uint8))[0]
    major, minor = rp.major_axis_length * pitch, rp.minor_axis_length * pitch
    perim = measure.perimeter(comp) * pitch
    # long axis of the section as a 3D direction (grid rows run along u, cols along v)
    rc = np.argwhere(comp).astype(float)
    if len(rc) >= 3:
        w_, V_ = np.linalg.eigh(np.cov((rc - rc.mean(0)).T))
        major_xyz = V_[0, 1] * u + V_[1, 1] * v
    else:
        major_xyz = np.full(3, np.nan)
    edge = comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
    oob = bool((~inb & binary_dilation_1(comp)).any())
    return {"area_nm2": float(area), "major_nm": float(major), "minor_nm": float(minor),
            "perimeter_nm": float(perim), "edge": bool(edge), "oob": oob,
            "major_xyz": major_xyz}


def binary_dilation_1(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:] |= m[:-1]; out[:-1] |= m[1:]; out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


# ------------------------------------------------------------------ profile features

def profile_features(s_nm: np.ndarray, r_nm: np.ndarray, step_nm: float,
                     baseline_um: float = 1.5, prominence_frac: float = 0.3,
                     min_spacing_um: float = 0.4) -> Dict[str, float]:
    """Radius profile -> caliber + bumpiness. `r_nm` may contain NaN (invalid
    sections); peaks are found on the longest valid run only so gaps can't
    fabricate a peak."""
    out = {}
    ok = np.isfinite(r_nm)
    out["n_sections"] = int(ok.sum())
    if ok.sum() < 10:
        return out
    rv = r_nm[ok]
    out["r_median_nm"] = float(np.median(rv))
    out["r_mean_nm"] = float(np.mean(rv))
    out["r_p10_nm"], out["r_p90_nm"] = map(float, np.percentile(rv, [10, 90]))
    out["r_cv"] = float(np.std(rv) / np.mean(rv))

    # longest contiguous valid run
    runs = np.split(np.arange(len(r_nm)), np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1)
    run = max((r for r in runs if ok[r[0]]), key=len)
    r = r_nm[run]
    L_um = len(run) * step_nm / 1000.0
    out["profile_length_um"] = float(L_um)
    if len(run) < 10:
        return out
    win = max(3, int(round(baseline_um * 1000 / step_nm)) | 1)
    base = median_filter(r, size=win, mode="nearest")
    excess = r / base - 1.0
    out["swollen_frac"] = float((excess > prominence_frac).mean())
    pk, props = find_peaks(r, prominence=prominence_frac * np.median(r),
                           distance=max(1, int(round(min_spacing_um * 1000 / step_nm))))
    out["n_varicosities"] = int(len(pk))
    out["varicosity_per_um"] = float(len(pk) / L_um)
    if len(pk):
        out["varicosity_peak_ratio"] = float(np.median(r[pk] / base[pk]))
    if len(pk) > 1:
        out["varicosity_spacing_um"] = float(np.median(np.diff(pk)) * step_nm / 1000.0)
    # autocorrelation of the detrended profile: first peak = dominant bead period
    x = excess - excess.mean()
    if x.std() > 0 and len(x) >= 20:
        ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
        apk, _ = find_peaks(ac[: len(x) // 2], height=0.1)
        if len(apk):
            out["bead_period_um"] = float(apk[0] * step_nm / 1000.0)
            out["bead_period_strength"] = float(ac[apk[0]])
    return out


# ------------------------------------------------------------------ side branches

def side_branches(vertices: np.ndarray, edges: np.ndarray, path: np.ndarray,
                  radius_nm: np.ndarray, min_protrusion_nm: float = 300.0
                  ) -> Dict[str, float]:
    """Branches hanging off the main path. A branch's protrusion = its length
    from the attachment vertex minus the shaft radius there, so a stub that
    stays inside the shaft doesn't count."""
    on = np.zeros(len(vertices), dtype=bool); on[path] = True
    off_edges = edges[~(on[edges[:, 0]] & on[edges[:, 1]])]
    if not len(off_edges):
        return {"n_protrusions": 0, "protrusion_lengths_nm": []}
    g = _graph(vertices, off_edges)
    lengths = []
    for a in np.unique(off_edges[on[off_edges]].ravel()):
        if not on[a]:
            continue
        d = dijkstra(g, indices=a)
        d[on] = np.inf                                   # stay off the shaft
        reach = np.isfinite(d)
        if reach.sum() == 0:
            continue
        # one attachment vertex can carry several branches; count its farthest tip
        lengths.append(float(d[reach].max()) - float(radius_nm[a]))
    lengths = [l for l in lengths if l >= min_protrusion_nm]
    return {"n_protrusions": len(lengths), "protrusion_lengths_nm": lengths}


# ------------------------------------------------------------------ orientation

def principal_axis(xyz_nm: np.ndarray) -> Dict[str, float]:
    """PCA of uniformly resampled path points (so every um counts equally)."""
    c = xyz_nm - xyz_nm.mean(0)
    w, V = np.linalg.eigh(np.cov(c.T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    v1 = V[:, 0]
    v1 = v1 if v1[np.argmax(np.abs(v1))] > 0 else -v1   # canonical sign
    tot = w.sum() if w.sum() > 0 else 1.0
    return {"axis_x": float(v1[0]), "axis_y": float(v1[1]), "axis_z": float(v1[2]),
            "axis_elevation_deg": float(np.degrees(np.arcsin(min(1.0, abs(v1[2]))))),
            "axis_azimuth_deg": float(np.degrees(np.arctan2(v1[1], v1[0])) % 180.0),
            "straightness": float(w[0] / tot),
            "pca_ratio_12": float(np.sqrt(w[0] / max(w[1], 1e-9))),
            "pca_ratio_23": float(np.sqrt(w[1] / max(w[2], 1e-9)))}
