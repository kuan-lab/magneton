"""
Per-mito feature math.

`compute_all_features(mask_xyz, vox_nm, cfg) -> dict` returns 19 morphometric
features. Designed to be called once per mitochondrion, after the bbox crop has
been masked to that mito's seg_id.

Conventions:
- mask_xyz is a 3D boolean ndarray in (X, Y, Z) order.
- vox_nm is (vx, vy, vz) in nanometers, isotropic for fib_c (4,4,4).
- cfg is a dict with keys: surface_area_method, symmetry_rounding,
  refine_bbox, require_center_object.
- All length/area/volume features are in nm (nm², nm³) except when the SA
  method is sqrt_kernel (paper-faithful, unitless).
"""
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from .plane_sampling import find_cross_section, cart_to_spherical, get_center_object
from .surface_area  import compute_sa, compute_perimeter


# ------------------------------------------------ tight bbox refinement helper

def refine_to_tight_bbox(mask_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop mask to the bounding box of its True voxels.
    Returns (cropped_mask, offset) where offset is the (x0, y0, z0) of the crop
    relative to the input mask.
    """
    coords = np.argwhere(mask_xyz)
    if len(coords) == 0:
        return mask_xyz, np.zeros(3, dtype=int)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    slc = tuple(slice(lo[i], hi[i]) for i in range(3))
    return mask_xyz[slc], lo


# ----------------------------------------------------------- principal components

def compute_pcs(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mu, PC_axes, eigvals) for an (n, 3) voxel coord array.
    PC_axes is a 3×3 matrix whose columns are PC1, PC2, PC3 (largest→smallest eig).
    """
    mu = coords.mean(axis=0)
    centered = coords - mu
    cov = (centered.T @ centered) / len(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip to descending
    PC_axes = eigvecs[:, ::-1]
    eigvals = eigvals[::-1]
    return mu, PC_axes, eigvals


# -------------------------------------------------------------- symmetry helper

def compute_symmetry(proj: np.ndarray, axis_i: int, rounding: str = "nearest") -> float:
    """
    2 - (|reflected ∪ original| / |original|) on integer-quantized PC coords.
    proj: (n, 3) projections onto PC axes.
    Returns a value in [0, 1]: 1 = perfectly symmetric about PC axis_i, 0 = perfectly asymmetric.
    """
    if len(proj) == 0:
        return 0.0
    if rounding == "nearest":
        q = np.round(proj).astype(np.int64)
    elif rounding == "truncate":
        q = proj.astype(np.int64)
    else:
        raise ValueError(f"unknown symmetry_rounding: {rounding!r}")

    flipped = q.copy()
    flipped[:, axis_i] = -flipped[:, axis_i]

    orig_unique  = np.unique(q, axis=0)
    union_unique = np.unique(np.concatenate([q, flipped], axis=0), axis=0)
    if len(orig_unique) == 0:
        return 0.0
    ratio = len(union_unique) / len(orig_unique)
    return float(2.0 - ratio)


# ----------------------------------- main entrypoint -------------------------------

def compute_all_features(mask_xyz: np.ndarray,
                         vox_nm: Tuple[float, float, float],
                         cfg: Dict) -> Dict[str, float]:
    """
    Compute 19 morphometric features. `mask_xyz` is expected to be a tight crop
    around a single mitochondrion (after the bbox refinement step in caller).
    """
    method   = cfg.get("surface_area_method", "marching_cubes")
    rounding = cfg.get("symmetry_rounding", "nearest")

    if cfg.get("require_center_object", False):
        mask_xyz = get_center_object(mask_xyz)

    if cfg.get("refine_bbox", True):
        mask_xyz, _ = refine_to_tight_bbox(mask_xyz)

    coords = np.argwhere(mask_xyz)
    n = len(coords)
    out: Dict[str, float] = {}

    if n == 0:
        # Empty mito (shouldn't happen but stay defensive). Return zeros.
        return {k: 0.0 for k in _FEATURE_NAMES}

    vx, vy, vz = vox_nm

    # ---------- volume & SA ----------
    volume = n * vx * vy * vz
    sa = compute_sa(mask_xyz, vox_nm, method=method)
    out["volume_nm3"]       = float(volume)
    out["surface_area_nm2"] = float(sa)
    out["sphericity"]       = float((np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / sa) if sa > 0 else 0.0

    # ---------- PCs ----------
    # coords are in voxel units; scale to nm before covariance to honor anisotropy.
    coords_nm = coords * np.array([vx, vy, vz])
    mu_nm, PC_axes, eigvals = compute_pcs(coords_nm)
    proj = (coords_nm - mu_nm) @ PC_axes        # (n, 3) projections onto PC axes (nm)

    for i in range(3):
        out[f"PC{i+1}_length_nm"] = float(proj[:, i].max() - proj[:, i].min())
        # Moment of inertia along PC_i = sum of squared distances perpendicular to axis i
        perp_sq = np.sum(proj[:, [j for j in range(3) if j != i]] ** 2, axis=1)
        out[f"PC{i+1}_inertia"]   = float(perp_sq.sum())
        out[f"PC{i+1}_symmetry"]  = compute_symmetry(proj, i, rounding=rounding)

    # ---------- convex hull ----------
    if n >= 4:
        try:
            hull = ConvexHull(coords_nm)
            out["convex_hull_sa_nm2"] = float(hull.area)
            verts = coords_nm[hull.vertices]
            out["max_diameter_nm"]    = float(pdist(verts).max()) if len(verts) > 1 else 0.0
        except Exception:
            # degenerate (e.g., all coplanar)
            out["convex_hull_sa_nm2"] = 0.0
            out["max_diameter_nm"]    = 0.0
    else:
        out["convex_hull_sa_nm2"] = 0.0
        out["max_diameter_nm"]    = float(pdist(coords_nm).max()) if n >= 2 else 0.0

    # ---------- cross-section through each PC plane ----------
    # find_cross_section samples a 2D plane normal to PC_i through the mito's
    # centroid (in voxel coords) — NOT the bbox center, which can sit outside
    # elongated/curved mitos and produce empty slices.
    face_areas = [vy * vz, vx * vz, vx * vy]   # for axis 0/1/2 respectively
    centroid_vox = tuple(int(round(c / s)) for c, s in zip(mu_nm, vox_nm))
    for i in range(3):
        pc_dir = PC_axes[:, i]
        _, theta, phi = cart_to_spherical(*pc_dir)
        try:
            slice2d = find_cross_section(theta, phi, mask_xyz, center_xyz=centroid_vox)
            slice_bool = slice2d.astype(bool)
            # The plane is sampled at integer voxel indices in XYZ. Each cell of the
            # 2D sample corresponds to ~1 voxel cross-section. Use the geometric-mean
            # face area as an approximation when the plane is oblique.
            area_per_cell = (vx * vy * vz) ** (2 / 3)
            out[f"PC{i+1}_cs_area_nm2"]  = float(slice_bool.sum() * area_per_cell)
            # Perimeter: use the same method, picking edge length = cube-root of voxel volume
            if method == "sqrt_kernel":
                out[f"PC{i+1}_cs_circum_nm"] = float(compute_perimeter(slice_bool, vox_nm, axes=(0, 1), method="sqrt_kernel"))
            else:
                edge = (vx * vy * vz) ** (1 / 3)
                out[f"PC{i+1}_cs_circum_nm"] = float(
                    compute_perimeter(slice_bool, (edge, edge, edge), axes=(0, 1), method=method)
                )
        except Exception:
            out[f"PC{i+1}_cs_area_nm2"]  = 0.0
            out[f"PC{i+1}_cs_circum_nm"] = 0.0

    return out


# Stable column order for the output parquet.
_FEATURE_NAMES = [
    "volume_nm3",
    "surface_area_nm2",
    "sphericity",
    "convex_hull_sa_nm2",
    "max_diameter_nm",
    "PC1_length_nm", "PC2_length_nm", "PC3_length_nm",
    "PC1_inertia",   "PC2_inertia",   "PC3_inertia",
    "PC1_symmetry",  "PC2_symmetry",  "PC3_symmetry",
    "PC1_cs_area_nm2",  "PC2_cs_area_nm2",  "PC3_cs_area_nm2",
    "PC1_cs_circum_nm", "PC2_cs_circum_nm", "PC3_cs_circum_nm",
]


def feature_names():
    return list(_FEATURE_NAMES)
