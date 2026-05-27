"""
Three surface-area implementations, selectable via config.surface_area_method.

All functions take a 3D boolean ndarray `mask` (XYZ ordering doesn't matter
for SA; only the per-axis voxel size does), and a (vx, vy, vz) voxel size in nm.

- face_count:    standard 6-face exposure counting × face area in nm². Monotone
                 with true SA, overestimates a smooth sphere by ~50%.
- marching_cubes: skimage marching_cubes mesh → mesh_surface_area. Closest to
                 true geometric SA. Default.
- sqrt_kernel:   paper-faithful (Sager et al. 2026, voxel_utils.get_foreground_stats).
                 Per inner-border voxel, add sqrt(num_bg_face_neighbors). Unitless.
"""
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_erosion, convolve
from skimage.measure import marching_cubes, mesh_surface_area


def _face_area_nm2(axis: int, vox_nm: Tuple[float, float, float]) -> float:
    """Area of a single voxel face whose normal is `axis` (0=x,1=y,2=z)."""
    vx, vy, vz = vox_nm
    return [vy * vz, vx * vz, vx * vy][axis]


# --------------------------------------------------------------------- face_count

def sa_face_count(mask: np.ndarray, vox_nm: Tuple[float, float, float]) -> float:
    """
    Count voxel faces of foreground that are exposed to background or out-of-bounds.
    Returns SA in nm².
    """
    sa = 0.0
    for axis in (0, 1, 2):
        # shift mask along the axis; pad the trailing slab with False so faces
        # at the volume edge count as exposed (they truly are — `mask` is the
        # whole mito, cropped + isolated).
        m = mask
        shp = list(m.shape)
        shp[axis] = 1
        pad = np.zeros(shp, dtype=bool)
        plus  = np.concatenate([np.take(m, range(1, m.shape[axis]), axis=axis), pad], axis=axis)
        minus = np.concatenate([pad, np.take(m, range(0, m.shape[axis]-1), axis=axis)], axis=axis)
        # foreground voxel with neighbor that's background (or out-of-bounds)
        exposed_plus  = m & (~plus)
        exposed_minus = m & (~minus)
        sa += (exposed_plus.sum() + exposed_minus.sum()) * _face_area_nm2(axis, vox_nm)
    return float(sa)


# ---------------------------------------------------------------- marching_cubes

def sa_marching_cubes(mask: np.ndarray, vox_nm: Tuple[float, float, float]) -> float:
    """
    Marching cubes on the boolean mask. spacing argument scales the mesh to nm.
    Pads with 1 layer of zeros so the level=0.5 transition is always present
    (otherwise a fully-filled bbox would have only 1s and marching_cubes raises).
    Returns SA in nm².
    """
    if mask.sum() < 8:
        # marching_cubes needs enough voxels to triangulate; fall back to face-count
        return sa_face_count(mask, vox_nm)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    verts, faces, _, _ = marching_cubes(padded.astype(np.uint8), level=0.5,
                                        spacing=tuple(vox_nm))
    return float(mesh_surface_area(verts, faces))


# ------------------------------------------------------------------- sqrt_kernel

def _build_face_kernel(ndim: int) -> np.ndarray:
    """3×3×... kernel marking the 2N face neighbors (radius 1, center excluded)."""
    shape = (3,) * ndim
    k = np.zeros(shape, dtype=float)
    center = (1,) * ndim
    for axis in range(ndim):
        idx = list(center)
        idx[axis] = 0
        k[tuple(idx)] = 1
        idx[axis] = 2
        k[tuple(idx)] = 1
    return k


_FACE_KERNEL_3D = _build_face_kernel(3)
_FACE_KERNEL_2D = _build_face_kernel(2)


def sa_sqrt_kernel(mask: np.ndarray, vox_nm: Tuple[float, float, float] = None) -> float:
    """
    Paper-faithful unitless SA: sum over inner-border voxels of
    sqrt(num_bg_face_neighbors).  vox_nm is unused here (paper code is unitless).
    """
    if not mask.any():
        return 0.0
    kernel = _FACE_KERNEL_3D if mask.ndim == 3 else _FACE_KERNEL_2D
    num_bg = convolve((~mask).astype(float), kernel, mode="constant", cval=1.0)
    # inner-border = foreground voxels touching background
    inner_border = mask & (~binary_erosion(mask))
    return float(np.sqrt(num_bg[inner_border]).sum())


# ---------------------------------------------------------------------- dispatch

def compute_sa(mask: np.ndarray, vox_nm: Tuple[float, float, float],
               method: str = "marching_cubes") -> float:
    if method == "face_count":
        return sa_face_count(mask, vox_nm)
    if method == "marching_cubes":
        return sa_marching_cubes(mask, vox_nm)
    if method == "sqrt_kernel":
        return sa_sqrt_kernel(mask, vox_nm)
    raise ValueError(f"unknown surface_area_method: {method!r}")


# ----------------------------------- 2D analog for cross-section perimeter -----

def perimeter_face_count(mask2d: np.ndarray, vox_nm: Tuple[float, float, float],
                         axes: Tuple[int, int]) -> float:
    """2D voxel-edge counting along the two given axes of vox_nm."""
    p = 0.0
    a0, a1 = axes
    edge_lens = [vox_nm[a1], vox_nm[a0]]   # axis 0 of mask2d uses vox_nm[a0]; edge length is along the other axis
    for axis in (0, 1):
        m = mask2d
        shp = list(m.shape)
        shp[axis] = 1
        pad = np.zeros(shp, dtype=bool)
        plus  = np.concatenate([np.take(m, range(1, m.shape[axis]), axis=axis), pad], axis=axis)
        minus = np.concatenate([pad, np.take(m, range(0, m.shape[axis]-1), axis=axis)], axis=axis)
        p += (m & (~plus)).sum() * edge_lens[axis]
        p += (m & (~minus)).sum() * edge_lens[axis]
    return float(p)


def perimeter_sqrt_kernel(mask2d: np.ndarray) -> float:
    """Unitless 2D analog of the paper's sqrt-kernel SA."""
    if not mask2d.any():
        return 0.0
    num_bg = convolve((~mask2d).astype(float), _FACE_KERNEL_2D, mode="constant", cval=1.0)
    inner_border = mask2d & (~binary_erosion(mask2d))
    return float(np.sqrt(num_bg[inner_border]).sum())


def compute_perimeter(mask2d: np.ndarray, vox_nm: Tuple[float, float, float],
                      axes: Tuple[int, int], method: str = "marching_cubes") -> float:
    """
    2D perimeter of a cross-section. `method` picks the same convention as SA:
    - face_count and marching_cubes → voxel-edge length (nm)
    - sqrt_kernel → paper-style unitless
    """
    if method == "sqrt_kernel":
        return perimeter_sqrt_kernel(mask2d)
    # face_count and marching_cubes share the same 2D edge-counting impl
    return perimeter_face_count(mask2d, vox_nm, axes)
