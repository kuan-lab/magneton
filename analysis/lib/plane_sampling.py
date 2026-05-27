"""
Plane sampling primitives ported from the paper repo
(ClarkLabCode/MitochondrialMorphologyPosition/util_files/voxel_utils.py).

Given a 3D boolean mask and a unit-vector axis, build a 2D plane normal to that
axis passing through the volume center, and return the 2D mask sampled at the
plane.

Coordinate convention here mirrors the paper: their `subvol` is (z, y, x) and
their plane-coord output is (y_plane, x_plane, [x, y, z]). The analysis module
uses (x, y, z) ordering for masks, so the helpers accept an XYZ ndarray and
internally transpose to match the paper math.
"""
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import measure


# ---------------------------------------------------------------- spherical math

def cart_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Return (r, theta, phi) where theta is polar (from +z), phi is azimuth (in xy)."""
    r = float(np.sqrt(x * x + y * y + z * z))
    if r == 0.0:
        return (0.0, 0.0, 0.0)
    theta = float(np.arccos(z / r))
    phi   = float(np.arctan2(y, x))
    return (r, theta, phi)


def orthonormal_basis(theta: float, phi: float):
    """
    Return three orthonormal vectors (x_hat, y_hat, z_hat) where x_hat is the
    plane normal (theta, phi) in cartesian, and y_hat / z_hat span the plane.
    Matches the paper's `utils.calc_orthonormal_basis([theta, phi])` output.
    """
    # plane normal in cartesian
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    # pick an arbitrary vector not parallel to n
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return n, u, v


# ---------------------------------------------------- plane coords + cross-section

def calc_plane_coords(theta: float, phi: float, box_shape_xyz: Tuple[int, int, int],
                      height: int = None, width: int = None,
                      center_xyz: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Generate integer XYZ coordinates of a 2D plane through a chosen point.

    box_shape_xyz: shape of the 3D crop in (X, Y, Z) voxels.
    center_xyz: optional center voxel of the plane. Defaults to the geometric
        center of box_shape_xyz; pass the mito centroid for elongated/curved
        mitos whose mass is far from the bbox center.
    Returns a (height, width, 3) int array of XYZ indices into the crop.
    """
    X, Y, Z = box_shape_xyz
    if center_xyz is None:
        center = np.array([X // 2, Y // 2, Z // 2])
    else:
        center = np.asarray(center_xyz, dtype=int)
    n, u, v = orthonormal_basis(theta, phi)
    if height is None:
        height = max(X, Y, Z)
    if width is None:
        width = height

    # height vector spans `u`, width vector spans `v`
    height_vec = u
    width_vec  = v

    hor, ver = np.meshgrid(np.linspace(-height / 2.0, height / 2.0, width),
                           np.linspace(-width / 2.0,  width / 2.0,  height))
    # each point: center + hor * width_vec + ver * height_vec
    plane = (center[None, None, :]
             + hor[:, :, None] * width_vec[None, None, :]
             + ver[:, :, None] * height_vec[None, None, :])
    return plane.astype(int)


def find_cross_section(theta: float, phi: float, subvol_xyz: np.ndarray,
                       height: int = None, width: int = None,
                       center_xyz: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Sample subvol_xyz at coordinates of the plane defined by (theta, phi).
    Boolean masks are hole-filled and the central connected component kept.
    Pass `center_xyz` (mito centroid in voxel coords) for correct plane
    placement on elongated/curved mitos whose mass is far from the bbox center.
    """
    X, Y, Z = subvol_xyz.shape
    plane = calc_plane_coords(theta, phi, (X, Y, Z), height=height, width=width,
                              center_xyz=center_xyz)
    h, w, _ = plane.shape
    flat = plane.reshape(-1, 3)

    valid = ((flat >= 0).all(axis=1)
             & (flat[:, 0] < X) & (flat[:, 1] < Y) & (flat[:, 2] < Z))
    vals = np.zeros(h * w, dtype=subvol_xyz.dtype)
    f_in = flat[valid]
    vals[valid] = subvol_xyz[f_in[:, 0], f_in[:, 1], f_in[:, 2]]
    plane_vals = vals.reshape(h, w)

    # If it's a boolean mask, keep only the central blob and fill holes
    if plane_vals.dtype == bool or set(np.unique(plane_vals).tolist()).issubset({0, 1}):
        return get_center_object(binary_fill_holes(plane_vals.astype(bool)))
    return plane_vals


# ------------------------------------------------- connected-component selector

def get_center_object(arr: np.ndarray, center=None) -> np.ndarray:
    """
    Return only the connected component closest to the array center. The paper's
    insurance against a stray same-ID fragment leaking into the bbox crop.
    """
    if not np.any(arr):
        return np.zeros_like(arr, dtype=bool)
    lbl = measure.label(arr.astype(np.uint8), background=0)
    if center is None:
        center = tuple(np.array(lbl.shape) // 2)
    # find label at center; if 0, use nearest foreground voxel
    if lbl.ndim == 3:
        cz = lbl[center[0], center[1], center[2]]
    else:
        cz = lbl[center[0], center[1]]
    if cz == 0:
        fg = np.array(np.where(lbl > 0))
        if fg.size == 0:
            return np.zeros_like(arr, dtype=bool)
        c = np.array(center).reshape(-1, 1)
        idx = np.argmin(((fg - c) ** 2).sum(axis=0))
        cz = lbl[tuple(fg[:, idx])]
    return lbl == cz
