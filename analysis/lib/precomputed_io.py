"""
CloudVolume helpers for the analysis module.

All paths in this module accept either a bare directory path or a `file://` URI.
Voxel coordinates are XYZ throughout (matching the rest of magneton).
parallel=1 on CloudVolume avoids a Python 3.13 multiprocessing spawn-vs-fork
issue we hit when reading mip-2 from fib_c_mito_instances_v3.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from cloudvolume import CloudVolume


@dataclass(frozen=True)
class VolumeSpecs:
    shape_xyz: Tuple[int, int, int]
    chunk_xyz: Tuple[int, int, int]
    voxel_nm:  Tuple[float, float, float]
    downsample_factor_xyz: Tuple[int, int, int]   # vs mip 0


def get_volume_specs(pc_path: str, mip: int = 0) -> VolumeSpecs:
    cv = CloudVolume(pc_path, mip=mip, parallel=1, progress=False, fill_missing=False)
    scales = cv.info["scales"]
    s_mip = scales[mip]
    s0 = scales[0]
    size_xyz  = tuple(int(v) for v in s_mip["size"])
    chunk_xyz = tuple(int(v) for v in s_mip["chunk_sizes"][0])
    voxel_nm  = tuple(float(v) for v in s_mip["resolution"])
    res0      = tuple(float(v) for v in s0["resolution"])
    factor    = tuple(int(round(r / r0)) for r, r0 in zip(voxel_nm, res0))
    return VolumeSpecs(shape_xyz=size_xyz, chunk_xyz=chunk_xyz,
                       voxel_nm=voxel_nm, downsample_factor_xyz=factor)


def read_full(pc_path: str, mip: int = 0) -> np.ndarray:
    """Read entire volume at the given mip. Use only for high-mip volumes."""
    cv = CloudVolume(pc_path, mip=mip, parallel=1, progress=False, fill_missing=False)
    return np.asarray(cv[:, :, :])[:, :, :, 0]


def read_bbox(pc_path: str, bbox_xyz: Tuple[int, int, int, int, int, int],
              mip: int = 0) -> np.ndarray:
    """Read a 3D crop as XYZ ndarray. bbox is (x0,x1,y0,y1,z0,z1) with exclusive uppers."""
    cv = CloudVolume(pc_path, mip=mip, parallel=1, progress=False, fill_missing=False)
    x0, x1, y0, y1, z0, z1 = bbox_xyz
    return np.asarray(cv[x0:x1, y0:y1, z0:z1])[:, :, :, 0]
