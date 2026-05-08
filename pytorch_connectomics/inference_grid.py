"""Block grid for direct-precomputed affinity inference.

Tiles a volume with non-overlapping cores. Each block reads `core + 2*halo`
from the input precomputed (clipped at volume edges) and writes only its
core to the output precomputed. Cores are chunk-aligned at interior
boundaries by construction (core_size is a multiple of output_chunk_size
and the grid origin is the precomputed voxel_offset, which CloudVolume
guarantees is chunk-aligned), so adjacent blocks never write to the same
output chunk — no locks needed.
"""
from dataclasses import dataclass
from typing import Tuple, List


BBox = Tuple[int, int, int, int, int, int]  # (z1, z2, y1, y2, x1, x2)


@dataclass(frozen=True)
class InferenceBlock:
    block_id: int
    read_bbox: BBox   # absolute voxel coords ZYX; halo clipped at volume edges
    core_bbox: BBox   # absolute voxel coords ZYX; clipped at volume edges


def _grid_dims(vol_shape_zyx: Tuple[int, int, int], core_size: int) -> Tuple[int, int, int]:
    Z, Y, X = vol_shape_zyx
    return (
        (Z + core_size - 1) // core_size,
        (Y + core_size - 1) // core_size,
        (X + core_size - 1) // core_size,
    )


def block_count(vol_shape_zyx: Tuple[int, int, int], *, core_size: int = 512) -> int:
    nz, ny, nx = _grid_dims(vol_shape_zyx, core_size)
    return nz * ny * nx


def block_by_id(
    block_id: int,
    vol_shape_zyx: Tuple[int, int, int],
    vol_offset_zyx: Tuple[int, int, int] = (0, 0, 0),
    *,
    core_size: int = 512,
    halo: int = 32,
    output_chunk_size: int = 128,
) -> InferenceBlock:
    if core_size % output_chunk_size != 0:
        raise ValueError(
            f"core_size ({core_size}) must be a multiple of "
            f"output_chunk_size ({output_chunk_size}) for lock-free writes"
        )
    nz, ny, nx = _grid_dims(vol_shape_zyx, core_size)
    total = nz * ny * nx
    if not 0 <= block_id < total:
        raise IndexError(f"block_id {block_id} out of range [0, {total})")

    zi = block_id // (ny * nx)
    yi = (block_id // nx) % ny
    xi = block_id % nx

    Z0, Y0, X0 = vol_offset_zyx
    Z, Y, X = vol_shape_zyx

    cz1 = Z0 + zi * core_size
    cy1 = Y0 + yi * core_size
    cx1 = X0 + xi * core_size
    cz2 = min(cz1 + core_size, Z0 + Z)
    cy2 = min(cy1 + core_size, Y0 + Y)
    cx2 = min(cx1 + core_size, X0 + X)

    rz1 = max(cz1 - halo, Z0)
    ry1 = max(cy1 - halo, Y0)
    rx1 = max(cx1 - halo, X0)
    rz2 = min(cz2 + halo, Z0 + Z)
    ry2 = min(cy2 + halo, Y0 + Y)
    rx2 = min(cx2 + halo, X0 + X)

    return InferenceBlock(
        block_id=block_id,
        read_bbox=(rz1, rz2, ry1, ry2, rx1, rx2),
        core_bbox=(cz1, cz2, cy1, cy2, cx1, cx2),
    )


def generate_blocks(
    vol_shape_zyx: Tuple[int, int, int],
    vol_offset_zyx: Tuple[int, int, int] = (0, 0, 0),
    *,
    core_size: int = 512,
    halo: int = 32,
    output_chunk_size: int = 128,
) -> List[InferenceBlock]:
    return [
        block_by_id(bid, vol_shape_zyx, vol_offset_zyx,
                    core_size=core_size, halo=halo,
                    output_chunk_size=output_chunk_size)
        for bid in range(block_count(vol_shape_zyx, core_size=core_size))
    ]


def task_block_range(task_id: int, chunks_per_task: int, total_blocks: int) -> Tuple[int, int]:
    """Block-id range [start, end) handled by SLURM array task `task_id`."""
    start = task_id * chunks_per_task
    end = min(start + chunks_per_task, total_blocks)
    return start, end


def volume_info_from_cv(input_url: str, mip: int = 0):
    """Return (vol_shape_zyx, vol_offset_zyx, resolution_zyx) from a precomputed URL."""
    from cloudvolume import CloudVolume
    cv = CloudVolume(input_url, mip=mip, progress=False)
    sz = cv.info["scales"][mip]["size"]
    off = cv.info["scales"][mip]["voxel_offset"]
    res = cv.info["scales"][mip]["resolution"]
    return (
        (sz[2], sz[1], sz[0]),
        (off[2], off[1], off[0]),
        (res[2], res[1], res[0]),
    )


def is_precomputed_url(path) -> bool:
    if not isinstance(path, str):
        return False
    return path.startswith(("precomputed://", "gs://", "file://"))


def precomputed_url_to_local_path(url: str) -> str:
    """Strip CloudVolume URL prefixes to get a plain filesystem path.

    `precomputed://file:///path` and `file:///path` both map to `/path`.
    `gs://...` URLs have no local path; this returns them unchanged so
    callers can detect non-local URLs.
    """
    if not isinstance(url, str):
        return url
    if url.startswith("precomputed://"):
        url = url[len("precomputed://"):]
    if url.startswith("file://"):
        url = url[len("file://"):]
    return url


def apply_roi(
    vol_shape_zyx: Tuple[int, int, int],
    vol_offset_zyx: Tuple[int, int, int],
    roi: list,
    output_chunk_size: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Override (vol_shape, vol_offset) with an ROI sub-region.

    `roi` is [z1, z2, y1, y2, x1, x2] in absolute voxel coords. Bounds are
    clipped to the actual volume and snapped to chunk boundaries so the
    block grid stays lock-free. Returns the (shape, offset) describing the
    snapped, clipped region.
    """
    if roi is None:
        return vol_shape_zyx, vol_offset_zyx
    if len(roi) != 6:
        raise ValueError(f"ROI must be a 6-int list [z1,z2,y1,y2,x1,x2]; got {roi}")
    Z, Y, X = vol_shape_zyx
    Z0, Y0, X0 = vol_offset_zyx
    z1, z2, y1, y2, x1, x2 = (int(v) for v in roi)

    # Clip to volume bounds
    z1c, z2c = max(z1, Z0), min(z2, Z0 + Z)
    y1c, y2c = max(y1, Y0), min(y2, Y0 + Y)
    x1c, x2c = max(x1, X0), min(x2, X0 + X)
    if z2c <= z1c or y2c <= y1c or x2c <= x1c:
        raise ValueError(f"ROI {roi} is empty after clipping to volume "
                         f"shape={vol_shape_zyx} offset={vol_offset_zyx}")

    # Snap start DOWN to chunk boundary (relative to vol_offset) so block
    # boundaries land on chunk multiples. Snap end UP to chunk boundary so
    # the requested region is fully covered.
    cs = output_chunk_size
    def _snap_down(v, origin): return origin + ((v - origin) // cs) * cs
    def _snap_up(v, origin):
        d = v - origin
        return origin + ((d + cs - 1) // cs) * cs
    z1s, y1s, x1s = _snap_down(z1c, Z0), _snap_down(y1c, Y0), _snap_down(x1c, X0)
    z2s, y2s, x2s = _snap_up(z2c, Z0), _snap_up(y2c, Y0), _snap_up(x2c, X0)
    # Re-clip end to volume bounds (volumes that don't end on chunk
    # boundaries are still valid; the last chunk is partial).
    z2s, y2s, x2s = min(z2s, Z0 + Z), min(y2s, Y0 + Y), min(x2s, X0 + X)

    if (z1s, y1s, x1s, z2s, y2s, x2s) != (z1c, y1c, x1c, z2c, y2c, x2c):
        print(f"[ROI] Snapped requested ROI {roi} -> "
              f"[{z1s}:{z2s}, {y1s}:{y2s}, {x1s}:{x2s}] for chunk-alignment")

    return (z2s - z1s, y2s - y1s, x2s - x1s), (z1s, y1s, x1s)
