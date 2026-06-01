def generate_blocks_zyx(vol_shape_zyx, block_size_zyx, overlap_zyx=(0, 0, 0),
                        origin_zyx=(0, 0, 0)):
    """
    Generate chunks based on volume size (Z, Y, X)
    vol_shape_zyx: (z, y, x)         -- extent to tile (NOT including origin)
    block_size_zyx: (bz, by, bx)
    overlap_zyx: (oz, oy, ox)
    origin_zyx: (oz0, oy0, ox0)      -- absolute voxel coord of the first block;
                                        returned coords are absolute (origin + local).
                                        Defaults to (0, 0, 0) -> identical to the
                                        previous full-volume behavior.

    Returns a deterministic list of (z1, z2, y1, y2, x1, x2) blocks in fixed
    nested-loop order. Pure function of its arguments: every caller that passes
    the same args reconstructs the same grid (and therefore the same block
    indices), with no shared state.
    """
    z_size, y_size, x_size = vol_shape_zyx
    bz, by, bx = block_size_zyx
    oz, oy, ox = overlap_zyx
    oz0, oy0, ox0 = origin_zyx
    blocks = []
    stepz, stepy, stepx = max(1, bz - oz), max(1, by - oy), max(1, bx - ox)

    for z in range(0, z_size, stepz):
        for y in range(0, y_size, stepy):
            for x in range(0, x_size, stepx):
                z2, y2, x2 = min(z + bz, z_size), min(y + by, y_size), min(x + bx, x_size)
                blocks.append((oz0 + z, oz0 + z2, oy0 + y, oy0 + y2, ox0 + x, ox0 + x2))
    return blocks


def apply_roi_zyx(vol_shape_zyx, roi, chunk_zyx=None):
    """
    Restrict a full-volume extent to an ROI sub-region.

    vol_shape_zyx: (Z, Y, X) full volume size (assumes volume origin (0,0,0),
        matching the rest of the instance-segmentation pipeline).
    roi: None for the full volume, or [z1, z2, y1, y2, x1, x2] in absolute voxel
        coords. Bounds are clipped to the actual volume.
    chunk_zyx: optional (cz, cy, cx) precomputed chunk size. When given, the ROI
        start is snapped DOWN and the end UP to chunk boundaries. This is REQUIRED
        for the merge stage: compute_core_region snaps block cores to chunk
        boundaries assuming block starts are chunk-aligned; a non-aligned ROI
        start makes the core start fall below the block start -> negative local
        slice -> empty write / AlignmentError in merge_apply.

    Returns (shape_zyx, origin_zyx): the extent to tile and the absolute coord
    of its first block. With roi=None this is ((Z,Y,X), (0,0,0)) -- a no-op that
    preserves the previous full-volume grid exactly.
    """
    if roi is None:
        return vol_shape_zyx, (0, 0, 0)
    if len(roi) != 6:
        raise ValueError(f"ROI must be a 6-int list [z1,z2,y1,y2,x1,x2]; got {roi}")
    Z, Y, X = vol_shape_zyx
    z1, z2, y1, y2, x1, x2 = (int(v) for v in roi)

    # Clip to volume bounds
    z1, z2 = max(z1, 0), min(z2, Z)
    y1, y2 = max(y1, 0), min(y2, Y)
    x1, x2 = max(x1, 0), min(x2, X)
    if z2 <= z1 or y2 <= y1 or x2 <= x1:
        raise ValueError(f"ROI {roi} is empty after clipping to volume shape {vol_shape_zyx}")

    # Snap to chunk boundaries (start down, end up) so block starts are
    # chunk-aligned -- required for the lock-free merge core-trimming.
    if chunk_zyx is not None:
        cz, cy, cx = (int(c) for c in chunk_zyx)
        snapped = (
            (z1 // cz) * cz, min(-(-z2 // cz) * cz, Z),
            (y1 // cy) * cy, min(-(-y2 // cy) * cy, Y),
            (x1 // cx) * cx, min(-(-x2 // cx) * cx, X),
        )
        if snapped != (z1, z2, y1, y2, x1, x2):
            print(f"[ROI] Snapped to chunk boundaries {chunk_zyx}: "
                  f"[z {z1}:{z2}, y {y1}:{y2}, x {x1}:{x2}] -> "
                  f"[z {snapped[0]}:{snapped[1]}, y {snapped[2]}:{snapped[3]}, x {snapped[4]}:{snapped[5]}]")
        z1, z2, y1, y2, x1, x2 = snapped

    return (z2 - z1, y2 - y1, x2 - x1), (z1, y1, x1)


def build_block_grid(aff_vol, cfg):
    """
    Single source of truth for the block grid across every stage.

    All three grid-recomputing entry points (segmentation_stage serial/parallel,
    run_local_shard, segmentation_stage_hpc) call this so they reconstruct an
    identical, deterministic block list -- and therefore agree on what block
    index `i` means. The grid is a pure function of the input volume's `info`
    size and the `block:` config (size / overlap / optional roi); no runtime
    state is shared between nodes.

    aff_vol: an opened CloudVolume for the input (caller controls the mip).
    cfg: the global config dict (uses cfg["block"]: size, overlap, optional roi).
    """
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    vol_shape_zyx = (vol_size_xyz[2], vol_size_xyz[1], vol_size_xyz[0])

    block_cfg = cfg["block"]
    block_size = tuple(block_cfg["size"])
    overlap = tuple(block_cfg.get("overlap", (0, 0, 0)))
    roi = block_cfg.get("roi", None)

    # Chunk size (xyz -> zyx) used to keep ROI block starts chunk-aligned.
    chunk_xyz = aff_vol.info["scales"][0]["chunk_sizes"][0]
    chunk_zyx = (chunk_xyz[2], chunk_xyz[1], chunk_xyz[0])

    shape_zyx, origin_zyx = apply_roi_zyx(vol_shape_zyx, roi, chunk_zyx=chunk_zyx)
    if roi is not None:
        # For multi-block ROIs, block.size must also be chunk-aligned or interior
        # block starts drift off chunk boundaries and merge_apply mis-trims cores.
        if any(bs % cs != 0 for bs, cs in zip(block_size, chunk_zyx)):
            print(f"[ROI][WARN] block.size {block_size} is not a multiple of chunk {chunk_zyx}; "
                  f"safe only if the ROI fits in a single block.")
        z1, y1, x1 = origin_zyx
        z2, y2, x2 = z1 + shape_zyx[0], y1 + shape_zyx[1], x1 + shape_zyx[2]
        print(f"[ROI] Tiling sub-region [z {z1}:{z2}, y {y1}:{y2}, x {x1}:{x2}] "
              f"of full volume {vol_shape_zyx}")
    return generate_blocks_zyx(shape_zyx, block_size, overlap, origin_zyx=origin_zyx)


def intersect_1d(a1, a2, b1, b2):
    """1D interval intersection"""
    c1 = max(a1, b1)
    c2 = min(a2, b2)
    if c2 <= c1:
        return None, None
    return c1, c2


def intersect_boxes_zyx(A, B):
    """
    Find the intersection of two 3D boxes.
    A/B = (z1,z2,y1,y2,x1,x2)
    """
    z1, z2, y1, y2, x1, x2 = A
    Z1, Z2, Y1, Y2, X1, X2 = B
    zz1, zz2 = intersect_1d(z1, z2, Z1, Z2)
    yy1, yy2 = intersect_1d(y1, y2, Y1, Y2)
    xx1, xx2 = intersect_1d(x1, x2, X1, X2)
    if None in (zz1, yy1, xx1):
        return None
    return (zz1, zz2, yy1, yy2, xx1, xx2)


def _snap_down(val, chunk):
    """Snap val down to nearest multiple of chunk."""
    return (val // chunk) * chunk


def compute_core_region(block_coords, overlap_zyx, vol_shape_zyx, chunk_size_zyx=None):
    """
    Compute the core (non-overlapping) region of a block by trimming overlap/2
    from each interior face. Boundary faces (at volume edges) are not trimmed.

    If chunk_size_zyx is provided, interior boundaries are snapped down to the
    nearest chunk boundary. Adjacent blocks compute the same midpoint and floor
    to the same value, so cores tile with no gaps and no shared chunks.

    block_coords: (z1, z2, y1, y2, x1, x2)
    overlap_zyx: (oz, oy, ox)
    vol_shape_zyx: (Z, Y, X)
    chunk_size_zyx: optional (cz, cy, cx) for chunk-aligned snapping

    Returns: (cz1, cz2, cy1, cy2, cx1, cx2)
    """
    z1, z2, y1, y2, x1, x2 = block_coords
    oz, oy, ox = overlap_zyx
    Z, Y, X = vol_shape_zyx
    pad_z, pad_y, pad_x = oz // 2, oy // 2, ox // 2

    cz1 = z1 + pad_z if z1 > 0 else z1
    cz2 = z2 - pad_z if z2 < Z else z2
    cy1 = y1 + pad_y if y1 > 0 else y1
    cy2 = y2 - pad_y if y2 < Y else y2
    cx1 = x1 + pad_x if x1 > 0 else x1
    cx2 = x2 - pad_x if x2 < X else x2

    if chunk_size_zyx is not None:
        cz, cy, cx = chunk_size_zyx
        if z1 > 0:  cz1 = _snap_down(cz1, cz)
        if z2 < Z:  cz2 = _snap_down(cz2, cz)
        if y1 > 0:  cy1 = _snap_down(cy1, cy)
        if y2 < Y:  cy2 = _snap_down(cy2, cy)
        if x1 > 0:  cx1 = _snap_down(cx1, cx)
        if x2 < X:  cx2 = _snap_down(cx2, cx)

    return (cz1, cz2, cy1, cy2, cx1, cx2)


def compute_chunk_set(core_bounds, chunk_size_zyx):
    """
    Return the set of chunk indices (iz, iy, ix) that a core region touches.

    core_bounds: (cz1, cz2, cy1, cy2, cx1, cx2) in voxel coordinates
    chunk_size_zyx: (cz, cy, cx) chunk dimensions
    """
    cz1, cz2, cy1, cy2, cx1, cx2 = core_bounds
    cz, cy, cx = chunk_size_zyx
    chunks = set()
    for iz in range(cz1 // cz, (cz2 - 1) // cz + 1):
        for iy in range(cy1 // cy, (cy2 - 1) // cy + 1):
            for ix in range(cx1 // cx, (cx2 - 1) // cx + 1):
                chunks.add((iz, iy, ix))
    return chunks


def color_blocks(block_indices, chunk_sets):
    """
    Greedy graph-color blocks by chunk conflict. Two blocks conflict if their
    chunk sets intersect. Returns dict {color_int: [block_index, ...]}.

    block_indices: list of block index ints
    chunk_sets: dict {block_index: set of (iz,iy,ix)}
    """
    # Build adjacency via inverted index: chunk -> list of blocks
    chunk_to_blocks = {}
    for bi in block_indices:
        for c in chunk_sets[bi]:
            chunk_to_blocks.setdefault(c, []).append(bi)

    # Adjacency set per block
    neighbors = {bi: set() for bi in block_indices}
    for blocks_in_chunk in chunk_to_blocks.values():
        for i in range(len(blocks_in_chunk)):
            for j in range(i + 1, len(blocks_in_chunk)):
                neighbors[blocks_in_chunk[i]].add(blocks_in_chunk[j])
                neighbors[blocks_in_chunk[j]].add(blocks_in_chunk[i])

    # Greedy coloring (largest-degree-first for fewer colors)
    order = sorted(block_indices, key=lambda bi: len(neighbors[bi]), reverse=True)
    color_of = {}
    for bi in order:
        used = {color_of[nb] for nb in neighbors[bi] if nb in color_of}
        c = 0
        while c in used:
            c += 1
        color_of[bi] = c

    # Invert to color -> [blocks]
    groups = {}
    for bi, c in color_of.items():
        groups.setdefault(c, []).append(bi)
    return groups
