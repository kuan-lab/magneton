def generate_blocks_zyx(vol_shape_zyx, block_size_zyx, overlap_zyx=(0, 0, 0)):
    """
    Generate chunks based on volume size (Z, Y, X)
    vol_shape_zyx: (z, y, x)
    block_size_zyx: (bz, by, bx)
    overlap_zyx: (oz, oy, ox)
    """
    z_size, y_size, x_size = vol_shape_zyx
    bz, by, bx = block_size_zyx
    oz, oy, ox = overlap_zyx
    blocks = []
    stepz, stepy, stepx = max(1, bz - oz), max(1, by - oy), max(1, bx - ox)

    for z in range(0, z_size, stepz):
        for y in range(0, y_size, stepy):
            for x in range(0, x_size, stepx):
                z2, y2, x2 = min(z + bz, z_size), min(y + by, y_size), min(x + bx, x_size)
                blocks.append((z, z2, y, y2, x, x2))
    return blocks


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
