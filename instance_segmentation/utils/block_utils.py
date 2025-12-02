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
