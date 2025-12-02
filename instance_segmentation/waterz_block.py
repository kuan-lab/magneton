import numpy as np
from scipy.ndimage import distance_transform_edt, watershed_ift
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from waterz import agglomerate
import mahotas

# ---------- Foundation ----------
def boundary_from_aff(aff):
    """
    Generating boundary diagrams from affinity map
    aff: (c,z,y,x)
    """
    # aff_local = aff.copy()
    B = 1.0 - aff.mean(axis=0)  # (z,y,x)
    return np.ascontiguousarray(B.astype(np.float32, copy=False))

def compact_labels_uint32(labels):
    """
    Compress label IDs into a continuous range [0..N]
    """
    lab = np.asarray(labels)
    ids = np.unique(lab)
    if ids.size == 0 or (ids.size == 1 and ids[0] == 0):
        return lab.astype(np.uint32, copy=False), np.arange(1, dtype=np.uint32)
    if ids[0] != 0:
        ids = np.insert(ids, 0, 0)
    lut = np.zeros(int(ids.max()) + 1, dtype=np.uint32)
    lut[ids] = np.arange(ids.size, dtype=np.uint32)
    comp = lut[lab].astype(np.uint32, copy=False)
    return np.ascontiguousarray(comp), lut

def seeds_3d_from_B(B, interior_thr=0.4, min_distance=15):
    """
    Generate seed points from boundaries (watershed markers)
    """
    interior = 1.0 - B
    mask = interior > interior_thr
    if not np.any(mask):
        thr = float(np.percentile(interior, 70.0))
        mask = interior > thr
    D = distance_transform_edt(mask)
    coords = peak_local_max(D, min_distance=min_distance, labels=mask, exclude_border=False)
    markers = np.zeros(B.shape, np.int32)
    for i, (z, y, x) in enumerate(coords, 1):
        markers[z, y, x] = i
    if markers.max() == 0 and np.any(mask):
        zmax = int(np.argmax(D.reshape(D.shape[0], -1).max(axis=1)))
        zy, zx = np.unravel_index(int(D[zmax].argmax()), D[zmax].shape)
        markers[zmax, zy, zx] = 1
    return np.ascontiguousarray(markers), mask

def getScoreFunc(scoreF="aff50_his256"):
    """
    Return the waterz scoring function (simplified version)
    """
    config = {x[:3]: x[3:] for x in scoreF.split('_')}
    print("waterz scoring:", config)
    if 'aff' in config:
        if 'his' in config and config['his'] != '0':
            return f"OneMinus<HistogramQuantileAffinity<RegionGraphType, {config['aff']}, ScoreValue, {config['his']}>>"
        else:
            return f"OneMinus<QuantileAffinity<RegionGraphType, {config['aff']}, ScoreValue>>"
    elif 'max' in config:
        return f"OneMinus<MeanMaxKAffinity<RegionGraphType, {config['max']}, ScoreValue>>"
    else:
        return "OneMinus<QuantileAffinity<RegionGraphType, 50, ScoreValue>>"

def get_seeds_2d(boundary, method='grid', next_id = 1,
             seed_distance = 10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]

        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0
    return seeds, num_seeds

def watershed_2d(affs, seed_method, use_mahotas_watershed = True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds_2d(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds

    return fragments


# ---------- Main ----------
def run_waterz_block(
    aff_block_czyx,
    mask=None,
    seg_thresholds=[0.4],
    aff_thresholds=[0.00001, 0.99999],
    sv_type="3d",
    interior_thr=0.1,
    min_distance=3,
    sv_2d='maxima_distance',
    merge_function=None,
):
    """
    Perform waterz partitioning within a block
    aff_block_czyx: (c,z,y,x)
    """
    aff = aff_block_czyx.astype(np.float32)
    if aff.max() > 1.0:
        aff /= 255.0
    aff = np.ascontiguousarray(aff.astype(np.float32))
    # Generate initial watershed
    if sv_type == "3d":
        B = boundary_from_aff(aff)
        markers, _ = seeds_3d_from_B(B, interior_thr=interior_thr, min_distance=min_distance)
        supervox = watershed(B, markers=markers, mask=mask).astype(np.int32, copy=False)
    elif sv_type == "2d":
        supervox = watershed_2d(aff, sv_2d) # sv_2d: grid, minima and maxima_distance
    else:
        raise RuntimeError("Supervoxle should be 3d or 2d.")
    if supervox.max() == 0:
        print("Watershed produced no segments.")
        return np.zeros_like(B, dtype=np.uint32)
        # raise RuntimeError("Watershed produced no segments.")
    supervox, _ = compact_labels_uint32(supervox)
    supervox = np.ascontiguousarray(supervox.astype(np.uint64, copy=False))
    # Run waterz aggregation
    outs = []
    for out in agglomerate(
        aff,
        seg_thresholds,
        aff_threshold_low=aff_thresholds[0],
        aff_threshold_high=aff_thresholds[1],
        fragments=supervox,
        scoring_function=getScoreFunc(merge_function),
        discretize_queue=256,
        force_rebuild=True
    ):
        out = np.ascontiguousarray(out)
        outs.append(out.copy())

    seg = outs[0] if isinstance(outs, list) else next(outs)
    return seg.astype(np.uint32, copy=False)
