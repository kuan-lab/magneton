import time
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
    max_id = int(lab.max())
    if max_id == 0:
        return lab.astype(np.uint32, copy=False), np.arange(1, dtype=np.uint32)
    # O(n) flag scan instead of O(n log n) np.unique sort
    present = np.zeros(max_id + 1, dtype=np.bool_)
    present[0] = True
    present[lab.ravel()] = True
    ids = np.where(present)[0]
    lut = np.zeros(max_id + 1, dtype=np.uint32)
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
    if len(coords) > 0:
        markers[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(1, len(coords) + 1)
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
    return_fragments=False,
):
    """
    Perform waterz partitioning within a block
    aff_block_czyx: (c,z,y,x)
    return_fragments: if True, return (supervox, seg) — the pre-agglomeration
        watershed supervoxels (uint64) alongside the agglomerated seg — for
        supervoxel-based proofreading (WebKnossos agglomerate files). Both (z,y,x).
    """
    t_total = time.time()
    aff = aff_block_czyx.astype(np.float32)
    if aff.max() > 1.0:
        aff /= 255.0

    # Handle single-channel affinity (e.g., mito interior probability)
    # by duplicating to 3 channels for waterz compatibility
    if aff.shape[0] == 1:
        print(f"[INFO] Single-channel affinity detected, duplicating to 3 channels for waterz")
        aff = np.concatenate([aff, aff, aff], axis=0)

    aff = np.ascontiguousarray(aff.astype(np.float32))
    vol_voxels = aff.shape[1] * aff.shape[2] * aff.shape[3]
    print(f"[TIMER] Block shape (c,z,y,x): {aff.shape}, voxels: {vol_voxels:,}")

    # Generate initial watershed
    if sv_type == "3d":
        t0 = time.time()
        B = boundary_from_aff(aff)
        t1 = time.time()
        print(f"[TIMER] boundary_from_aff: {t1 - t0:.2f}s")

        markers, _ = seeds_3d_from_B(B, interior_thr=interior_thr, min_distance=min_distance)
        t2 = time.time()
        print(f"[TIMER] seeds_3d_from_B (EDT + peak_local_max): {t2 - t1:.2f}s, num_seeds: {markers.max()}")

        supervox = watershed(B, markers=markers, mask=mask).astype(np.int32, copy=False)
        t3 = time.time()
        print(f"[TIMER] watershed_3d: {t3 - t2:.2f}s")
    elif sv_type == "2d":
        t0 = time.time()
        supervox = watershed_2d(aff, sv_2d) # sv_2d: grid, minima and maxima_distance
        t3 = time.time()
        print(f"[TIMER] watershed_2d: {t3 - t0:.2f}s")
    else:
        raise RuntimeError("Supervoxle should be 3d or 2d.")
    if supervox.max() == 0:
        print("Watershed produced no segments.")
        empty = np.zeros_like(B, dtype=np.uint32)
        return (empty.astype(np.uint64), [empty]) if return_fragments else empty
        # raise RuntimeError("Watershed produced no segments.")

    t4 = time.time()
    supervox, _ = compact_labels_uint32(supervox)
    supervox = np.ascontiguousarray(supervox.astype(np.uint64, copy=False))
    t5 = time.time()
    num_supervox = int(supervox.max())
    print(f"[TIMER] compact_labels: {t5 - t4:.2f}s, num_supervoxels: {num_supervox}")

    # waterz agglomerate() mutates `fragments` IN PLACE, so snapshot the watershed
    # supervoxels first if the caller wants them back.
    supervox_frozen = supervox.copy() if return_fragments else None

    # Run waterz aggregation
    t6 = time.time()
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
    t7 = time.time()
    print(f"[TIMER] waterz_agglomerate: {t7 - t6:.2f}s")

    seg = outs[0] if isinstance(outs, list) else next(outs)
    t_end = time.time()
    print(f"[TIMER] TOTAL run_waterz_block: {t_end - t_total:.2f}s")
    if return_fragments:
        # return the watershed supervoxels + ONE agglomeration per threshold (the
        # generator yields outs aligned with seg_thresholds), for per-threshold
        # WebKnossos agglomerate files.
        return (supervox_frozen.astype(np.uint64, copy=False),
                [o.astype(np.uint32, copy=False) for o in outs])
    return seg.astype(np.uint32, copy=False)
