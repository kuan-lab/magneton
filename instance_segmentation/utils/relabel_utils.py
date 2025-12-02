import numpy as np
from collections import defaultdict

# ---------- ID pool operations ----------
def update_id_pools(id_pools: list, a: int, b: int):
    """Put a and b in the same pool."""
    if a == 0 or b == 0:
        return
    found = []
    for idx, s in enumerate(id_pools):
        if a in s or b in s:
            found.append(idx)
    if not found:
        id_pools.append(set([a, b]))
        return
    if len(found) == 1:
        id_pools[found[0]].update([a, b])
        return
    merged = set([a, b])
    for idx in sorted(found, reverse=True):
        merged.update(id_pools[idx])
        del id_pools[idx]
    id_pools.append(merged)

def build_rep_map_from_pools(id_pools: list):
    """Construct a mapping {id -> representative id} from the pool."""
    rep_map = {}
    for s in id_pools:
        if not s:
            continue
        rep = int(min(s))
        for x in s:
            rep_map[int(x)] = rep
    return rep_map

def _build_totals_and_best(pair_counts):
    by_local  = defaultdict(int)
    by_global = defaultdict(int)
    for (la, gb), c in pair_counts.items():
        by_local[la]  += c
        by_global[gb] += c

    best_gb_for_la = {}
    second_la = {}
    tmp_map = defaultdict(list)
    for (la, gb), c in pair_counts.items():
        tmp_map[la].append((gb, c))
    for la, lst in tmp_map.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        best_gb_for_la[la] = lst[0][0]
        second_la[la] = lst[1][1] if len(lst) > 1 else 0

    best_la_for_gb = {}
    second_gb = {}
    tmp_map.clear()
    for (la, gb), c in pair_counts.items():
        tmp_map[gb].append((la, c))
    for gb, lst in tmp_map.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        best_la_for_gb[gb] = lst[0][0]
        second_gb[gb] = lst[1][1] if len(lst) > 1 else 0

    return by_local, by_global, best_gb_for_la, best_la_for_gb, second_la, second_gb

def select_pairs(
    pair_counts: dict,
    min_overlap_vox: int,
    min_frac_local: float,
    min_frac_global: float,
    max_voxel_size: int,
    require_reciprocal: bool,
    allow_union_ambiguity: bool,
    dom_ratio: float,
    min_iou: float,
    debug: bool = False
):
    if not pair_counts:
        return []

    (by_local, by_global,
     best_gb_for_la, best_la_for_gb,
     second_la, second_gb) = _build_totals_and_best(pair_counts)

    candidates = []
    for (la, gb), c in pair_counts.items():
        if la == 0 or gb == 0:
            continue
        if c < int(min_overlap_vox):
            continue

        tot_la = by_local.get(la, 0)
        tot_gb = by_global.get(gb, 0)
        if tot_la == 0 or tot_gb == 0:
            continue

        frac_local  = c / float(tot_la)
        frac_global = c / float(tot_gb)
        
        if tot_gb > max_voxel_size:
            continue
        if frac_local  < float(min_frac_local) and frac_global < float(min_frac_global):
            continue
        
        # if frac_global < float(min_frac_global):
        #     continue

        # if require_reciprocal:
        #     if not (best_gb_for_la.get(la) == gb and best_la_for_gb.get(gb) == la):
        #         continue

        # sec_la = second_la.get(la, 0)
        # sec_gb = second_gb.get(gb, 0)
        # if sec_la > 0 and c < sec_la * dom_ratio:
        #     continue
        # if sec_gb > 0 and c < sec_gb * dom_ratio:
        #     continue

        denom = (tot_la + tot_gb - c)
        iou = c / float(denom) if denom > 0 else 0.0

        candidates.append((la, gb, c, frac_local, frac_global, iou))

    if not candidates:
        return []

    if allow_union_ambiguity:
        if debug:
            candidates.sort(key=lambda x: (x[2], x[5]), reverse=True)
            for la, gb, c, fl, fg, iou in candidates[:10]:
                print(f"[DEBUG] cand la={la} gb={gb} c={c} fracL={fl:.3f} fracG={fg:.3f} IoU={iou:.3f}")
        return [(la, gb) for la, gb, *_ in candidates]

    candidates.sort(key=lambda x: (x[2], x[5]), reverse=True)
    used_la = set()
    used_gb = set()
    selected = []
    for la, gb, c, fl, fg, iou in candidates:
        if la in used_la and gb in used_gb:
            continue
        selected.append((la, gb))
        used_la.add(la)
        used_gb.add(gb)
    if debug:
        print(f"[DEBUG] selected {len(selected)} pairs (1-1), from {len(candidates)} candidates")
    return selected

# ---------- relabel ----------
def relabel_array_inplace_with_map(arr: np.ndarray, mapping: dict):
    """In-place relabeling ID"""
    ids = np.unique(arr)
    ids = ids[ids != 0]
    if ids.size == 0 or not mapping:
        return

    mapped_ids = np.array([int(x) for x in ids if int(x) in mapping], dtype=np.uint32)
    if mapped_ids.size == 0:
        return

    max_id = int(max(ids.max(), max(mapped_ids)))
    DENSE_MAX_BYTES = 128 * 1024 * 1024  # 128MB limitation
    use_dense = (max_id + 1) * 4 <= DENSE_MAX_BYTES and (mapped_ids.size / (max_id + 1)) > 0.1

    if use_dense:
        table = np.arange(max_id + 1, dtype=np.uint32)
        for k, v in mapping.items():
            if k <= max_id:
                table[k] = np.uint32(v)
        arr[:] = table[arr]
        return

    flat = arr.ravel()
    nz = flat != 0
    vals = flat[nz].astype(np.uint32, copy=False)

    keys = np.fromiter(mapping.keys(), dtype=np.uint32)
    vals_map = np.fromiter(mapping.values(), dtype=np.uint32)
    if keys.size == 0:
        return
    sorter = np.argsort(keys)
    keys_sorted = keys[sorter]
    vals_sorted = vals_map[sorter]

    idx = np.searchsorted(keys_sorted, vals, side='left')
    valid = idx < keys_sorted.size

    match = np.zeros(vals.shape, dtype=bool)
    match[valid] = (keys_sorted[idx[valid]] == vals[valid])

    if match.any():
        flat_idx = np.nonzero(nz)[0]
        flat[flat_idx[match]] = vals_sorted[idx[match]]

# ---------- overlap statistics ----------
def accumulate_local_global_pairs(seg_local_zyx: np.ndarray,
                                  seg_global_overlap_zyx: np.ndarray,
                                  pair_counts: dict):
    """Count the frequency of paired IDs with local and global overlap"""
    a = seg_local_zyx
    b = seg_global_overlap_zyx
    m = (a != 0) & (b != 0)
    if not np.any(m):
        return
    a1 = a[m].astype(np.uint32, copy=False)
    b1 = b[m].astype(np.uint32, copy=False)
    keys = (a1.astype(np.uint64) << 32) | b1.astype(np.uint64)
    uniq, cnt = np.unique(keys, return_counts=True)
    la = (uniq >> 32).astype(np.uint32)
    gb = (uniq & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    for u_la, u_gb, c in zip(la.tolist(), gb.tolist(), cnt.tolist()):
        pair_counts[(u_la, u_gb)] = pair_counts.get((u_la, u_gb), 0) + int(c)
