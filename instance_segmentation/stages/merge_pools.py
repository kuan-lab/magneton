# -*- coding: utf-8 -*-
import os
import json
import math
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume

from magneton.instance_segmentation.config import (
    load_config,
    get_stage_config,
    load_global_config_path,
)

from magneton.instance_segmentation.utils.meta_utils import load_index_meta
from magneton.instance_segmentation.utils.block_utils import intersect_boxes_zyx
from magneton.instance_segmentation.utils.relabel_utils import (
    accumulate_local_global_pairs,
    update_id_pools,               # For optional memory aggregation only
    build_rep_map_from_pools,      # Optional
)
from magneton.instance_segmentation.utils.relabel_utils import select_pairs 


def _compute_global_offsets(blocks_meta, start_gid=1):
    """
    Assign a global starting offset to each block based on the max_id in the metadata, ensuring cross-block ID uniqueness.
    offsets[i] = global offset (sum only for non-zero IDs)
    Returns: offsets: {index -> offset}, next_gid: int
    """
    done_blocks = [b for b in blocks_meta if b.get("done", False)]
    done_blocks.sort(key=lambda b: b["index"])
    offsets = {}
    cur = int(start_gid)
    for b in done_blocks:
        i = int(b["index"])
        offsets[i] = cur
        mmax = int(b.get("max_id", 0))
        if mmax > 0:
            cur += mmax
    return offsets, cur


def _pairs_for_overlaps(blocks_meta):
    """
    List all pairs of blocks (i, j, ov_zyx, global_box_i, global_box_j) that intersect, where i < j.
    ov_zyx = (zz1, zz2, yy1, yy2, xx1, xx2)
    """
    done = [b for b in blocks_meta if b.get("done", False)]
    done.sort(key=lambda b: b["index"])
    pairs = []
    #  Simple O(N^2); if there are many blocks, grid/index optimization can be used.
    for a in range(len(done)):
        i = done[a]["index"]
        Ai = tuple(done[a]["coords"])
        for b in range(a + 1, len(done)):
            j = done[b]["index"]
            Bj = tuple(done[b]["coords"])
            ov = intersect_boxes_zyx(Ai, Bj)
            if ov is None:
                continue
            pairs.append((i, j, ov, Ai, Bj))
    return pairs


def _overlap_union_task(
    i, j, ov, Ai, Bj,
    path_i, path_j,
    offset_i, offset_j,
    thresholds_pack
):
    """
    Child process task: Read two partitions in the overlap region, apply a global offset, count pairs, and select union pairs.
    Return: [(gid_a, gid_b), ...] where gid_* is a globally unique ID with the offset already applied.
    """
    (zz1, zz2, yy1, yy2, xx1, xx2) = ov
    (z1, z2, y1, y2, x1, x2) = Ai
    (Z1, Z2, Y1, Y2, X1, X2) = Bj

    min_overlap_vox, min_frac_local, min_frac_global, max_voxel_size, require_recip, allow_union_amb, dom_ratio, min_iou = thresholds_pack

    # Read both sides of the overlap (using CloudVolume's global slice: xyz)
    vi = CloudVolume(path_i, mip=0, bounded=False, progress=False)
    vj = CloudVolume(path_j, mip=0, bounded=False, progress=False)
    a_xyz = vi[xx1:xx2, yy1:yy2, zz1:zz2][:, :, :, 0]
    b_xyz = vj[xx1:xx2, yy1:yy2, zz1:zz2][:, :, :, 0]
    a = np.transpose(a_xyz, (2, 1, 0)).astype(np.uint32, copy=False)  # zyx
    b = np.transpose(b_xyz, (2, 1, 0)).astype(np.uint32, copy=False)

    # Global offset, ensuring cross-block uniqueness
    if offset_i:
        ai = a != 0
        a[ai] += np.uint32(offset_i)
    if offset_j:
        bj = b != 0
        b[bj] += np.uint32(offset_j)

    pair_counts = {}
    accumulate_local_global_pairs(a, b, pair_counts)

    if not pair_counts:
        return []

    selected = select_pairs(
        pair_counts=pair_counts,
        min_overlap_vox=min_overlap_vox,
        min_frac_local=min_frac_local,
        min_frac_global=min_frac_global,
        max_voxel_size = max_voxel_size,
        require_reciprocal=require_recip,
        allow_union_ambiguity=allow_union_amb,
        dom_ratio=dom_ratio,
        min_iou=min_iou,
        debug=False
    )
    # Return the global ID pair directly
    return [(int(la), int(gb)) for (la, gb) in selected]


def build_id_pools_parallel(global_cfg, stage_cfg, restart=False):
    """
    Phase 1:
    - Calculate global block offsets based on metadata (using max_id prefix sums)
    - Parallel traverse all intersecting block pairs, count overlaps, select pairs, and generate union pairs
    - Write union pairs to merge_ckpt_dir/unions.txt (each line: “<a> <b>”),
        and write merge_ckpt_dir/global_offsets.json
    """
    metadata_dir   = stage_cfg.get("metadata_dir", "./local_metadata")
    merge_ckpt_dir = global_cfg["checkpoint"]["merge_dir"]

    # Thresholds Package
    thresholds_pack = (
        stage_cfg.get("min_overlap_vox", 20),
        stage_cfg.get("min_frac_local", 0.7),
        stage_cfg.get("min_frac_global", 0.7),
        stage_cfg.get("max_voxel_size", 100000000),
        stage_cfg.get("require_recip", False),
        stage_cfg.get("allow_union_amb", True),
        stage_cfg.get("dom_ratio", 1.0),
        stage_cfg.get("min_iou", 0.7),
    )

    # Read metadata
    index_data = load_index_meta(metadata_dir)
    blocks_meta = index_data.get("blocks", [])
    print(f"[INFO] Loaded metadata for {len(blocks_meta)} blocks")

    # Global Offset
    offsets, next_gid = _compute_global_offsets(blocks_meta, start_gid=1)
    os.makedirs(merge_ckpt_dir, exist_ok=True)
    with open(os.path.join(merge_ckpt_dir, "global_offsets.json"), "w") as f:
        json.dump({"offsets": offsets, "next_gid": next_gid}, f, indent=2)

    # List all intersecting block pairs
    pairs = _pairs_for_overlaps(blocks_meta)
    if not pairs:
        print("[INFO] No overlapping pairs found.")
        # Still writing blank, unions.txt
        open(os.path.join(merge_ckpt_dir, "unions.txt"), "w").close()
        return

    unions_path = os.path.join(merge_ckpt_dir, "unions.txt")
    # Reset mode
    if restart and os.path.exists(unions_path):
        os.remove(unions_path)

    workers = int(stage_cfg.get("workers", os.cpu_count() or 1))
    print(f"[INFO] Overlap pairs: {len(pairs)}; dispatch with {workers} workers.")

    # Create an ndex->path mapping
    path_by_idx = {b["index"]: b["path"] for b in blocks_meta if b.get("done", False)}
    # Parallel processing
    with ProcessPoolExecutor(max_workers=workers) as ex, open(unions_path, "a") as out:
        futs = []
        for (i, j, ov, Ai, Bj) in pairs:
            futs.append(ex.submit(
                _overlap_union_task,
                i, j, ov, Ai, Bj,
                path_by_idx[i], path_by_idx[j],
                int(offsets[i]), int(offsets[j]),
                thresholds_pack
            ))
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Pools Phase (pairs)"):
            try:
                pairs_sel = fut.result()
                for a, b in pairs_sel:
                    out.write(f"{a} {b}\n")
            except Exception as e:
                print(f"[WARN] pair task failed: {e}")

    print(f"[DONE] Pooling finished. unions -> {unions_path}, offsets -> global_offsets.json")

def main():
    parser = argparse.ArgumentParser(description="Convert 3D/4D TIFF or HDF5 to Neuroglancer Precomputed format.")
    parser.add_argument("--config", default="configs/config_prec.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = get_stage_config(cfg, "merge")
    build_id_pools_parallel(cfg, stage_cfg)


if __name__ == "__main__":
    main()