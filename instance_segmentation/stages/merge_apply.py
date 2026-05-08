# -*- coding: utf-8 -*-
import os
import json
import gc
import math
import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume
import argparse

from magneton.instance_segmentation.config import (
    load_config,
    get_stage_config,
    load_global_config_path,
)

from magneton.instance_segmentation.utils.meta_utils import load_index_meta
from magneton.instance_segmentation.utils.block_utils import compute_core_region
from magneton.instance_segmentation.utils.relabel_utils import (
    update_id_pools, build_rep_map_from_pools, relabel_array_inplace_with_map
)
from magneton.instance_segmentation.utils.io_utils import export_tif_from_volume
from magneton.instance_segmentation.state.checkpoint import load_merge_state, save_merge_state


def _load_offsets(merge_ckpt_dir):
    p = os.path.join(merge_ckpt_dir, "global_offsets.json")
    with open(p, "r") as f:
        j = json.load(f)
    return {int(k): int(v) for k, v in j["offsets"].items()}, int(j["next_gid"])


def _load_unions(merge_ckpt_dir):
    path = os.path.join(merge_ckpt_dir, "unions.txt")
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            pairs.append((int(a), int(b)))
    return pairs


def _apply_block(block_meta, offsets, rep_map, output_path, core_bounds):
    """
    Worker function: read one block, apply offset + rep_map, trim to core, write.
    No lock needed — caller ensures no two concurrent blocks share chunk files.
    """
    i = block_meta["index"]
    z1, z2, y1, y2, x1, x2 = block_meta["coords"]
    cz1, cz2, cy1, cy2, cx1, cx2 = core_bounds
    off = int(offsets.get(i, 0))

    # Read full block
    local_vol = CloudVolume(block_meta["path"], mip=0, bounded=False, progress=False)
    seg_xyz = local_vol[x1:x2, y1:y2, z1:z2][:, :, :, 0]
    seg_zyx = np.transpose(seg_xyz, (2, 1, 0)).astype(np.uint32, copy=False)
    del seg_xyz

    # Apply global offset
    if off:
        nz = seg_zyx != 0
        seg_zyx[nz] += np.uint32(off)

    # Apply representative mapping
    if rep_map:
        relabel_array_inplace_with_map(seg_zyx, rep_map)

    # Trim to core region (slice in block-local coordinates)
    core_local = seg_zyx[cz1 - z1:cz2 - z1, cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]

    # Write core to output volume (no lock — graph coloring guarantees no chunk conflicts)
    out_xyz = np.transpose(core_local, (2, 1, 0))[:, :, :, np.newaxis]
    out_vol = CloudVolume(output_path, mip=0, bounded=False, progress=False,
                          non_aligned_writes=True, fill_missing=True, compress=False)
    out_vol[cx1:cx2, cy1:cy2, cz1:cz2] = out_xyz

    del seg_zyx, core_local, out_xyz
    gc.collect()
    return i


def _ensure_output_volume(input_path, output_path, mip):
    """Pre-create the segmentation output precomputed (idempotent)."""
    try:
        out_vol = CloudVolume(output_path, mip=mip, progress=False)
        return out_vol
    except Exception:
        pass
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    seg_info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
        resolution=aff_vol.resolution, voxel_offset=aff_vol.voxel_offset,
        volume_size=vol_size_xyz, chunk_size=aff_vol.chunk_size,
    )
    out_vol = CloudVolume(output_path, info=seg_info, compress=False,
                          progress=False, non_aligned_writes=True)
    out_vol.commit_info(); out_vol.commit_provenance()
    return out_vol


def apply_pools_to_global(global_cfg, stage_cfg, task_id=None, blocks_per_task=None):
    """
    Phase 2 worker: process this SLURM array task's slice of blocks.

    The block grid produces cores that are chunk-disjoint by construction
    (compute_core_region snaps interior boundaries to chunk multiples), so
    SLURM array tasks can write concurrently without locks or graph-coloring.

    task_id / blocks_per_task may also be passed via env vars or CLI; if both
    are None, all blocks are processed in one go (legacy local-mode).
    """
    input_path     = global_cfg["paths"]["input"]
    output_path    = global_cfg["paths"]["output"]
    merge_ckpt_dir = global_cfg["checkpoint"]["merge_dir"]

    metadata_dir   = stage_cfg.get("metadata_dir", "./local_metadata")
    mip            = stage_cfg.get("mip", 0)

    export_cfg         = stage_cfg.get("export_tif", {})
    export_tif_enabled = export_cfg.get("enable", False)
    export_tif_path    = export_cfg.get("path", "preview.tif")
    max_slices         = export_cfg.get("max_slices", 200)

    overlap_zyx = tuple(global_cfg.get("block", {}).get("overlap", [0, 0, 0]))

    # Load all done blocks; sorted by index so task slicing is deterministic.
    index_data = load_index_meta(metadata_dir)
    blocks_meta = [b for b in index_data.get("blocks", []) if b.get("done", False)]
    blocks_meta.sort(key=lambda b: b["index"])
    n_blocks = len(blocks_meta)
    print(f"[INFO] Loaded metadata for {n_blocks} blocks")

    # Slice for this SLURM array task. If unset (single-process local mode),
    # process all blocks.
    if task_id is None:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task is not None:
            task_id = int(env_task)
    if task_id is None:
        bid_start, bid_end = 0, n_blocks
    else:
        bpt = int(blocks_per_task or 1)
        bid_start = task_id * bpt
        bid_end = min(bid_start + bpt, n_blocks)
        if bid_start >= bid_end:
            print(f"[INFO] task_id={task_id} has no blocks (n_blocks={n_blocks}).")
            return
    task_blocks = blocks_meta[bid_start:bid_end]
    print(f"[MERGE-APPLY] task_id={task_id} processing blocks "
          f"[{bid_start},{bid_end}) of {n_blocks}")

    # Offsets / unions (every task loads identical small files)
    offsets, next_gid = _load_offsets(merge_ckpt_dir)
    unions = _load_unions(merge_ckpt_dir)
    print(f"[INFO] Loaded {len(unions)} union pairs, next_gid={next_gid}")

    id_pools = []
    for a, b in unions:
        update_id_pools(id_pools, a, b)
    rep_map = build_rep_map_from_pools(id_pools)
    print(f"[INFO] Pools={len(id_pools)}, rep_map entries={len(rep_map)}")

    # Pre-create output volume (idempotent — first task to arrive writes info,
    # subsequent calls reuse). HPC submitter may also create this at submit
    # time, in which case this is a no-op.
    _ensure_output_volume(input_path, output_path, mip)

    # Volume dims and chunk size for snap
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    vol_shape_zyx = (vol_size_xyz[2], vol_size_xyz[1], vol_size_xyz[0])
    chunk_size_xyz = tuple(aff_vol.chunk_size)
    chunk_size_zyx = (chunk_size_xyz[2], chunk_size_xyz[1], chunk_size_xyz[0])

    # Process this task's blocks serially. Cores are chunk-disjoint between
    # adjacent blocks by snap-down construction, so concurrent SLURM tasks
    # writing different blocks never collide on a chunk.
    for blk in tqdm(task_blocks, desc=f"merge_apply task={task_id}"):
        core_bounds = compute_core_region(
            tuple(blk["coords"]), overlap_zyx, vol_shape_zyx, chunk_size_zyx
        )
        _apply_block(blk, offsets, rep_map, output_path, core_bounds)

    # Preview tif export only when running the whole volume in one process
    if export_tif_enabled and task_id is None:
        out_vol = CloudVolume(output_path, mip=mip, bounded=False, progress=False)
        export_tif_from_volume(out_vol, export_tif_path, max_slices=max_slices)

    print(f"[DONE] task_id={task_id} processed {len(task_blocks)} blocks.")


def main():
    parser = argparse.ArgumentParser(description="Apply merge pools to global segmentation precomputed.")
    parser.add_argument("--config", default="configs/config_prec.yaml", type=str, help="Path to configuration YAML.")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM array task id (slices the block list). "
                             "If unset, falls back to SLURM_ARRAY_TASK_ID env var, "
                             "then to single-process mode (all blocks).")
    parser.add_argument("--blocks-per-task", type=int, default=None,
                        help="Blocks processed per SLURM array task.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = get_stage_config(cfg, "merge")
    apply_pools_to_global(cfg, stage_cfg,
                          task_id=args.task_id,
                          blocks_per_task=args.blocks_per_task)


if __name__ == "__main__":
    main()
