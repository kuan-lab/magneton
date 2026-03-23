# -*- coding: utf-8 -*-
import os
import json
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
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


def _apply_block(block_meta, offsets, rep_map, output_path, core_bounds, lock):
    """
    Worker function: read one block, apply offset + rep_map, trim to core, write.
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

    # Write core to output volume (serialize to prevent chunk-file races)
    out_xyz = np.transpose(core_local, (2, 1, 0))[:, :, :, np.newaxis]
    out_vol = CloudVolume(output_path, mip=0, bounded=False, progress=False,
                          non_aligned_writes=True, fill_missing=True)
    with lock:
        out_vol[cx1:cx2, cy1:cy2, cz1:cz2] = out_xyz

    del seg_zyx, core_local, out_xyz
    gc.collect()
    return i


def apply_pools_to_global(global_cfg, stage_cfg):
    """
    Phase 2:
    - Read offsets and unions generated in Phase 1
    - Construct id_pools -> rep_map
    - Parallel: read each block, apply offset + rep_map, trim to core, write to out_vol
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

    # Worker count: config > SLURM allocation > CPU count
    workers = int(stage_cfg.get("workers",
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))

    # Block overlap from global config
    overlap_zyx = tuple(global_cfg.get("block", {}).get("overlap", [0, 0, 0]))

    # Read metadata
    index_data = load_index_meta(metadata_dir)
    blocks_meta = [b for b in index_data.get("blocks", []) if b.get("done", False)]
    blocks_meta.sort(key=lambda b: b["index"])
    print(f"[INFO] Loaded metadata for {len(blocks_meta)} blocks")

    # Offsets / unions
    offsets, next_gid = _load_offsets(merge_ckpt_dir)
    unions = _load_unions(merge_ckpt_dir)
    print(f"[INFO] Loaded {len(unions)} union pairs, next_gid={next_gid}")

    # Generate pools
    id_pools = []
    for a, b in unions:
        update_id_pools(id_pools, a, b)
    rep_map = build_rep_map_from_pools(id_pools)
    print(f"[INFO] Pools={len(id_pools)}, rep_map entries={len(rep_map)}")

    # Create global out_vol (using input resolution/voxel_offset/size)
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    vol_shape_zyx = (vol_size_xyz[2], vol_size_xyz[1], vol_size_xyz[0])
    seg_info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
        resolution=aff_vol.resolution, voxel_offset=aff_vol.voxel_offset,
        volume_size=vol_size_xyz, chunk_size=aff_vol.chunk_size,
    )
    out_vol = CloudVolume(output_path, info=seg_info, compress=False,
                          progress=False, non_aligned_writes=True)
    out_vol.commit_info(); out_vol.commit_provenance()

    # Compute core regions for all blocks
    core_bounds = {}
    for blk in blocks_meta:
        core_bounds[blk["index"]] = compute_core_region(
            tuple(blk["coords"]), overlap_zyx, vol_shape_zyx
        )

    print(f"[INFO] Starting parallel apply with {workers} workers, "
          f"overlap={overlap_zyx}, core trimming enabled")

    # Parallel block processing
    manager = Manager()
    lock = manager.Lock()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for blk in blocks_meta:
            fut = ex.submit(
                _apply_block,
                blk, offsets, rep_map, output_path,
                core_bounds[blk["index"]], lock
            )
            futures[fut] = blk["index"]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Apply Pools (parallel)"):
            try:
                block_idx = fut.result()
            except Exception as e:
                print(f"[WARN] Block {futures[fut]} failed: {e}")

    # Optional: export preview
    if export_tif_enabled:
        from magneton.instance_segmentation.utils.io_utils import export_tif_from_volume
        export_tif_from_volume(out_vol, export_tif_path, max_slices=max_slices)

    print("[DONE] Phase-2 finished, global volume ready.")


def main():
    parser = argparse.ArgumentParser(description="Convert 3D/4D TIFF or HDF5 to Neuroglancer Precomputed format.")
    parser.add_argument("--config", default="configs/config_prec.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = get_stage_config(cfg, "merge")
    apply_pools_to_global(cfg, stage_cfg)


if __name__ == "__main__":
    main()
