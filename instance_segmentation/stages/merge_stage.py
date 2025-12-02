import os
import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume

from magneton.instance_segmentation.utils.meta_utils import load_index_meta
from magneton.instance_segmentation.utils.relabel_utils import (
    accumulate_local_global_pairs, update_id_pools,
    build_rep_map_from_pools, relabel_array_inplace_with_map
)
from magneton.instance_segmentation.state.checkpoint import load_merge_state, save_merge_state
from magneton.instance_segmentation.utils.io_utils import export_tif_from_volume


def merge_local_blocks(global_cfg, stage_cfg,
                       restart=False, force_overlap_identity=False):
    """
    Execute merge stage:
    - Read block information from local_metadata/index.json
    - Merge per-block segmentation into the global CloudVolume
    - Resolve cross-block overlaps using ID pools
    """

    input_path  = global_cfg["paths"]["input"]
    output_path = global_cfg["paths"]["output"]
    merge_ckpt_dir = global_cfg["checkpoint"]["merge_dir"]

    metadata_dir    = stage_cfg.get("metadata_dir", "./local_metadata")
    # min_overlap_vox = stage_cfg.get("min_overlap_vox", 20)
    # min_frac_local  = stage_cfg.get("min_frac_local", 0.7)
    # min_frac_global = stage_cfg.get("min_frac_global", 0.7)
    # require_recip   = stage_cfg.get("require_recip", False)
    # allow_union_amb = stage_cfg.get("allow_union_amb", True)
    # dom_ratio       = stage_cfg.get("dom_ratio", 1.0)
    # min_iou         = stage_cfg.get("min_iou", 0.7)
    mip             = stage_cfg.get("mip", 0)

    export_cfg      = stage_cfg.get("export_tif", {})
    export_tif_enabled = export_cfg.get("enable", False)
    export_tif_path    = export_cfg.get("path", "preview.tif")
    max_slices         = export_cfg.get("max_slices", 200)

    # Open and enter affinity to obtain size/resolution.
    
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])

    # Create a global output volume
    seg_info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
        resolution=aff_vol.resolution, voxel_offset=aff_vol.voxel_offset,
        volume_size=vol_size_xyz, chunk_size=aff_vol.chunk_size,
    )
    out_vol = CloudVolume(output_path, info=seg_info, compress=False,
                          progress=False, non_aligned_writes=True)
    out_vol.commit_info()
    out_vol.commit_provenance()

    # Read metadata
    index_data = load_index_meta(metadata_dir)
    blocks_meta = index_data.get("blocks", [])
    print(f"[INFO] Loaded metadata for {len(blocks_meta)} blocks")

    # Initialize global ID pools
    id_pools = []

    # Load merge state
    state = load_merge_state(merge_ckpt_dir)
    next_gid = int(state.get("next_gid", 1))
    merged_blocks = set(state.get("merged_blocks", []))

    # Traverse block
    for blk in tqdm(blocks_meta, desc="Merge Blocks"):
        try:
            i = blk["index"]
            if not blk.get("done", False):
                continue
            if i in merged_blocks and not restart:
                continue

            z1, z2, y1, y2, x1, x2 = blk["coords"]
            in_path = blk["path"]

            try:
                local_vol = CloudVolume(in_path, mip=0, bounded=False, progress=False)
                seg_local = local_vol[:][:,:,:,0]
                seg_local = np.transpose(seg_local, (2, 1, 0))  # (z,y,x)
            except Exception as e:
                raise RuntimeError(f"Failed to read local block {i} at {in_path}: {e}")

            # Offset ID
            mmax = int(seg_local.max())
            if mmax > 0:
                seg_local[seg_local != 0] += next_gid
                next_gid += mmax

            # Write to the global volume
            seg_xyz = np.transpose(seg_local, (2, 1, 0))
            out_vol[x1:x2, y1:y2, z1:z2] = seg_xyz[:, :, :, np.newaxis]

            # Update checkpoint
            merged_blocks.add(i)
            save_merge_state(merge_ckpt_dir, {
                "next_gid": next_gid,
                "merged_blocks": sorted(list(merged_blocks)),
                "num_pools": len(id_pools)
            })

            print(f"[INFO] Merged block {i}, gid advanced to {next_gid}")
        except KeyboardInterrupt:
            break

    # Construct the final rep_map and relabel
    final_rep_map = build_rep_map_from_pools(id_pools)
    if final_rep_map:
        print(f"[INFO] Applying final relabel with {len(final_rep_map)} entries...")
        for blk in tqdm(blocks_meta, desc="FinalRelabel"):
            if not blk.get("done", False):
                continue
            z1, z2, y1, y2, x1, x2 = blk["coords"]
            seg_blk = out_vol[x1:x2, y1:y2, z1:z2][:, :, :, 0]
            seg_blk = np.transpose(seg_blk, (2, 1, 0))
            relabel_array_inplace_with_map(seg_blk, final_rep_map)
            seg_xyz = np.transpose(seg_blk, (2, 1, 0))
            out_vol[x1:x2, y1:y2, z1:z2] = seg_xyz[:, :, :, np.newaxis]
    else:
        print("[INFO] No ID pools recorded; skipping final mapping pass.")

    # Optional TIFF Export
    if export_tif_enabled:
        export_tif_from_volume(out_vol, export_tif_path, max_slices=max_slices)

    print("[DONE] Merge stage finished.")
