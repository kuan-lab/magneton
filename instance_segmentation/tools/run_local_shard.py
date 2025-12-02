# -*- coding: utf-8 -*-
import os
import gc
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from magneton.instance_segmentation.config import load_config, get_stage_config
from magneton.instance_segmentation.stages.segmentation_stage import _process_block   
from magneton.instance_segmentation.state.checkpoint import mark_local_done, is_local_done
from magneton.instance_segmentation.utils.meta_utils import save_block_meta
from magneton.instance_segmentation.utils.block_utils import generate_blocks_zyx
from cloudvolume import CloudVolume


def _get_blocks(cfg):
    input_path = cfg["paths"]["input"]
    mip = cfg.get("local_stage", {}).get("mip", 0)
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    vol_shape_zyx = (vol_size_xyz[2], vol_size_xyz[1], vol_size_xyz[0])
    block_size = tuple(cfg["block"]["size"])
    overlap = tuple(cfg["block"]["overlap"])
    return generate_blocks_zyx(vol_shape_zyx, block_size, overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./instance_segmentation/configs/config.yaml", type=str)
    ap.add_argument("--indices", required=True, type=str, help="Comma-separated block indices, such as: 0,1,2")
    ap.add_argument("--workers", default=2, type=int, help="Number of parallel workers within a node")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stage_cfg = get_stage_config(cfg, "segmentation_stage")
    blocks = _get_blocks(cfg)

    input_path  = cfg["paths"]["input"]
    mask_flag   = cfg["mask"]["flag"]
    mask_path   = cfg["mask"]["path"]
    output_local_base = cfg["paths"]["output_local_base"]

    # thresholds = stage_cfg.get("thresholds", [0.4])
    mip = stage_cfg.get("mip", 0)
    local_ckpt_dir = cfg["checkpoint"]["segmentation_dir"]
    metadata_dir   = stage_cfg.get("metadata_dir", "magneton/local_metadata")

    # Analyzing indices
    idx_list = [int(x) for x in args.indices.strip().split(",") if x.strip() != ""]
    # Filtering completed
    todo = [i for i in idx_list if not is_local_done(local_ckpt_dir, i)]
    if not todo:
        print("[INFO] The blocks corresponding to this task have been completed and skipped.")
        return

    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i in todo:
            coords = blocks[i]
            fut = ex.submit(
                _process_block,
                i, coords,
                input_path=input_path,
                mask_flag=mask_flag,
                mask_path=mask_path,
                output_local_base=output_local_base,
                mip=mip,
                stage_cfg=stage_cfg
            )
            futures.append(fut)

        for fut in as_completed(futures):
            meta = fut.result()
            save_block_meta(metadata_dir, meta)
            mark_local_done(local_ckpt_dir, meta["index"])
            print(f"[INFO] Finished block {meta['index']} (HPC shard), max_id={meta['max_id']}, path={meta['path']}")

    gc.collect()
    print("[DONE] Local shard finished.")


if __name__ == "__main__":
    main()
