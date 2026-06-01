import os
import gc
import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume
from concurrent.futures import ProcessPoolExecutor, as_completed

from magneton.instance_segmentation.waterz_block import run_waterz_block
from magneton.instance_segmentation.mito_block import run_mito_block
from magneton.instance_segmentation.bouton_block import run_bouton_block, read_neuron_ref_block
from magneton.instance_segmentation.utils.block_utils import build_block_grid
from magneton.instance_segmentation.state.checkpoint import mark_local_done, is_local_done
from magneton.instance_segmentation.utils.meta_utils import save_block_meta

def segmentation_blocks(global_cfg, stage_cfg, restart=False):
    """
    Execute local stage:
    - Partition large-volume affinity
    - Run `run_waterz_block` for each block
    - Output per-block CloudVolume
    - Write metadata and checkpoint
    """

    input_path  = global_cfg["paths"]["input"]
    mask_flag   = global_cfg["mask"]["flag"]
    mask_path   = global_cfg["mask"]["path"]
    output_local_base = global_cfg["paths"]["output_local_base"]

    block_size  = tuple(global_cfg["block"]["size"])
    overlap     = tuple(global_cfg["block"]["overlap"])

    local_ckpt_dir = global_cfg["checkpoint"]["segmentation_dir"]
    metadata_dir   = stage_cfg.get("metadata_dir", "./local_metadata")
    mip            = stage_cfg.get("mip", 0)

    # Mode configuration (neuron vs mito)
    mode_cfg = global_cfg.get("mode", {})
    mode_type = mode_cfg.get("type", "neuron")
    mito_cfg = mode_cfg.get("mito", {})
    bouton_cfg = mode_cfg.get("bouton", {})

    # Neuron mode (waterz) parameters
    thresholds     = stage_cfg.get("thresholds", [0.4])
    aff_thresholds = stage_cfg.get("aff_thresholds", [0.00001, 0.99999])
    sv_type        = stage_cfg.get("sv_type", "3d")
    interior_thr   = stage_cfg.get("interior_thr", 0.1)
    min_distance   = stage_cfg.get("min_distance", 3)
    sv_2d          = stage_cfg.get("sv_2d", 'maxima_distance')
    merge_function = stage_cfg.get("merge_function", 'aff50_his256' )

    print(f"[INFO] Segmentation mode: {mode_type}")

    # Open the volume input
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    if mask_flag:
        mask_vol = CloudVolume(mask_path, mip=0, bounded=False, progress=False)
    else:
        mask_vol = None
    # Bouton mode: reference neuron affinity volume for membrane gating
    if mode_type == "bouton":
        # Neuron ref opened at ITS OWN finest mip (mip=0); resolution mismatch
        # with the bouton input is handled by read_neuron_ref_block.
        neuron_ref_vol = CloudVolume(bouton_cfg["neuron_ref_path"], mip=0,
                                     bounded=False, fill_missing=True, progress=False)
    else:
        neuron_ref_vol = None
    # Generate blocks (single source of truth: ROI-aware, deterministic)
    blocks = build_block_grid(aff_vol, global_cfg)

    if restart:
        print(f"[INFO] Restart mode: clearing local checkpoints and metadata at {local_ckpt_dir}, {metadata_dir}")
        if os.path.exists(local_ckpt_dir):
            for fn in os.listdir(local_ckpt_dir):
                os.remove(os.path.join(local_ckpt_dir, fn))
        if os.path.exists(metadata_dir):
            for fn in os.listdir(metadata_dir):
                os.remove(os.path.join(metadata_dir, fn))

    # Traverse block
    for i, (z1, z2, y1, y2, x1, x2) in enumerate(tqdm(blocks, desc="Local Blocks")):
        out_path = f"{output_local_base}_{i}"
        on_disk = out_path.replace("file://", "")

        # Skip completed blocks
        if is_local_done(local_ckpt_dir, i):
            continue

        # Read sub-block affinity
        aff = aff_vol[x1:x2, y1:y2, z1:z2]
        aff = np.transpose(aff, (3, 2, 1, 0))   # (c, z, y, x)
        if mask_flag:
            mask = mask_vol[x1:x2, y1:y2, z1:z2]
            mask = np.transpose(mask, (3, 2, 1, 0))[0] > 0
        else:
            mask = None
        # Run segmentation based on mode
        if mode_type == "mito":
            seg_local = run_mito_block(
                aff,
                mask=mask,
                seed_threshold=mito_cfg.get("seed_threshold", 0.98),
                foreground_threshold=mito_cfg.get("foreground_threshold", 0.85),
                min_segment_size=mito_cfg.get("min_segment_size", 128),
                seed_min_size=mito_cfg.get("seed_min_size", 32),
                remove_small_mode=mito_cfg.get("remove_small_mode", "background"),
                erosion_iters=mito_cfg.get("erosion_iters", 0)
            )
        elif mode_type == "bouton":
            neuron_ref = read_neuron_ref_block(neuron_ref_vol, (z1, z2, y1, y2, x1, x2), aff_vol.resolution)
            seg_local = run_bouton_block(
                aff,
                neuron_ref,
                mask=mask,
                seed_threshold=bouton_cfg.get("seed_threshold", 0.98),
                foreground_threshold=bouton_cfg.get("foreground_threshold", 0.85),
                min_segment_size=bouton_cfg.get("min_segment_size", 128),
                seed_min_size=bouton_cfg.get("seed_min_size", 32),
                remove_small_mode=bouton_cfg.get("remove_small_mode", "background"),
                erosion_iters=bouton_cfg.get("erosion_iters", 0),
                neuron_aff_threshold=bouton_cfg.get("neuron_aff_threshold", 0.5),
                neuron_aff_reduce=bouton_cfg.get("neuron_aff_reduce", "mean"),
                dilation_iters=bouton_cfg.get("dilation_iters", 0),
            )
        else:  # neuron mode (waterz)
            seg_local = run_waterz_block(aff, mask=mask,
                                         seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                                         sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                                         sv_2d=sv_2d, merge_function=merge_function)
        seg_xyz = np.transpose(seg_local, (2, 1, 0))

        # Write CloudVolume
        vol_size_block = (x2 - x1, y2 - y1, z2 - z1)
        seg_info = CloudVolume.create_new_info(
            num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
            resolution=aff_vol.resolution, voxel_offset=[int(x1), int(y1), int(z1)],
            volume_size=list(map(int, vol_size_block)), chunk_size=aff_vol.chunk_size,
        )
        out_local = CloudVolume(out_path, info=seg_info, compress=False,
                                progress=False, non_aligned_writes=True)
        out_local.commit_info()
        out_local.commit_provenance()
        out_local[:, :, :] = seg_xyz[:, :, :, np.newaxis]

        # Mark checkpoint
        mark_local_done(local_ckpt_dir, i)

        # Save metadata
        block_meta = {
            "index": i,
            "coords": [z1, z2, y1, y2, x1, x2],
            "path": out_path,
            "done": True,
            "max_id": int(seg_local.max())
        }
        save_block_meta(metadata_dir, block_meta)

        print(f"[INFO] Finished block {i}, max_id={block_meta['max_id']}, saved at {out_path}")

    print("[DONE] Local stage finished.")


def _process_block(
    i: int,
    coords: tuple,
    *,
    input_path: str,
    mask_flag: bool,
    mask_path: str,
    output_local_base: str,
    mip: int,
    stage_cfg,
    mode_cfg=None,
) -> dict:
    """Process a single block in an independent process; return block_meta (without writing to metadata/index.json)"""
    (z1, z2, y1, y2, x1, x2) = coords
    out_path = f"{output_local_base}_{i}"

    # Open input volume (in-process isolated instance to prevent handle sharing)
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    aff = aff_vol[x1:x2, y1:y2, z1:z2]
    aff = np.transpose(aff, (3, 2, 1, 0))  # (c, z, y, x)
    
    # Mode configuration
    if mode_cfg is None:
        mode_cfg = {}
    mode_type = mode_cfg.get("type", "neuron")
    mito_cfg = mode_cfg.get("mito", {})
    bouton_cfg = mode_cfg.get("bouton", {})

    # Neuron mode (waterz) parameters
    thresholds     = stage_cfg.get("thresholds", [0.4])
    aff_thresholds = stage_cfg.get("aff_thresholds", [0.00001, 0.99999])
    sv_type        = stage_cfg.get("sv_type", "3d")
    interior_thr   = stage_cfg.get("interior_thr", 0.1)
    min_distance   = stage_cfg.get("min_distance", 3)
    sv_2d          = stage_cfg.get("sv_2d", 'maxima_distance')
    merge_function = stage_cfg.get("merge_function", 'aff50_his256' )

    # Optional: mask
    mask = None
    if mask_flag:
        mask_vol = CloudVolume(mask_path, mip=mip, bounded=False, progress=False)
        mask = mask_vol[x1:x2, y1:y2, z1:z2]
        mask = np.transpose(mask, (3, 2, 1, 0))[0] > 0

    # Segmentation based on mode
    if mode_type == "mito":
        seg_local = run_mito_block(
            aff,
            mask=mask,
            seed_threshold=mito_cfg.get("seed_threshold", 0.98),
            foreground_threshold=mito_cfg.get("foreground_threshold", 0.85),
            min_segment_size=mito_cfg.get("min_segment_size", 128),
            seed_min_size=mito_cfg.get("seed_min_size", 32),
            remove_small_mode=mito_cfg.get("remove_small_mode", "background"),
            erosion_iters=mito_cfg.get("erosion_iters", 0)
        )
    elif mode_type == "bouton":
        # Neuron ref opened at its own finest mip; resolution mismatch handled below.
        neuron_ref_vol = CloudVolume(bouton_cfg["neuron_ref_path"], mip=0,
                                     bounded=False, fill_missing=True, progress=False)
        neuron_ref = read_neuron_ref_block(neuron_ref_vol, (z1, z2, y1, y2, x1, x2), aff_vol.resolution)
        seg_local = run_bouton_block(
            aff,
            neuron_ref,
            mask=mask,
            seed_threshold=bouton_cfg.get("seed_threshold", 0.98),
            foreground_threshold=bouton_cfg.get("foreground_threshold", 0.85),
            min_segment_size=bouton_cfg.get("min_segment_size", 128),
            seed_min_size=bouton_cfg.get("seed_min_size", 32),
            remove_small_mode=bouton_cfg.get("remove_small_mode", "background"),
            erosion_iters=bouton_cfg.get("erosion_iters", 0),
            neuron_aff_threshold=bouton_cfg.get("neuron_aff_threshold", 0.5),
            neuron_aff_reduce=bouton_cfg.get("neuron_aff_reduce", "mean"),
            dilation_iters=bouton_cfg.get("dilation_iters", 0),
        )
    else:  # neuron mode (waterz)
        seg_local = run_waterz_block(aff, mask=mask, seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                                        sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                                        sv_2d=sv_2d, merge_function=merge_function)
    seg_xyz = np.transpose(seg_local, (2, 1, 0))  # (x,y,z)

    # Write to this CloudVolume block
    vol_size_block = (x2 - x1, y2 - y1, z2 - z1)
    seg_info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint32",
        encoding="raw",
        resolution=aff_vol.resolution,
        voxel_offset=[int(x1), int(y1), int(z1)],
        volume_size=list(map(int, vol_size_block)),
        chunk_size=aff_vol.chunk_size,
    )
    out_local = CloudVolume(
        out_path, info=seg_info, compress=False, progress=False, non_aligned_writes=True
    )
    out_local.commit_info()
    out_local.commit_provenance()
    out_local[:, :, :] = seg_xyz[:, :, :, np.newaxis]

    max_id = int(seg_local.max())
    del aff, seg_local, seg_xyz
    gc.collect()

    # Return metadata (written uniformly by the main process to metadata & checkpoint to avoid concurrent contention)
    return {
        "index": i,
        "coords": [z1, z2, y1, y2, x1, x2],
        "path": out_path,
        "done": True,
        "max_id": max_id,
    }


def segmentation_blocks_parallel(global_cfg, stage_cfg, restart=False):
    """
    Parallel execution of local stage:
    - Partition large-volume affinity into chunks
    - Run `run_waterz_block` in parallel for each chunk
    - Output per-block CloudVolume
    - Write metadata and checkpoint via master process (to avoid contention for concurrent writes to `index.json`)
    """
    input_path = global_cfg["paths"]["input"]
    mask_flag = global_cfg["mask"]["flag"]
    mask_path = global_cfg["mask"]["path"]
    output_local_base = global_cfg["paths"]["output_local_base"]

    block_size = tuple(global_cfg["block"]["size"])
    overlap = tuple(global_cfg["block"]["overlap"])

    local_ckpt_dir = global_cfg["checkpoint"]["segmentation_dir"]
    metadata_dir = stage_cfg.get("metadata_dir", "./local_metadata")
    mip = stage_cfg.get("mip", 0)
    workers = int(stage_cfg.get("workers", os.cpu_count() or 1))

    # Mode configuration
    mode_cfg = global_cfg.get("mode", {})
    mode_type = mode_cfg.get("type", "neuron")
    print(f"[INFO] Segmentation mode: {mode_type}")

    # Open input volume (main process used only for retrieving shape/meta information)
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)

    # Generated in blocks (single source of truth: ROI-aware, deterministic)
    blocks = build_block_grid(aff_vol, global_cfg)

    # Restart
    if restart:
        print(f"[INFO] Restart mode: clearing local checkpoints and metadata at {local_ckpt_dir}, {metadata_dir}")
        if os.path.exists(local_ckpt_dir):
            for fn in os.listdir(local_ckpt_dir):
                os.remove(os.path.join(local_ckpt_dir, fn))
        if os.path.exists(metadata_dir):
            for fn in os.listdir(metadata_dir):
                os.remove(os.path.join(metadata_dir, fn))
                
    #  Filter out completed blocks
    tasks = []
    for i, coords in enumerate(blocks):
        if is_local_done(local_ckpt_dir, i):
            continue
        tasks.append((i, coords))

    if not tasks:
        print("[INFO] No pending blocks. Local stage up-to-date.")
        return

    print(f"[INFO] Dispatching {len(tasks)} blocks with {workers} workers...")

    # Parallel processing
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, coords in tasks:
            futures.append(
                ex.submit(
                    _process_block,
                    i,
                    coords,
                    input_path=input_path,
                    mask_flag=mask_flag,
                    mask_path=mask_path,
                    output_local_base=output_local_base,
                    mip=mip,
                    stage_cfg=stage_cfg,
                    mode_cfg=mode_cfg
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Local Blocks (parallel)"):
            try:
                block_meta = fut.result()  # If a single block encounters an exception, it will be thrown here to facilitate troubleshooting.
                # Write metadata and checkpoints sequentially to avoid concurrent write contention on index.json.
                save_block_meta(metadata_dir, block_meta)
                mark_local_done(local_ckpt_dir, block_meta["index"])
                print(
                    f"[INFO] Finished block {block_meta['index']}, "
                    f"max_id={block_meta['max_id']}, saved at {block_meta['path']}"
                )
            except KeyboardInterrupt:
                break

    print("[DONE] Local stage finished (parallel).")
