import os
import gc
import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume
from concurrent.futures import ProcessPoolExecutor, as_completed

from magneton.instance_segmentation.waterz_block import run_waterz_block
from magneton.instance_segmentation.mito_block import run_mito_block
from magneton.instance_segmentation.bouton_block import run_bouton_block, read_neuron_ref_block
from magneton.instance_segmentation.utils.block_utils import build_block_grid, compute_core_region
from magneton.instance_segmentation.state.checkpoint import mark_local_done, is_local_done
from magneton.instance_segmentation.utils.meta_utils import save_block_meta
from magneton.proofreading.lib.agglomerate_io import (
    compute_block_sv_partial,
    write_block_sv_partial,
)


def _write_block_seg(out_path, seg_xyz, resolution, offset_xyz, size_xyz, chunk_size):
    """Write a single (x,y,z) uint32 label block as its own precomputed volume.
    Shared by the agglomerated-seg write and the optional supervoxel write."""
    info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint32", encoding="raw",
        resolution=resolution, voxel_offset=list(map(int, offset_xyz)),
        volume_size=list(map(int, size_xyz)), chunk_size=chunk_size,
    )
    vol = CloudVolume(out_path, info=info, compress=False, progress=False,
                      non_aligned_writes=True)
    vol.commit_info()
    vol.commit_provenance()
    vol[:, :, :] = seg_xyz[:, :, :, np.newaxis]


def _supervox_base(output_local_base, stage_cfg):
    """Per-block supervoxel volume base path (parallel to the agglomerated seg
    blocks). Written only when emit_supervoxels is on — feeds merge-supervox."""
    return stage_cfg.get("supervox_output_local_base", output_local_base + "_sv")


def _emit_sv_partial(supervox_zyx, aff_czyx, coords, overlap, aff_vol, aff_order, out_npz):
    """Compute + persist this block's within-core supervoxel RAG partial.

    The block already holds the fragments (supervox_zyx) and the affinity
    (aff_czyx) right after watershed, so the RAG is computed HERE instead of being
    re-derived from the big affinity volume at merge. Uses the SAME core region as
    merge_apply/merge-supervox so ids/positions align with the written global volume.
    """
    z1, z2, y1, y2, x1, x2 = coords
    vsz = aff_vol.info["scales"][0]["size"]                 # (X,Y,Z)
    vol_shape_zyx = (vsz[2], vsz[1], vsz[0])
    ch = tuple(aff_vol.chunk_size)                          # (X,Y,Z)
    chunk_zyx = (ch[2], ch[1], ch[0])
    cz1, cz2, cy1, cy2, cx1, cx2 = compute_core_region(
        (z1, z2, y1, y2, x1, x2), tuple(overlap), vol_shape_zyx, chunk_zyx)
    core_local = (cz1 - z1, cz2 - z1, cy1 - y1, cy2 - y1, cx1 - x1, cx2 - x1)
    partial = compute_block_sv_partial(supervox_zyx, aff_czyx, core_local,
                                       (x1, y1, z1), aff_order)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    write_block_sv_partial(out_npz, partial)
    return out_npz


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
    metadata_dir   = stage_cfg.get("metadata_dir", "./metadata/local_metadata")
    mip            = stage_cfg.get("mip", 0)

    # Mode configuration (neuron vs mito)
    mode_cfg = global_cfg.get("mode", {})
    mode_type = mode_cfg.get("type", "neuron")
    mito_cfg = mode_cfg.get("mito", {})
    bouton_cfg = mode_cfg.get("bouton", {})

    # Neuron mode (waterz) parameters
    thresholds     = stage_cfg.get("thresholds", [0.4])
    aff_thresholds = stage_cfg.get("aff_thresholds", [0.00001, 0.99999])
    # Config uses documented keys (supervoxel/interior_threshold/method); fall
    # back to the old internal names for backward compat. Without this, the
    # documented keys were silently ignored and the code always ran the 3d
    # default (e.g. config_30tb's `supervoxel: "2d"` never took effect).
    sv_type        = stage_cfg.get("supervoxel", stage_cfg.get("sv_type", "3d"))
    interior_thr   = stage_cfg.get("interior_threshold", stage_cfg.get("interior_thr", 0.1))
    min_distance   = stage_cfg.get("min_distance", 3)
    sv_2d          = stage_cfg.get("method", stage_cfg.get("sv_2d", 'maxima_distance'))
    merge_function = stage_cfg.get("merge_function", 'aff50_his256' )

    # Supervoxel proofreading: also emit the pre-agglomeration watershed fragments
    # per block (neuron/waterz only) for the merge-supervox stage.
    emit_supervoxels = stage_cfg.get("emit_supervoxels", False)
    if emit_supervoxels and mode_type != "neuron":
        print(f"[WARN] emit_supervoxels only supported in neuron mode, not '{mode_type}'; disabling.")
        emit_supervoxels = False
    sv_output_base = _supervox_base(output_local_base, stage_cfg)
    # RAG edge-weight orientation; must match merge_stage.aff_channel_order.
    aff_order = stage_cfg.get("aff_channel_order", "zyx")

    print(f"[INFO] Segmentation mode: {mode_type}"
          + (" (+supervoxels)" if emit_supervoxels else ""))

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
            if emit_supervoxels:
                supervox, segs = run_waterz_block(
                    aff, mask=mask, seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                    sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                    sv_2d=sv_2d, merge_function=merge_function, return_fragments=True)
                seg_local = segs[0]                 # agglomeration at thresholds[0], as before
            else:
                seg_local = run_waterz_block(aff, mask=mask,
                                             seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                                             sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                                             sv_2d=sv_2d, merge_function=merge_function)
        seg_xyz = np.transpose(seg_local, (2, 1, 0))

        # Write CloudVolume
        vol_size_block = (x2 - x1, y2 - y1, z2 - z1)
        _write_block_seg(out_path, seg_xyz, aff_vol.resolution,
                         (x1, y1, z1), vol_size_block, aff_vol.chunk_size)

        block_meta = {
            "index": i,
            "coords": [z1, z2, y1, y2, x1, x2],
            "path": out_path,
            "done": True,
            "max_id": int(seg_local.max())
        }
        # Optional: write the matching supervoxel block for supervoxel proofreading.
        if emit_supervoxels:
            sv_path = f"{sv_output_base}_{i}"
            sv_xyz = np.transpose(supervox, (2, 1, 0)).astype(np.uint32)
            _write_block_seg(sv_path, sv_xyz, aff_vol.resolution,
                             (x1, y1, z1), vol_size_block, aff_vol.chunk_size)
            block_meta["sv_path"] = sv_path
            block_meta["sv_max_id"] = int(supervox.max())
            rag_npz = os.path.join(metadata_dir, f"sv_rag_{i}.npz")
            _emit_sv_partial(supervox, aff, (z1, z2, y1, y2, x1, x2),
                             overlap, aff_vol, aff_order, rag_npz)
            block_meta["sv_rag_path"] = rag_npz

        # Mark checkpoint
        mark_local_done(local_ckpt_dir, i)
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
    overlap=(0, 0, 0),
    metadata_dir=None,
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
    # Config uses documented keys (supervoxel/interior_threshold/method); fall
    # back to the old internal names for backward compat. Without this, the
    # documented keys were silently ignored and the code always ran the 3d
    # default (e.g. config_30tb's `supervoxel: "2d"` never took effect).
    sv_type        = stage_cfg.get("supervoxel", stage_cfg.get("sv_type", "3d"))
    interior_thr   = stage_cfg.get("interior_threshold", stage_cfg.get("interior_thr", 0.1))
    min_distance   = stage_cfg.get("min_distance", 3)
    sv_2d          = stage_cfg.get("method", stage_cfg.get("sv_2d", 'maxima_distance'))
    merge_function = stage_cfg.get("merge_function", 'aff50_his256' )
    emit_supervoxels = stage_cfg.get("emit_supervoxels", False) and mode_type == "neuron"
    sv_output_base = _supervox_base(output_local_base, stage_cfg)

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
        if emit_supervoxels:
            supervox, segs = run_waterz_block(
                aff, mask=mask, seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                sv_2d=sv_2d, merge_function=merge_function, return_fragments=True)
            seg_local = segs[0]                     # agglomeration at thresholds[0], as before
        else:
            supervox = None
            seg_local = run_waterz_block(aff, mask=mask, seg_thresholds=thresholds, aff_thresholds=aff_thresholds,
                                            sv_type=sv_type, interior_thr=interior_thr, min_distance=min_distance,
                                            sv_2d=sv_2d, merge_function=merge_function)
    seg_xyz = np.transpose(seg_local, (2, 1, 0))  # (x,y,z)

    # Write to this CloudVolume block
    vol_size_block = (x2 - x1, y2 - y1, z2 - z1)
    _write_block_seg(out_path, seg_xyz, aff_vol.resolution,
                     (x1, y1, z1), vol_size_block, aff_vol.chunk_size)

    block_meta = {
        "index": i,
        "coords": [z1, z2, y1, y2, x1, x2],
        "path": out_path,
        "done": True,
        "max_id": int(seg_local.max()),
    }
    # Optional: matching supervoxel block for supervoxel proofreading.
    if emit_supervoxels:
        sv_path = f"{sv_output_base}_{i}"
        sv_xyz = np.transpose(supervox, (2, 1, 0)).astype(np.uint32)
        _write_block_seg(sv_path, sv_xyz, aff_vol.resolution,
                         (x1, y1, z1), vol_size_block, aff_vol.chunk_size)
        block_meta["sv_path"] = sv_path
        block_meta["sv_max_id"] = int(supervox.max())
        if metadata_dir:
            rag_npz = os.path.join(metadata_dir, f"sv_rag_{i}.npz")
            _emit_sv_partial(supervox, aff, coords, overlap, aff_vol,
                             stage_cfg.get("aff_channel_order", "zyx"), rag_npz)
            block_meta["sv_rag_path"] = rag_npz

    del aff, seg_local, seg_xyz, supervox
    gc.collect()

    # Return metadata (written uniformly by the main process to metadata & checkpoint to avoid concurrent contention)
    return block_meta


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
    metadata_dir = stage_cfg.get("metadata_dir", "./metadata/local_metadata")
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
                    mode_cfg=mode_cfg,
                    overlap=overlap,
                    metadata_dir=metadata_dir,
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
