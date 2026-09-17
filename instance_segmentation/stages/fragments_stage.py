# -*- coding: utf-8 -*-
"""
Pass 1 of the graph pipeline: watershed fragments on NON-overlapping cores.

Each core is written exactly once into a single fragments volume, with ids made
globally unique arithmetically (block_index * voxels_per_core). No block ever
labels a voxel owned by another block, so there is nothing to reconcile later --
cross-block connectivity comes from the region graph built in pass 2.

Contrast with segmentation_stage.py (legacy): that writes each 768^3 block --
halo included -- as its own precomputed volume, so every overlap voxel carries
2-8 competing labels that merge_pools then has to vote on.
"""
import gc
import os

import numpy as np
from tqdm import tqdm
from cloudvolume import CloudVolume
from concurrent.futures import ProcessPoolExecutor, as_completed

from magneton.instance_segmentation.waterz_block import run_watershed_fragments
from magneton.instance_segmentation.utils.block_utils import (
    build_core_grid,
    core_slices_in_read,
    assert_cores_tile,
)
from magneton.instance_segmentation.utils.graph_utils import fragment_id_bump, bump_ids
from magneton.instance_segmentation.state.checkpoint import mark_local_done, is_local_done
from magneton.instance_segmentation.utils.meta_utils import save_block_meta


def volume_geometry(aff_vol):
    """(shape_zyx, chunk_zyx, resolution) of an opened CloudVolume."""
    size_xyz = aff_vol.info["scales"][0]["size"]
    chunk_xyz = tuple(aff_vol.chunk_size)
    return ((size_xyz[2], size_xyz[1], size_xyz[0]),
            (chunk_xyz[2], chunk_xyz[1], chunk_xyz[0]),
            aff_vol.resolution)


def build_grid(cfg, aff_vol):
    """The core grid for this config (single source of truth for block indices).

    block.roi = [z1, z2, y1, y2, x1, x2] tiles only that sub-volume (for tests);
    its origin must be chunk-aligned or cores would straddle chunk files.
    """
    blk = cfg.get("block", {})
    core = tuple(blk.get("core", [512, 512, 512]))
    halo = tuple(blk.get("halo", [128, 128, 128]))
    shape_zyx, chunk_zyx, _ = volume_geometry(aff_vol)
    roi = blk.get("roi", None)
    origin = (0, 0, 0)
    if roi is not None:
        z1, z2, y1, y2, x1, x2 = [int(v) for v in roi]
        bad = [(n, o, c) for n, o, c in zip("zyx", (z1, y1, x1), chunk_zyx) if o % c]
        if bad:
            raise ValueError(f"block.roi origin must be chunk-aligned; offending: {bad}")
        origin = (z1, y1, x1)
        shape_zyx = (min(z2, shape_zyx[0]) - z1,
                     min(y2, shape_zyx[1]) - y1,
                     min(x2, shape_zyx[2]) - x1)
    blocks = build_core_grid(shape_zyx, core, halo, origin_zyx=origin,
                             chunk_zyx=chunk_zyx)
    assert_cores_tile(blocks, shape_zyx)
    return blocks, core


def ensure_fragments_volume(aff_vol, fragments_path):
    """Create the (empty) global fragments volume; idempotent across workers."""
    size_xyz = list(map(int, aff_vol.info["scales"][0]["size"]))
    info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint64",
        encoding="raw", resolution=list(map(int, aff_vol.resolution)),
        voxel_offset=list(map(int, aff_vol.voxel_offset)),
        volume_size=size_xyz, chunk_size=list(map(int, aff_vol.chunk_size)),
    )
    vol = CloudVolume(fragments_path, info=info, compress=False, progress=False)
    vol.commit_info()
    vol.commit_provenance()
    return vol


def process_core(index, block, core_size, *, input_path, fragments_path, mip,
                 stage_cfg, mask_flag=False, mask_path="", ckpt_dir=None,
                 metadata_dir=None):
    """Watershed one core's read region, crop to the core, bump ids, write."""
    z1, z2, y1, y2, x1, x2 = block["read"]
    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False,
                          fill_missing=True)
    aff = np.asarray(aff_vol[x1:x2, y1:y2, z1:z2])
    aff = np.transpose(aff, (3, 2, 1, 0))                 # (c, z, y, x)

    mask = None
    if mask_flag and mask_path:
        mvol = CloudVolume(mask_path, mip=mip, bounded=False, progress=False,
                           fill_missing=True)
        mask = np.transpose(np.asarray(mvol[x1:x2, y1:y2, z1:z2]), (3, 2, 1, 0))[0] > 0

    frag = run_watershed_fragments(
        aff, mask=mask,
        sv_type=stage_cfg.get("supervoxel", "3d"),
        interior_thr=stage_cfg.get("interior_threshold", 0.1),
        min_distance=stage_cfg.get("min_distance", 3),
        sv_2d=stage_cfg.get("method", "maxima_distance"),
    )
    del aff
    core = frag[core_slices_in_read(block["core"], block["read"])]

    # renumber within the core so local ids stay <= voxels in a core, which is
    # what makes the arithmetic id bump collision-free
    ids, inv = np.unique(core, return_inverse=True)
    local = inv.reshape(core.shape).astype(np.uint64)
    if ids[0] != 0:                                       # no background present
        local += np.uint64(1)
    n_local = int(local.max())

    bump = fragment_id_bump(index, core_size)              # NOMINAL core size
    if n_local > int(np.prod(np.asarray(core_size, dtype=np.uint64))):
        raise RuntimeError(f"block {index}: {n_local} fragments exceeds id capacity")
    out = bump_ids(local, bump)

    cz1, cz2, cy1, cy2, cx1, cx2 = block["core"]
    fvol = CloudVolume(fragments_path, mip=0, progress=False, compress=False)
    fvol[cx1:cx2, cy1:cy2, cz1:cz2] = np.transpose(out, (2, 1, 0))[..., np.newaxis]

    meta = {"index": int(index), "core": list(map(int, block["core"])),
            "read": list(map(int, block["read"])), "bump": int(bump),
            "n_fragments": n_local, "done": True}
    if metadata_dir:
        save_block_meta(metadata_dir, meta)
    if ckpt_dir:
        mark_local_done(ckpt_dir, index)
    del frag, core, local, out
    gc.collect()
    return meta


def _stage_paths(global_cfg, stage_cfg):
    return dict(
        input_path=global_cfg["paths"]["input"],
        fragments_path=global_cfg["paths"]["fragments"],
        mip=stage_cfg.get("mip", 0),
        mask_flag=global_cfg.get("mask", {}).get("flag", False),
        mask_path=global_cfg.get("mask", {}).get("path", ""),
        ckpt_dir=global_cfg["checkpoint"]["fragments_dir"],
        metadata_dir=global_cfg["checkpoint"]["fragments_dir"],
    )


def fragments_blocks(global_cfg, stage_cfg, restart=False, indices=None, workers=1):
    """Run pass 1 locally (serial or ProcessPool)."""
    paths = _stage_paths(global_cfg, stage_cfg)
    aff_vol = CloudVolume(paths["input_path"], mip=paths["mip"], bounded=False,
                          progress=False)
    blocks, core_size = build_grid(global_cfg, aff_vol)
    ensure_fragments_volume(aff_vol, paths["fragments_path"])
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    todo = list(range(len(blocks))) if indices is None else [int(i) for i in indices]
    if not restart:
        todo = [i for i in todo if not is_local_done(paths["ckpt_dir"], i)]
    if not todo:
        print("[INFO] fragments: nothing to do (all cores done).")
        return
    print(f"[INFO] fragments: {len(todo)}/{len(blocks)} cores to process, "
          f"core={core_size}, workers={workers}")

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_core, i, blocks[i], core_size,
                              stage_cfg=stage_cfg, **paths) for i in todo]
            for fut in as_completed(futs):
                m = fut.result()
                print(f"[INFO] core {m['index']} done, {m['n_fragments']} fragments")
    else:
        for i in tqdm(todo, desc="fragments"):
            m = process_core(i, blocks[i], core_size, stage_cfg=stage_cfg, **paths)
            print(f"[INFO] core {m['index']} done, {m['n_fragments']} fragments")
    print("[DONE] fragments stage finished.")
