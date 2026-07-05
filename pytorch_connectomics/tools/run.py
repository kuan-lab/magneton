#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connectomics training/inference module — unified CLI and interactive interface.

Provides:
- main(): command-line interface for training/inference
- run(args, global_cfg): external entrypoint
- run_interactive(): interactive menu
"""

import argparse
import logging
import os
import yaml
import signal
import time
import torch
import warnings

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer
from magneton.instance_segmentation.utils.interrupts import InterruptController
from magneton.pytorch_connectomics.inference_grid import (
    is_precomputed_url, volume_info_from_cv, block_count, block_by_id,
    task_block_range, apply_roi, precomputed_url_to_local_path,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(args):
    """Unified CLI-compatible entrypoint with interrupt handling and config editing."""
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    # If DATASET.OUTPUT_PATH was set to a precomputed URL (e.g. via the
    # interactive editor mirroring INFERENCE.OUTPUT_PATH), strip the
    # `precomputed://file://` prefix so makedirs/save_all_cfg below land
    # in the right place. CloudVolume URLs aren't filesystem paths.
    if is_precomputed_url(cfg.DATASET.OUTPUT_PATH):
        local_out = precomputed_url_to_local_path(cfg.DATASET.OUTPUT_PATH)
        cfg.defrost()
        cfg.DATASET.OUTPUT_PATH = local_out
        cfg.freeze()

    if args.local_rank == 0 or args.local_rank is None:
        print(f"\nPyTorch: {torch.__version__}")
        print(f"Output directory: {cfg.DATASET.OUTPUT_PATH}")
        os.makedirs(cfg.DATASET.OUTPUT_PATH, exist_ok=True)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    mode = "test" if args.inference else "train"

    # Direct-precomputed inference: read input region from a precomputed
    # volume, run PyTC, and write only the chunk-aligned core to an output
    # precomputed. Cores tile disjointly so concurrent SLURM tasks never
    # write to the same output chunk — no locks. Multi-block-per-task is
    # handled here too via --task-id / --chunks-per-task.
    if (args.inference
            and is_precomputed_url(cfg.INFERENCE.IMAGE_NAME)
            and is_precomputed_url(cfg.INFERENCE.OUTPUT_PATH)):
        _run_precomputed(args, cfg, device, mode)
        return

    # Multi-chunk inference: list-valued IMAGE_NAME packs N chunks into one
    # process so CUDA warmup + model load are paid once per SLURM task instead
    # of once per chunk. Mirrors the Trainer.test_singly() pattern.
    if args.inference and isinstance(cfg.INFERENCE.IMAGE_NAME, (list, tuple)):
        from connectomics.data.dataset import get_dataset, build_dataloader

        image_names = list(cfg.INFERENCE.IMAGE_NAME)
        # OUTPUT_NAME can't be a list in YACS (its default is a string), so
        # derive per-chunk output filenames from the input filenames here.
        output_names = [os.path.splitext(img)[0] + '.h5' for img in image_names]

        input_folder = cfg.DATASET.INPUT_PATH

        # Reduce cfg to the first chunk so Trainer.__init__ sees scalars.
        cfg.defrost()
        cfg.INFERENCE.IMAGE_NAME = image_names[0]
        cfg.INFERENCE.OUTPUT_NAME = output_names[0]
        cfg.DATASET.IMAGE_NAME = image_names[0]
        cfg.freeze()

        trainer = Trainer(cfg, device, mode,
                          rank=args.local_rank,
                          checkpoint=args.checkpoint)

        print(f"[MULTI-CHUNK] Processing {len(image_names)} chunks in one process")

        with InterruptController():
            for idx, (img, out) in enumerate(zip(image_names, output_names)):
                resolved_out = trainer.augmentor.update_name(out)
                out_path = os.path.join(trainer.output_dir, resolved_out)
                if os.path.exists(out_path):
                    print(f"[SKIP {idx+1}/{len(image_names)}] {resolved_out} already exists")
                    continue

                print(f"[CHUNK {idx+1}/{len(image_names)}] {img} -> {resolved_out}")
                t0 = time.perf_counter()

                # Mutate cfg rather than passing dir_name_init/img_name_init
                # kwargs: get_dataset has a positional-arg bug that sends
                # dir_name_init into _get_input's preload_data slot.
                cfg.defrost()
                cfg.DATASET.IMAGE_NAME = img
                cfg.INFERENCE.IMAGE_NAME = img
                cfg.freeze()

                dataset = get_dataset(
                    trainer.cfg, trainer.augmentor, trainer.mode, trainer.rank,
                )
                trainer.dataloader = build_dataloader(
                    trainer.cfg, trainer.augmentor, trainer.mode, dataset, trainer.rank,
                )
                trainer.dataloader = iter(trainer.dataloader)
                trainer.test_filename = resolved_out

                trainer.test()

                print(f"[CHUNK {idx+1}/{len(image_names)}] done in {time.perf_counter()-t0:.1f}s")
        return

    trainer = Trainer(cfg, device, mode,
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Run under interrupt controller
    with InterruptController():
        if cfg.DATASET.DO_CHUNK_TITLE == 0:
            test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
            test_func() if args.inference else trainer.train()
        else:
            trainer.run_chunk(mode)

def _run_precomputed(args, cfg, device, mode):
    """Direct-precomputed inference. See run() for high-level docs."""
    import gc
    import numpy as np
    from cloudvolume import CloudVolume
    from connectomics.data.dataset import get_dataset, build_dataloader
    from magneton.pytorch_connectomics.init_prec_output import init_output_volume

    input_url = cfg.INFERENCE.IMAGE_NAME
    output_url = cfg.INFERENCE.OUTPUT_PATH
    geom = cfg.INFERENCE.GEOMETRY
    core_size = int(geom.CORE_SIZE)
    halo = int(geom.HALO)
    chunk_size = int(geom.OUTPUT_CHUNK_SIZE)
    mip = int(geom.MIP)

    vol_shape_zyx, vol_offset_zyx, _ = volume_info_from_cv(input_url, mip=mip)
    roi = list(geom.ROI) if geom.ROI is not None else None
    vol_shape_zyx, vol_offset_zyx = apply_roi(
        vol_shape_zyx, vol_offset_zyx, roi, output_chunk_size=chunk_size)
    # Optionally size the output to the ROI (offset = ROI start) instead of the
    # full input extent. ZYX -> XYZ for CloudVolume. See GEOMETRY.CROP_OUTPUT_TO_ROI.
    crop_out = bool(getattr(geom, "CROP_OUTPUT_TO_ROI", False)) and roi is not None
    out_size_xyz = (vol_shape_zyx[2], vol_shape_zyx[1], vol_shape_zyx[0]) if crop_out else None
    out_offset_xyz = (vol_offset_zyx[2], vol_offset_zyx[1], vol_offset_zyx[0]) if crop_out else None
    init_output_volume(input_url, output_url, mip=mip,
                       num_channels=int(cfg.MODEL.OUT_PLANES),
                       dtype="uint8", chunk_size=chunk_size,
                       output_size_xyz=out_size_xyz, output_offset_xyz=out_offset_xyz)

    in_cv = CloudVolume(input_url, mip=mip, bounded=False, fill_missing=True,
                        progress=False)
    # Output is a freshly created single-scale precomputed (scales[0] = input's
    # `mip`), so always open it at mip=0.
    # compress=False matches init_output_volume so chunks land as raw bytes
    # (no gzip wrapping) — required for Neuroglancer to decode them per the
    # `encoding: raw` declaration in the info file.
    out_cv = CloudVolume(output_url, mip=0, bounded=False, progress=False,
                         non_aligned_writes=False, compress=False)

    total_blocks = block_count(vol_shape_zyx, core_size=core_size)
    task_id = args.task_id if args.task_id is not None else 0
    chunks_per_task = args.chunks_per_task if args.chunks_per_task is not None else total_blocks
    bid_start, bid_end = task_block_range(task_id, chunks_per_task, total_blocks)
    if bid_start >= bid_end:
        print(f"[INFO] task_id={task_id} has no blocks (total={total_blocks}).")
        return

    print(f"[PREC] Volume ZYX={vol_shape_zyx} offset={vol_offset_zyx} "
          f"total_blocks={total_blocks} this_task=[{bid_start},{bid_end}) "
          f"core={core_size} halo={halo} chunk={chunk_size}")

    # Build the trainer once. DO_SINGLY=True skips the constructor's
    # auto-built dataloader (trainer.py:78-81); we rebuild it per block.
    cfg.defrost()
    cfg.INFERENCE.DO_SINGLY = True
    cfg.freeze()
    trainer = Trainer(cfg, device, mode,
                      rank=args.local_rank, checkpoint=args.checkpoint)
    trainer.output_dir = None  # makes Trainer.test() return result instead of writeh5

    done_dir = os.path.join(cfg.DATASET.OUTPUT_PATH, "done_blocks")
    os.makedirs(done_dir, exist_ok=True)

    with InterruptController():
        for bid in range(bid_start, bid_end):
            blk = block_by_id(bid, vol_shape_zyx, vol_offset_zyx,
                              core_size=core_size, halo=halo,
                              output_chunk_size=chunk_size)
            done_path = os.path.join(done_dir, f"block_{bid:08d}.done")
            if os.path.exists(done_path):
                print(f"[SKIP {bid}] already done")
                continue

            t0 = time.perf_counter()
            rz1, rz2, ry1, ry2, rx1, rx2 = blk.read_bbox
            cz1, cz2, cy1, cy2, cx1, cx2 = blk.core_bbox
            print(f"[BLOCK {bid}/{total_blocks}] "
                  f"read=({rz1}:{rz2},{ry1}:{ry2},{rx1}:{rx2}) "
                  f"core=({cz1}:{cz2},{cy1}:{cy2},{cx1}:{cx2})")

            # CloudVolume is XYZ-indexed; transpose to ZYX after read.
            vol_xyzc = in_cv[rx1:rx2, ry1:ry2, rz1:rz2]
            vol_zyx = np.asarray(vol_xyzc)
            if vol_zyx.shape[-1] == 1:
                vol_zyx = vol_zyx[..., 0]
            vol_zyx = np.transpose(vol_zyx, (2, 1, 0))

            dataset = get_dataset(trainer.cfg, trainer.augmentor, trainer.mode,
                                  trainer.rank, preload_data=[vol_zyx])
            loader = build_dataloader(trainer.cfg, trainer.augmentor,
                                      trainer.mode, dataset, trainer.rank)
            trainer.dataloader = iter(loader)

            result = trainer.test()  # list of (C, Z, Y, X) uint8 arrays
            pred_zyx = result[0]  # shape (C, rZ, rY, rX) — read-region sized

            # Crop the read-frame prediction to core_bbox
            oz, oy, ox = cz1 - rz1, cy1 - ry1, cx1 - rx1
            sz, sy, sx = cz2 - cz1, cy2 - cy1, cx2 - cx1
            core = pred_zyx[:, oz:oz+sz, oy:oy+sy, ox:ox+sx]

            # CloudVolume expects (X, Y, Z, C)
            out_xyzc = np.transpose(core, (3, 2, 1, 0))
            out_cv[cx1:cx2, cy1:cy2, cz1:cz2, :] = out_xyzc

            with open(done_path, "w"):
                pass
            print(f"[BLOCK {bid}] done in {time.perf_counter()-t0:.1f}s")

            del vol_zyx, dataset, loader, result, pred_zyx, core, out_xyzc
            gc.collect()


# ==========================================================
# main()
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--inference', action='store_true',
                        help='inference mode')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to load the checkpoint')
    parser.add_argument('--manual-seed', type=int, default=None)
    parser.add_argument('--local_world_size', type=int, default=1,
                        help='number of GPUs each process.')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='node rank for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='run the scripts in debug mode')
    # Direct-precomputed inference: SLURM array slicing of the block grid.
    parser.add_argument('--task-id', type=int, default=None,
                        help='SLURM array task id (precomputed inference only)')
    parser.add_argument('--chunks-per-task', type=int, default=None,
                        help='blocks processed per SLURM task (precomputed inference only)')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
