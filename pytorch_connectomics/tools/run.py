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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(args):
    """Unified CLI-compatible entrypoint with interrupt handling and config editing."""
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    if args.local_rank == 0 or args.local_rank is None:
        print(f"\nPyTorch: {torch.__version__}")
        print(f"Output directory: {cfg.DATASET.OUTPUT_PATH}")
        os.makedirs(cfg.DATASET.OUTPUT_PATH, exist_ok=True)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    mode = "test" if args.inference else "train"

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
