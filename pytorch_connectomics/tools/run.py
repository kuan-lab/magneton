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
