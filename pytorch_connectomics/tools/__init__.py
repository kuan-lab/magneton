# -*- coding: utf-8 -*-
"""
tools: Command-line tools and in-node sharded executors
"""

try:
    from .run import run as pytc_run
except Exception:
    pytc_run = None
try:
    from .run_hpc import run_hpc as pytc_run_hpc
except Exception:
    pytc_run_hpc = None

try:
    from .split import split_volume as split_volume
except Exception:
    split_volume = None
try:
    from .split_hpc import split_volume_hpc as split_volume_hpc
except Exception:
    split_volume_hpc = None

try:
    from .merge import merge_volume as merge_volume
except Exception:
    merge_volume = None
try:
    from .merge_hpc import merge_volume_hpc as merge_volume_hpc
except Exception:
    merge_volume_hpc = None

try:
    from .vis import launch_tensorboard as launch_tensorboard
except Exception:
    launch_tensorboard = None

__all__ = ["pytc_run", "pytc_run_hpc", 
           "split_volume", "split_volume_hpc",
           "merge_volume", "merge_volume_hpc",
           "launch_tensorboard",]
