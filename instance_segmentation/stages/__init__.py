# -*- coding: utf-8 -*-
"""
Stages: set of entry points for each stage
"""
from .segmentation_stage import segmentation_blocks
from .segmentation_stage import segmentation_blocks_parallel
try:
    from .segmentation_stage_hpc import segmentation_blocks_hpc
except Exception:
    segmentation_blocks_hpc = None

from .merge_stage import merge_local_blocks

try:
    from .merge_pools import build_id_pools_parallel
    from .merge_apply import apply_pools_to_global
except Exception:
    build_id_pools_parallel = None
    apply_pools_to_global = None

__all__ = [
    "segmentation_blocks",
    "segmentation_blocks_parallel",
    "segmentation_blocks_hpc",
    "merge_local_blocks",
    "build_id_pools_parallel",
    "apply_pools_to_global",
]
