# -*- coding: utf-8 -*-
"""
tools: Command-line tools and in-node sharded executors
"""
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
    from .convert_prec import main as convert_prec_main
except Exception:
    convert_prec_main = None
try:
    from .convert_prec import convert_prec as convert_prec_tool
except Exception:
    convert_prec_tool = None
try:
    from .convert_prec_hpc import convert_prec_hpc as convert_prec_tool_hpc
except Exception:
    convert_prec_tool_hpc = None

try:
    from .downsample_prec import main as downsample_prec_main
except Exception:
    downsample_prec_main = None
try:
    from .downsample_prec import downsample_prec as downsample_prec_tool
except Exception:
    downsample_prec_tool = None
try:
    from .downsample_prec_hpc import downsample_prec_hpc as downsample_prec_tool_hpc
except Exception:
    downsample_prec_tool_hpc = None

try:
    from .mask_prec import main as mask_prec_main
except Exception:
    mask_prec_main = None
try:
    from .mask_prec import mask_prec as mask_prec_tool
except Exception:
    mask_prec_tool = None
try:
    from .mask_prec_hpc import mask_prec_hpc as mask_prec_tool_hpc
except Exception:
    mask_prec_tool_hpc = None

try:
    from .mask_tif import main as mask_tif_main
except Exception:
    mask_tif_main = None
try:
    from .mask_tif import mask_tif as mask_tif_tool
except Exception:
    mask_tif_tool = None
try:
    from .mask_tif_hpc import mask_tif_hpc as mask_tif_tool_hpc
except Exception:
    mask_tif_tool_hpc = None

try:
    from .gen_mask import main as gen_mask_main
except Exception:
    gen_mask_main = None
try:
    from .gen_mask import gen_aff_mask as gen_aff_mask_tool
except Exception:
    gen_aff_mask_tool = None
try:
    from .gen_mask_hpc import gen_aff_mask_hpc as gen_aff_mask_tool_hpc
except Exception:
    gen_aff_mask_tool_hpc = None

try:
    from .resize_tif import main as resize_tif_main
except Exception:
    gen_mask_main = None
try:
    from .resize_tif import resize_tif as resize_tif_tool
except Exception:
    resize_tif_tool = None
try:
    from .resize_tif_hpc import resize_tif_hpc as resize_tif_tool_hpc
except Exception:
    resize_tif_tool_hpc = None


__all__ = ["split_volume", "split_volume_hpc",
           "merge_volume", "merge_volume_hpc", 
           "convert_prec_main", "convert_prec_tool", "convert_prec_tool_hpc", 
           "downsample_prec_main", "downsample_prec_tool", "downsample_prec_tool_hpc", 
           "mask_prec_main", "mask_prec_tool", "mask_prec_tool_hpc", 
           "mask_tif_main", "mask_tif_tool", "mask_tif_tool_hpc", 
           "gen_mask_main", "gen_aff_mask_tool", "gen_aff_mask_tool_hpc",
           "resize_tif_main", "resize_tif_tool", "resize_tif_tool_hpc", ]
