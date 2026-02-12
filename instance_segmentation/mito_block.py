"""
Mitochondria segmentation using binary watershed.

This module provides segmentation for sparse objects like mitochondria,
using scikit-image's watershed with foreground masking instead of waterz.
Unlike waterz (designed for dense neuron segmentation), this approach:
- Only creates segments where probability exceeds a threshold
- Seeds are placed at high-confidence regions
- Watershed grows from seeds but is bounded by the foreground mask
- Empty space remains unsegmented
"""

import numpy as np
from scipy.ndimage import label as scipy_label, binary_erosion
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label


def remove_small_instances(segm, thres_small=128, mode='background'):
    """
    Remove small spurious instances.

    Args:
        segm: Segmentation array
        thres_small: Size threshold for removal
        mode: 'background' (set to 0), 'neighbor' (merge to neighbor), or 'none'
    """
    if mode == 'none':
        return segm
    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    # 'neighbor' mode would require more complex logic, fall back to background
    return remove_small_objects(segm, thres_small)


def cast2dtype(segm):
    """Cast segmentation to appropriate dtype based on max ID."""
    max_id = int(np.max(segm))
    if max_id < 2**8:
        return segm.astype(np.uint8)
    elif max_id < 2**16:
        return segm.astype(np.uint16)
    else:
        return segm.astype(np.uint32)


def binary_watershed(
    volume,
    thres1=0.98,
    thres2=0.85,
    thres_small=128,
    seed_thres=32,
    remove_small_mode='background',
    erosion_iters=0
):
    """
    Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    This is adapted from pytorch_connectomics for use with single-channel
    probability maps (e.g., mitochondria interior probability).

    Args:
        volume: Foreground probability of shape (C, Z, Y, X) or (Z, Y, X).
                Values should be in range [0, 255] or [0, 1].
        thres1: Threshold for seeds (high confidence). Default: 0.98
        thres2: Threshold for foreground mask. Default: 0.85
        thres_small: Size threshold for small objects to remove. Default: 128
        seed_thres: Size threshold for small seeds to remove. Default: 32
        remove_small_mode: 'background', 'neighbor', or 'none'. Default: 'background'
        erosion_iters: Number of binary erosion iterations on seed mask. Helps
            break bridges between adjacent objects to reduce merge errors. Default: 0 (no erosion)

    Returns:
        Segmentation array of shape (Z, Y, X) with instance labels.
    """
    # Handle input shape
    if volume.ndim == 4:
        semantic = volume[0]  # Take first channel
    else:
        semantic = volume

    # Normalize to [0, 255] range if needed
    if semantic.max() <= 1.0:
        semantic = (semantic * 255).astype(np.float32)

    # Create seed map from high-confidence regions
    seed_map = semantic > int(255 * thres1)

    # Erode seed map to break bridges between adjacent objects
    if erosion_iters > 0:
        seed_map = binary_erosion(seed_map, iterations=erosion_iters)

    # Create foreground mask from moderate-confidence regions
    foreground = semantic > int(255 * thres2)

    # Label connected components in seed map
    seed = label(seed_map)

    # Remove small seed regions
    if seed_thres > 0:
        seed = remove_small_objects(seed, seed_thres)

    # Run watershed: grow from seeds, bounded by foreground mask
    # Use negative semantic as the "elevation" map so watershed grows from high to low probability
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)

    # Remove small segments
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    return cast2dtype(segm)


def run_mito_block(
    aff_block_czyx,
    mask=None,
    seed_threshold=0.98,
    foreground_threshold=0.85,
    min_segment_size=128,
    seed_min_size=32,
    remove_small_mode='background',
    erosion_iters=0
):
    """
    Run mitochondria segmentation on a single block using binary watershed.

    This is the main entry point for mito mode segmentation, analogous to
    run_waterz_block for neuron mode.

    Args:
        aff_block_czyx: Input volume of shape (C, Z, Y, X). For mito segmentation,
                        typically C=1 (interior probability). Values in [0, 255] or [0, 1].
        mask: Optional boolean mask of shape (Z, Y, X). Only segment where mask is True.
        seed_threshold: Probability threshold for seed regions (default: 0.98)
        foreground_threshold: Probability threshold for foreground mask (default: 0.85)
        min_segment_size: Remove segments smaller than this (default: 128)
        seed_min_size: Remove seed regions smaller than this (default: 32)
        remove_small_mode: How to handle small segments (default: 'background')
        erosion_iters: Number of erosion iterations on seed mask to break merges (default: 0)

    Returns:
        Segmentation array of shape (Z, Y, X) with instance labels (uint32).
    """
    # Convert to float32 and normalize
    aff = aff_block_czyx.astype(np.float32)
    if aff.max() > 1.0:
        aff = aff / 255.0

    # Handle multi-channel input by taking the first channel (interior probability)
    if aff.shape[0] > 1:
        print(f"[INFO] Multi-channel input ({aff.shape[0]} channels), using first channel for mito segmentation")
        aff = aff[0:1]

    # Apply external mask if provided
    if mask is not None:
        # Zero out probability outside mask
        aff = aff * mask.astype(np.float32)

    # Run binary watershed
    segm = binary_watershed(
        aff,
        thres1=seed_threshold,
        thres2=foreground_threshold,
        thres_small=min_segment_size,
        seed_thres=seed_min_size,
        remove_small_mode=remove_small_mode,
        erosion_iters=erosion_iters
    )

    # Ensure output is uint32 for compatibility with rest of pipeline
    segm = segm.astype(np.uint32)

    print(f"[INFO] Mito segmentation complete: {segm.max()} instances found")

    return segm
