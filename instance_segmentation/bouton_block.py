"""
Bouton segmentation: membrane-gated binary watershed.

Boutons (axonal synaptic terminals) are sparse, blob-like objects, so the core
segmentation is the same binary watershed used for mitochondria
(see `mito_block.binary_watershed`). The one extra step that distinguishes
bouton mode is *membrane gating*:

    Wherever a reference neuron affinity map has LOW affinity (cell membranes
    and extracellular space), the bouton affinity at the same coordinates is
    forced to 0 before watershed.

This breaks the many false merges where a single bouton instance would
otherwise bleed across a membrane into a neighboring axon. Optionally, the
resulting instances can be dilated a few voxels to recover the rounder bouton
shape -- but the dilation is constrained to the non-membrane region so it never
grows into or through a membrane.
"""

import numpy as np
from scipy.ndimage import grey_dilation

from magneton.instance_segmentation.mito_block import binary_watershed


def read_neuron_ref_block(neuron_vol, coords_zyx, aff_resolution):
    """
    Read the neuron reference block covering the same physical region as a bouton
    block, handling a resolution mismatch by upsampling.

    The neuron reference may be coarser than the bouton input (e.g. 8nm neuron vs
    4nm bouton). Coords are given in BOUTON voxel space; this maps them into the
    neuron's voxel grid (floor start / ceil end), reads, upsamples by the integer
    ratio, then crops to exactly match the bouton block extent.

    Args:
        neuron_vol: opened CloudVolume of the neuron affinity reference.
        coords_zyx: (z1, z2, y1, y2, x1, x2) in bouton (=input) voxel coords.
        aff_resolution: the bouton input resolution as (rx, ry, rz) in nm (xyz),
            e.g. CloudVolume.resolution.

    Returns:
        Neuron affinity block of shape (C, Z, Y, X) aligned to the bouton block.
    """
    z1, z2, y1, y2, x1, x2 = coords_zyx
    nres = [int(round(r)) for r in neuron_vol.resolution]   # xyz, e.g. [8,8,8]
    ares = [int(round(r)) for r in aff_resolution]          # xyz, e.g. [4,4,4]
    for n, a in zip(nres, ares):
        if n < a or n % a != 0:
            raise ValueError(
                f"Neuron ref resolution {nres} must be a coarser integer multiple "
                f"of the bouton input resolution {ares}")
    rx, ry, rz = nres[0] // ares[0], nres[1] // ares[1], nres[2] // ares[2]

    # Map bouton coords -> neuron voxel coords (floor start, ceil end), clipped.
    nsize = neuron_vol.info["scales"][0]["size"]  # xyz
    nx1, nx2 = x1 // rx, min(-(-x2 // rx), nsize[0])
    ny1, ny2 = y1 // ry, min(-(-y2 // ry), nsize[1])
    nz1, nz2 = z1 // rz, min(-(-z2 // rz), nsize[2])

    nref = neuron_vol[nx1:nx2, ny1:ny2, nz1:nz2]          # (x, y, z, c)
    nref = np.transpose(nref, (3, 2, 1, 0))               # (c, z, y, x)

    # Upsample to bouton resolution, then crop off the floor-rounding offset.
    nref = np.repeat(np.repeat(np.repeat(nref, rz, axis=1), ry, axis=2), rx, axis=3)
    oz, oy, ox = z1 - nz1 * rz, y1 - ny1 * ry, x1 - nx1 * rx
    nref = nref[:, oz:oz + (z2 - z1), oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
    return nref


def _reduce_neuron_aff(neuron_czyx, reduce="mean"):
    """
    Reduce a multi-channel neuron affinity block (C, Z, Y, X) to a single
    (Z, Y, X) probability map in [0, 1], where low values mark membranes / ECS.
    """
    aff = neuron_czyx.astype(np.float32)
    if aff.max() > 1.0:
        aff = aff / 255.0
    if aff.ndim == 3:
        return aff
    if reduce == "min":
        return aff.min(axis=0)
    if reduce == "first":
        return aff[0]
    return aff.mean(axis=0)  # default: mean, matches boundary_from_aff convention


def _masked_label_dilation(seg, allowed_mask, iters):
    """
    Grow labels outward by `iters` voxels, but only into voxels that are
    currently background AND inside `allowed_mask` (the non-membrane region).
    This rounds out bouton instances without crossing membranes.
    """
    seg = seg.copy()
    for _ in range(iters):
        dil = grey_dilation(seg, size=(3, 3, 3))
        fill = (seg == 0) & allowed_mask & (dil > 0)
        if not np.any(fill):
            break
        seg[fill] = dil[fill]
    return seg


def run_bouton_block(
    aff_block_czyx,
    neuron_ref_czyx,
    mask=None,
    seed_threshold=0.98,
    foreground_threshold=0.85,
    min_segment_size=128,
    seed_min_size=32,
    remove_small_mode='background',
    erosion_iters=0,
    neuron_aff_threshold=0.5,
    neuron_aff_reduce='mean',
    dilation_iters=0,
):
    """
    Run bouton segmentation on a single block: membrane-gated binary watershed.

    Args:
        aff_block_czyx: Bouton interior probability of shape (C, Z, Y, X).
            Values in [0, 255] or [0, 1].
        neuron_ref_czyx: Reference neuron affinity of shape (C, Z, Y, X) covering
            the SAME coordinates as `aff_block_czyx`. Low affinity = membrane/ECS.
        mask: Optional boolean mask (Z, Y, X). Only segment where mask is True.
        seed_threshold: Probability threshold for seed regions (default: 0.98).
        foreground_threshold: Probability threshold for the watershed foreground
            mask (default: 0.85).
        min_segment_size: Remove segments smaller than this (default: 128).
        seed_min_size: Remove seed regions smaller than this (default: 32).
        remove_small_mode: 'background', 'neighbor', or 'none' (default: 'background').
        erosion_iters: Seed-mask erosion iterations (default: 0).
        neuron_aff_threshold: Neuron affinity below this is treated as membrane/ECS;
            bouton affinity there is zeroed before watershed (default: 0.5).
        neuron_aff_reduce: How to collapse neuron channels to one map:
            'mean' (default), 'min', or 'first'.
        dilation_iters: If > 0, dilate final instances this many voxels, constrained
            to the non-membrane region (default: 0 = no dilation).

    Returns:
        Segmentation array of shape (Z, Y, X) with instance labels (uint32).
    """
    # Normalize bouton affinity to [0, 1].
    aff = aff_block_czyx.astype(np.float32)
    if aff.max() > 1.0:
        aff = aff / 255.0

    # Use the interior (first) channel for bouton segmentation.
    if aff.shape[0] > 1:
        print(f"[INFO] Multi-channel bouton input ({aff.shape[0]} channels), using first channel")
        aff = aff[0:1]

    # Membrane gating: zero bouton affinity wherever the neuron reference is
    # below threshold (membrane / extracellular space). This breaks merges.
    neuron_prob = _reduce_neuron_aff(neuron_ref_czyx, reduce=neuron_aff_reduce)
    non_membrane = neuron_prob >= neuron_aff_threshold
    n_gated = int((~non_membrane).sum())
    aff = aff * non_membrane[np.newaxis, :, :, :].astype(np.float32)
    print(f"[INFO] Membrane gating zeroed {n_gated} voxels "
          f"({100.0 * n_gated / non_membrane.size:.1f}% of block) below neuron aff {neuron_aff_threshold}")

    # Apply external mask if provided.
    if mask is not None:
        aff = aff * mask.astype(np.float32)

    # Core segmentation: identical binary watershed as mito mode.
    segm = binary_watershed(
        aff,
        thres1=seed_threshold,
        thres2=foreground_threshold,
        thres_small=min_segment_size,
        seed_thres=seed_min_size,
        remove_small_mode=remove_small_mode,
        erosion_iters=erosion_iters,
    ).astype(np.uint32)

    # Optional membrane-constrained dilation to round out boutons.
    if dilation_iters > 0:
        segm = _masked_label_dilation(segm, non_membrane, dilation_iters).astype(np.uint32)

    print(f"[INFO] Bouton segmentation complete: {segm.max()} instances found")
    return segm
