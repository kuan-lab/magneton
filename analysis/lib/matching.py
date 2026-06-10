"""
Cross-volume instance matching for the relational analysis.

The three fib_b instance volumes (mito / bouton / synapse) are co-registered on
the SAME grid (offset [0,0,0], chunk 128³, identical mip-0 shape), so matching is
a direct same-index voxel lookup — no resolution remapping.

Two match modes, one per object pair (semantics fixed by design):

- match_mito_to_bouton  — medoid-snapped centroid. For each mito, snap its centroid
  to the nearest real mito voxel (a raw centroid of a curved/elongated mito can land
  in empty space or a neighbor), then sample the bouton label there. One mito → 0 or
  1 bouton. Runs in-memory at a coarse mip (both volumes are small there).

- match_synapse_to_bouton — direct voxel overlap, best (max-contact) bouton. For each
  synapse, count bouton labels under the synapse mask and take the argmax. One synapse
  → its single best bouton (0 = touches none). Tiny per-synapse crops at mip-0.

All voxel coordinates are XYZ.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from scipy.ndimage import find_objects

from .precomputed_io import read_full, get_volume_specs


def _open(pc_path: str, mip: int = 0) -> CloudVolume:
    return CloudVolume(pc_path, mip=mip, parallel=1, progress=False, fill_missing=False)


# ----------------------------------------------- mito → bouton (centroid) -----

def match_mito_to_bouton(mito_pc: str, bouton_pc: str, mip: int = 2) -> pd.DataFrame:
    """
    Assign each mito a parent bouton by sampling the bouton label at the mito's
    medoid-snapped centroid. Both volumes are read in full at `mip` (co-registered,
    so same shape → direct index). Returns a DataFrame with one row per mito:
        mito_seg_id, parent_bouton, sample_x, sample_y, sample_z
    sample_* are mip-0 voxel coords of the sampled point (handy for Neuroglancer).
    parent_bouton == 0 ⇒ orphan (not inside any bouton).
    """
    specs = get_volume_specs(mito_pc, mip=mip)
    fX, fY, fZ = specs.downsample_factor_xyz

    print(f"[matching.mito] reading mito + bouton at mip-{mip} "
          f"(factor {fX},{fY},{fZ}) — co-registered direct lookup")
    mito_vol   = read_full(mito_pc, mip=mip)
    bouton_vol = read_full(bouton_pc, mip=mip)
    if mito_vol.shape != bouton_vol.shape:
        raise ValueError(
            f"mito and bouton volumes are not co-registered at mip-{mip}: "
            f"{mito_vol.shape} vs {bouton_vol.shape}"
        )

    slices = find_objects(mito_vol)
    rows = []
    for label_minus_1, slc in enumerate(slices):
        if slc is None:
            continue
        seg_id = label_minus_1 + 1
        sub = mito_vol[slc] == seg_id
        coords = np.argwhere(sub)            # (n,3) local XYZ
        if len(coords) == 0:
            continue
        centroid = coords.mean(axis=0)
        # medoid = the actual mito voxel nearest the centroid (guaranteed inside)
        d2 = ((coords - centroid) ** 2).sum(axis=1)
        med_local = coords[int(np.argmin(d2))]
        gx = int(med_local[0] + slc[0].start)
        gy = int(med_local[1] + slc[1].start)
        gz = int(med_local[2] + slc[2].start)
        parent = int(bouton_vol[gx, gy, gz])
        rows.append(dict(
            mito_seg_id=int(seg_id),
            parent_bouton=parent,
            sample_x=gx * fX, sample_y=gy * fY, sample_z=gz * fZ,
        ))

    df = pd.DataFrame(rows, columns=[
        "mito_seg_id", "parent_bouton", "sample_x", "sample_y", "sample_z",
    ])
    n_matched = int((df["parent_bouton"] > 0).sum()) if len(df) else 0
    print(f"[matching.mito] {len(df)} mitos, {n_matched} inside a bouton "
          f"({100.0 * n_matched / max(1, len(df)):.1f}%)")
    return df


# --------------------------------------------- synapse → bouton (overlap) -----

def match_synapse_to_bouton(synapse_pc: str, bouton_pc: str,
                            syn_bboxes: pd.DataFrame) -> pd.DataFrame:
    """
    For each synapse, count bouton labels under the synapse mask (direct overlap)
    and assign the single best (max-overlap) bouton. `syn_bboxes` is the synapse
    volume's bboxes.parquet (mip-0 bbox columns). Returns one row per synapse:
        synapse_seg_id, best_bouton, overlap_voxels, n_boutons_touched
    best_bouton == 0 ⇒ touches no bouton.
    """
    syn_cv = _open(synapse_pc, mip=0)
    bou_cv = _open(bouton_pc, mip=0)

    rows = []
    n = len(syn_bboxes)
    for i, r in enumerate(syn_bboxes.itertuples(index=False)):
        seg_id = int(r.seg_id)
        x0, x1 = int(r.bbox_x0), int(r.bbox_x1)
        y0, y1 = int(r.bbox_y0), int(r.bbox_y1)
        z0, z1 = int(r.bbox_z0), int(r.bbox_z1)
        syn_crop = np.asarray(syn_cv[x0:x1, y0:y1, z0:z1])[:, :, :, 0]
        mask = syn_crop == seg_id
        if not mask.any():
            rows.append(dict(synapse_seg_id=seg_id, best_bouton=0,
                             overlap_voxels=0, n_boutons_touched=0))
            continue
        bou_crop = np.asarray(bou_cv[x0:x1, y0:y1, z0:z1])[:, :, :, 0]
        labels = bou_crop[mask]
        bc = np.bincount(labels.reshape(-1))
        if bc.shape[0] > 0:
            bc[0] = 0                       # ignore background
        if bc.sum() == 0:
            rows.append(dict(synapse_seg_id=seg_id, best_bouton=0,
                             overlap_voxels=0, n_boutons_touched=0))
        else:
            best = int(np.argmax(bc))
            rows.append(dict(
                synapse_seg_id=seg_id,
                best_bouton=best,
                overlap_voxels=int(bc[best]),
                n_boutons_touched=int(np.count_nonzero(bc)),
            ))
        if (i + 1) % 500 == 0:
            print(f"[matching.synapse]   {i + 1}/{n} synapses")

    df = pd.DataFrame(rows, columns=[
        "synapse_seg_id", "best_bouton", "overlap_voxels", "n_boutons_touched",
    ])
    n_matched = int((df["best_bouton"] > 0).sum()) if len(df) else 0
    print(f"[matching.synapse] {len(df)} synapses, {n_matched} touch a bouton "
          f"({100.0 * n_matched / max(1, len(df)):.1f}%); "
          f"{len(df) - n_matched} untouched")
    return df
