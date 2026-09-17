# -*- coding: utf-8 -*-
"""
Pass 5: merge the per-core segment counts into one Neuroglancer property list.

Pass 4 writes counts_XXXX.npz per core (each array task only touches its own
block). This stage reads them all, sums the sizes, and writes
<output>/segment_properties/info plus the pointer in the volume's info.

Without a property list a segmentation layer opens with an empty visible-segment
set, so Neuroglancer renders nothing until a segment is picked in the EM. The
list also carries size tags (#tiny/#small/#medium/#large) and, optionally,
#debris for segments that never contain a confident-interior voxel -- the
membrane litter that a low fragments_stage.interior_threshold produces.
"""
import glob
import os

import numpy as np
from cloudvolume import CloudVolume

from magneton.instance_segmentation.stages.fragments_stage import build_grid
from magneton.toolkit.tools.segment_props import write_segment_properties
from magneton.instance_segmentation.state.checkpoint import is_local_done


def _debris_max_affinity(global_cfg, stage_cfg, ids_needed):
    """Max interior affinity per segment, for the #debris tag (optional, slower)."""
    out_path = global_cfg["paths"]["output"]
    aff_path = global_cfg["paths"]["input"]
    step = int(stage_cfg.get("z_step", 128))
    seg = CloudVolume(out_path, mip=0, progress=False, fill_missing=True)
    aff = CloudVolume(aff_path, mip=0, progress=False, fill_missing=True)
    sx, sy, sz = seg.shape[:3]
    ox, oy, oz = seg.voxel_offset
    mx = {}
    for z0 in range(0, sz, step):
        z1 = min(z0 + step, sz)
        s = np.asarray(seg[ox:ox + sx, oy:oy + sy, oz + z0:oz + z1])[..., 0]
        a = np.asarray(aff[ox:ox + sx, oy:oy + sy, oz + z0:oz + z1]).astype(np.float32).mean(-1)
        order = np.unique(s)
        idx = np.searchsorted(order, s.ravel())
        m = np.zeros(len(order), np.float32)
        np.maximum.at(m, idx, a.ravel())
        for i, v in zip(order.tolist(), m.tolist()):
            if i:
                mx[i] = max(mx.get(i, 0.0), v)
        print(f"[INFO] debris scan z {z0}-{z1}", flush=True)
    return mx


def segment_properties(global_cfg, stage_cfg):
    out_path = global_cfg["paths"]["output"]
    ckpt = global_cfg["checkpoint"].get("relabel_dir")
    if not ckpt or not os.path.isdir(ckpt):
        raise FileNotFoundError(f"no relabel checkpoint dir ({ckpt}); run pass 4 first")

    aff_vol = CloudVolume(global_cfg["paths"]["input"], mip=0, bounded=False, progress=False)
    blocks, _ = build_grid(global_cfg, aff_vol)
    missing = [i for i in range(len(blocks)) if not is_local_done(ckpt, i)]
    if missing and stage_cfg.get("require_all_cores", True):
        raise RuntimeError(
            f"pass 4 incomplete: {len(missing)} cores missing (e.g. {missing[:5]}); "
            "the property list would be partial. "
            "Set segment_props_stage.require_all_cores=false to override.")

    files = sorted(glob.glob(os.path.join(ckpt, "counts_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"no counts_*.npz in {ckpt}; re-run pass 4 (--restart) to regenerate them")
    counts = {}
    for f in files:
        d = np.load(f)
        for i, n in zip(d["ids"].tolist(), d["counts"].tolist()):
            counts[i] = counts.get(i, 0) + n
    print(f"[INFO] merged {len(files)} core files -> {len(counts)} segments")

    maxaff = None
    if stage_cfg.get("debris_tag", False):
        print("[INFO] scanning affinity for the #debris tag (this reads the volume)")
        maxaff = _debris_max_affinity(global_cfg, stage_cfg, counts)

    return write_segment_properties(
        out_path, counts, maxaff,
        debris_thr=float(stage_cfg.get("debris_threshold", 76)),
        dir_name=stage_cfg.get("dir_name", "segment_properties"))
