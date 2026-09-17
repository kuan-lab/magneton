# -*- coding: utf-8 -*-
"""
Write a Neuroglancer segment-property list for a precomputed segmentation.

Why this is a pipeline step and not a nicety: without a property list a
segmentation layer opens with an empty visible-segment set, so Neuroglancer
renders nothing until you double-click a segment in the EM. With the list, the
layer knows every id up front, and each segment carries a label (voxel count)
and tags you can filter on (#tiny, #small, #debris, ...).

Scanning at a coarser mip is much cheaper and lists every segment that survives
downsampling; use mip 0 when small objects matter.
"""
import json
import os

import numpy as np
from cloudvolume import CloudVolume

SIZE_TAGS = (("tiny", 0, 1_000), ("small", 1_000, 10_000),
             ("medium", 10_000, 100_000), ("large", 100_000, None))


def _slabs(n, step):
    for s in range(0, n, step):
        yield s, min(s + step, n)


def write_segment_properties(src, counts, maxaff=None, debris_thr=76.0,
                             dir_name="segment_properties", mip=0):
    """
    Write <src>/<dir_name>/info from {segment_id: voxel_count} and point the
    volume's info at it. Shared by the standalone tool and the relabel stage.
    """
    if not counts:
        print("[WARN] no segments; segment properties not written.")
        return None
    tag_names = [t[0] for t in SIZE_TAGS] + (["debris"] if maxaff else [])
    ids_sorted = sorted(counts)
    labels, tags = [], []
    for i in ids_sorted:
        n = counts[i]
        t = [k for k, (name, lo, hi) in enumerate(SIZE_TAGS)
             if n >= lo and (hi is None or n < hi)]
        lab = f"{n:,} vox"
        if maxaff and maxaff.get(i, 0.0) < debris_thr:
            t.append(tag_names.index("debris"))
            lab += " (debris)"
        labels.append(lab)
        tags.append(t)
    sp = {"@type": "neuroglancer_segment_properties",
          "inline": {"ids": [str(i) for i in ids_sorted],
                     "properties": [
                         {"id": "label", "type": "label", "values": labels},
                         {"id": "tags", "type": "tags", "tags": tag_names, "values": tags}]}}
    out_dir = os.path.join(src.replace("file://", ""), dir_name)
    os.makedirs(out_dir, exist_ok=True)
    # On HPC the last few array tasks can reach this concurrently. The content is
    # identical, so write via a unique temp file + atomic replace: a reader (or
    # the other task) always sees a complete file, never a truncated one.
    tmp = os.path.join(out_dir, f".info.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(sp, f)
    os.replace(tmp, os.path.join(out_dir, "info"))
    vol = CloudVolume(src, mip=mip, progress=False)
    info = vol.info
    if info.get("segment_properties") != dir_name:
        info["segment_properties"] = dir_name
        CloudVolume(src, mip=mip, info=info, progress=False).commit_info()
    print(f"[DONE] segment properties: {len(ids_sorted)} segments -> {out_dir}")
    return out_dir


def segment_props(cfg):
    c = cfg["segment_props"]
    src = c["source_path"]
    mip = int(c.get("mip", 1))
    step = int(c.get("z_step", 128))
    aff_path = c.get("affinity_path") or None
    debris_thr = float(c.get("debris_threshold", 76))    # mean affinity, 0-255
    out_name = c.get("dir_name", "segment_properties")

    vol = CloudVolume(src, mip=mip, progress=False, fill_missing=True)
    sx, sy, sz = vol.shape[:3]
    ox, oy, oz = vol.voxel_offset
    print(f"[INFO] scanning {src} at mip {mip} ({sx}x{sy}x{sz})")

    counts = {}
    maxaff = {}
    aff = CloudVolume(aff_path, mip=mip, progress=False, fill_missing=True) if aff_path else None
    for z0, z1 in _slabs(sz, step):
        sub = np.asarray(vol[ox:ox + sx, oy:oy + sy, oz + z0:oz + z1])[..., 0]
        ids, cnt = np.unique(sub, return_counts=True)
        for i, n in zip(ids.tolist(), cnt.tolist()):
            if i:
                counts[i] = counts.get(i, 0) + n
        if aff is not None:
            a = np.asarray(aff[ox:ox + sx, oy:oy + sy, oz + z0:oz + z1]).astype(np.float32).mean(-1)
            flat, af = sub.ravel(), a.ravel()
            order = np.unique(flat)
            idx = np.searchsorted(order, flat)
            mx = np.zeros(len(order), np.float32)
            np.maximum.at(mx, idx, af)
            for i, m in zip(order.tolist(), mx.tolist()):
                if i:
                    maxaff[i] = max(maxaff.get(i, 0.0), m)
        print(f"[INFO]   z {z0}-{z1}: {len(counts)} ids so far", flush=True)

    if not counts:
        print("[WARN] no segments found; nothing written.")
        return None

    return write_segment_properties(src, counts, maxaff if aff is not None else None,
                                    debris_thr, out_name, mip)
