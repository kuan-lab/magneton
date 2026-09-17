# -*- coding: utf-8 -*-
"""
Pass 3: one global decision over the whole region graph.

Reads every per-core edge file, collapses duplicates, keeps edges whose score
passes the threshold and whose contact area is large enough, then takes
connected components. The result is a LUT fragment_id -> segment_id, where the
segment id is the SMALLEST fragment id in the component (so applying the LUT is
idempotent and pass 4 can be retried, or run in place).

Cheap by design: no image data is read, so a threshold sweep is minutes
(pass 3 + pass 4) instead of a re-segmentation.
"""
import os

import numpy as np

from magneton.instance_segmentation.utils.graph_utils import (
    read_edges_dir, dedupe_edges, components_from_edges, write_lut_npz,
)


def global_merge(global_cfg, stage_cfg, threshold=None, min_contact=None,
                 lut_path=None):
    edges_dir = (global_cfg.get("edges_stage", {})
                 .get("edges_dir", "magneton/metadata/edges"))
    threshold = float(stage_cfg.get("threshold", 0.45) if threshold is None else threshold)
    min_contact = int(stage_cfg.get("min_contact", 0) if min_contact is None else min_contact)
    how = stage_cfg.get("dedupe", "max")
    lut_dir = stage_cfg.get("lut_dir", "magneton/checkpoints/lut")
    if lut_path is None:
        lut_path = os.path.join(lut_dir, f"lut_t{str(threshold).replace('.', '')}"
                                         f"_c{min_contact}.npz")

    u, v, s, c = read_edges_dir(edges_dir)
    print(f"[INFO] {len(u)} raw edges from {edges_dir}")
    u, v, s, c = dedupe_edges(u, v, s, c, how=how)
    print(f"[INFO] {len(u)} unique pairs (dedupe='{how}')")

    keep = s <= threshold
    drop_contact = int((keep & (c < min_contact)).sum()) if min_contact else 0
    print(f"[INFO] score <= {threshold}: {int(keep.sum())} edges; "
          f"of those {drop_contact} dropped by min_contact={min_contact}")
    if len(s):
        qs = np.percentile(s, [1, 10, 50, 90, 99]).round(3).tolist()
        print(f"[INFO] score percentiles (1/10/50/90/99): {qs}")

    nodes, roots = components_from_edges(u, v, s, c, threshold, min_contact)
    n_comp = len(np.unique(roots)) if len(roots) else 0
    print(f"[INFO] {len(nodes)} fragments merged into {n_comp} multi-fragment segments")
    write_lut_npz(lut_path, nodes, roots)
    print(f"[DONE] LUT written: {lut_path}")
    return lut_path
