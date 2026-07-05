#!/usr/bin/env python
"""
Emit a STANDALONE WebKnossos Zarr-v3 agglomerate attachment from an npz bundle
(the merge-supervox / agglomerate_io output), for serving alongside a REMOTE
precomputed supervoxel layer: Globus -> nginx -> WebKnossos "Add Remote Dataset".

Unlike upload_webknossos.upload_supervoxels_official (which builds a full WKS Zarr
dataset and UPLOADS it), this writes ONLY the agglomerate attachment — no EM, no
segmentation volume, no upload, no token/context. Just the zarr3 group + the
datasource-properties.json snippet needed to register it under the remote
segmentation layer's `attachments.agglomerates`.

Runs in the wk-latest env (webknossos-py >= 3.5.3), e.g.:
  /gpfs/radev/project/kuan/yf354/conda_envs/wk-latest/bin/python emit_agglomerate_zarr.py \
      --bundle .../supervox_agglomerate.npz --out .../fib_b_neuron_supervox_sv/agglomerates/agglomerate

Bundle arrays (all dense, supervoxel ids 1..N): seg_ids, positions(N,3), edge_a,
edge_b, affinity. Connected components of the affinity graph become the agglomerates.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from webknossos import Vec3Int
from webknossos.dataset.layer.segmentation_layer.attachments.agglomerate_attachment import (
    AgglomerateGraph,
    AgglomerateAttachment,
)


def emit(bundle_path, out_path, seg_dtype="uint32"):
    b = np.load(bundle_path)
    g = AgglomerateGraph()
    for sid, xyz in zip(b["seg_ids"], b["positions"]):
        g.add_segment(int(sid), position=Vec3Int(int(xyz[0]), int(xyz[1]), int(xyz[2])))
    for a, bb, w in zip(b["edge_a"], b["edge_b"], b["affinity"]):
        g.add_affinity_edge(int(a), int(bb), affinity=float(w))
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    AgglomerateAttachment.create(out_path, g, segmentation_dtype=seg_dtype)
    return int(len(b["seg_ids"])), int(len(b["edge_a"]))


def main():
    ap = argparse.ArgumentParser(
        description="Emit a standalone Zarr-v3 agglomerate attachment from an npz bundle.")
    ap.add_argument("--bundle", required=True,
                    help="npz bundle (seg_ids, positions, edge_a, edge_b, affinity)")
    ap.add_argument("--out", required=True,
                    help="output attachment dir; its basename is the mapping name "
                         "(convention: <layer>/agglomerates/<name>)")
    ap.add_argument("--seg-dtype", default="uint32", choices=["uint32", "uint64"],
                    help="must match the supervoxel layer dtype")
    args = ap.parse_args()

    n_nodes, n_edges = emit(args.bundle, args.out, args.seg_dtype)
    out = Path(args.out)
    name = out.name
    print(f"WROTE {out}  ({n_nodes} nodes, {n_edges} edges)")
    print("\nAdd under the segmentation layer in datasource-properties.json:")
    print(json.dumps(
        {"attachments": {"agglomerates": [
            {"name": name, "path": f"./agglomerates/{name}", "dataFormat": "zarr3"}]}},
        indent=2))
    print(f"\nServe {out} at  <layer-url>/agglomerates/{name}/  (with CORS enabled).")


if __name__ == "__main__":
    main()
