"""
Stage B — expand a (corrected) skeleton into dense segments via nnInteractive.

For each neuron tree in the NML, feed its skeleton as a prompt (scribble of node
voxels, or subsampled points) to an nnInteractive session over the EM, and collect
the predicted mask under that neuron's label. Writes <output>/expanded.tif.

Runs in the nnInteractive env on a GPU node — submit via expand_hpc, or run
directly:  conda run -p <nninteractive_env> python -m magneton.proofreading.stages.expand --config <cfg>
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from magneton.proofreading.config import (
    load_config, get_stage_config, strip_file_prefix,
)
from magneton.proofreading.lib import skeleton_io


def _resolve_model_dir(ex: dict):
    if ex.get("model_dir"):
        return ex["model_dir"]
    if ex.get("hf_cache"):
        os.environ.setdefault("HF_HUB_CACHE", ex["hf_cache"])
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="nnInteractive/nnInteractive",
                          allow_patterns=["nnInteractive_v1.0/*"])
    return os.path.join(p, "nnInteractive_v1.0")


def expand(cfg: dict):
    import tifffile
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    paths = get_stage_config(cfg, "paths")
    ex = get_stage_config(cfg, "expand")
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    nml = ex.get("nml") or os.path.join(out_dir, "skeletons.nml")
    out_tif = os.path.join(out_dir, "expanded.tif")

    em_path = strip_file_prefix(paths["em"])
    em = tifffile.imread(em_path).transpose(2, 1, 0).astype(np.float32)   # (x,y,z)
    X, Y, Z = em.shape
    hi = np.array([X - 1, Y - 1, Z - 1])
    trees = skeleton_io.parse_nml(nml)                 # per-tree records
    trees.sort(key=lambda r: r["tree_id"])
    if ex.get("max_neurons", 0):
        trees = trees[: ex["max_neurons"]]
    prompt = ex.get("prompt", "scribble")
    print(f"[expand] EM {em.shape}  nml={nml}  expanding {len(trees)} trees "
          f"(prompt={prompt})", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sess = nnInteractiveInferenceSession(device=dev, verbose=False, torch_n_threads=8)
    sess.initialize_from_trained_model_folder(_resolve_model_dir(ex), use_fold=0)
    sess.set_image(em[None])                                              # (1,x,y,z)
    print(f"[expand] device={dev}  image set", flush=True)

    out = np.zeros((X, Y, Z), dtype=np.uint16)
    t0 = time.time()
    for i, rec in enumerate(trees, 1):
        pts = rec["points"]
        out_label = rec["tree_id"]          # unique per tree → split neurons stay split
        target = np.zeros((X, Y, Z), dtype=np.uint8)
        sess.set_target_buffer(target)
        sess.reset_interactions()
        if prompt == "scribble":
            scr = np.zeros((X, Y, Z), dtype=np.uint8)
            idx = np.clip(np.round(pts).astype(int), 0, hi)
            scr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
            sess.add_scribble_interaction(scr, include_interaction=True)
        else:
            sub = pts[:: ex.get("point_subsample", 10)]
            for j, p in enumerate(sub):
                c = tuple(int(v) for v in np.clip(np.round(p).astype(int), 0, hi))
                sess.add_point_interaction(c, include_interaction=True,
                                           run_prediction=(j == len(sub) - 1))
        res = np.asarray(target)
        out[res > 0] = out_label
        print(f"  [{i}/{len(trees)}] tree {rec['tree_id']} ({rec['name']}): "
              f"{len(pts)} nodes -> {int((res > 0).sum())} voxels", flush=True)

    tifffile.imwrite(out_tif, out.transpose(2, 1, 0))                     # -> (z,y,x)
    print(f"[expand] {len(labels)} neurons in {time.time()-t0:.1f}s -> {out_tif}", flush=True)
    return out_tif


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading stage B — nnInteractive expand")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    expand(load_config(args.config))


if __name__ == "__main__":
    main()
