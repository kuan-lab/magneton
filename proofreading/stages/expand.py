"""
Stage B — expand (corrected) skeletons into dense segments via nnInteractive.

Each neuron tree's skeleton nodes are a POSITIVE scribble; the shared `membrane_net`
tree (the affinity/seg-boundary fishnet) is a NEGATIVE scribble that walls off the
flood across membranes. For each neuron: reset, add positive scribble, add the
negative net, predict once, collect the mask under that tree's id. Writes
<output>/expanded.tif and reports foreground % + overlap (the flooding metric).

Runs in the nnInteractive env on a GPU node — submit via expand_hpc, or:
  conda run -p <nninteractive_env> python -m magneton.proofreading.stages.expand --config <cfg>
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


def _scribble(shape, pts, hi):
    """Binary scribble volume with 1s at the (clipped, rounded) point voxels."""
    scr = np.zeros(shape, dtype=np.uint8)
    idx = np.clip(np.round(pts).astype(int), 0, hi)
    scr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return scr


def expand(cfg: dict):
    import tifffile
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    paths = get_stage_config(cfg, "paths")
    ex = get_stage_config(cfg, "expand")
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    nml = ex.get("nml") or os.path.join(out_dir, "skeletons.nml")
    em_path = strip_file_prefix(ex.get("em") or paths["em"])
    out_tif = os.path.join(out_dir, ex.get("out_name", "expanded.tif"))

    em = tifffile.imread(em_path).transpose(2, 1, 0).astype(np.float32)   # (x,y,z)
    X, Y, Z = em.shape
    hi = np.array([X - 1, Y - 1, Z - 1])

    neg_prefix = ex.get("negative_prefix", "membrane_net")
    recs = skeleton_io.parse_nml(nml)
    negs = [r for r in recs if r["name"].startswith(neg_prefix)]
    neurons = [r for r in recs if not r["name"].startswith(neg_prefix)]
    neurons.sort(key=lambda r: r["tree_id"])
    if ex.get("max_neurons", 0):
        neurons = neurons[: ex["max_neurons"]]

    # Shared negative scribble from ALL membrane_net nodes (curated once, reused).
    use_neg = ex.get("use_negatives", True) and len(negs) > 0
    neg_scr = None
    if use_neg:
        allpts = np.concatenate([r["points"] for r in negs], axis=0)
        neg_scr = _scribble((X, Y, Z), allpts, hi)
        print(f"[expand] negatives ON: {len(allpts)} membrane points", flush=True)
    else:
        print(f"[expand] negatives OFF (use_negatives={ex.get('use_negatives', True)}, "
              f"membrane trees={len(negs)})", flush=True)
    print(f"[expand] EM {em.shape}  nml={nml}  expanding {len(neurons)} neurons", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sess = nnInteractiveInferenceSession(device=dev, verbose=False, torch_n_threads=8)
    sess.initialize_from_trained_model_folder(_resolve_model_dir(ex), use_fold=0)
    sess.set_image(em[None])                                              # (1,x,y,z)
    print(f"[expand] device={dev}  image set", flush=True)

    out = np.zeros((X, Y, Z), dtype=np.uint16)
    cover = np.zeros((X, Y, Z), dtype=np.uint16)        # how many masks hit each voxel
    t0 = time.time()
    for i, rec in enumerate(neurons, 1):
        pts, lab = rec["points"], rec["tree_id"]
        target = np.zeros((X, Y, Z), dtype=np.uint8)
        sess.set_target_buffer(target)
        sess.reset_interactions()
        sess.add_scribble_interaction(_scribble((X, Y, Z), pts, hi),
                                      include_interaction=True,
                                      run_prediction=not use_neg)
        if use_neg:
            sess.add_scribble_interaction(neg_scr, include_interaction=False,
                                          run_prediction=True)
        res = np.asarray(target) > 0
        out[res] = lab
        cover[res] += 1
        print(f"  [{i}/{len(neurons)}] tree {lab} ({rec['name']}): "
              f"{len(pts)} nodes -> {int(res.sum())} voxels", flush=True)

    tifffile.imwrite(out_tif, out.transpose(2, 1, 0))                     # -> (z,y,x)
    fg = int((cover > 0).sum())
    ov = int((cover > 1).sum())
    tot = int(cover.sum())
    print(f"[expand] {len(neurons)} neurons in {time.time()-t0:.1f}s | "
          f"foreground {fg} ({100*fg/em.size:.1f}%), overlap {ov} ({100*ov/em.size:.1f}%), "
          f"mask-sum/volume {tot/em.size:.2f}x -> {out_tif}", flush=True)
    return out_tif


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading stage B — nnInteractive expand")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    expand(load_config(args.config))


if __name__ == "__main__":
    main()
