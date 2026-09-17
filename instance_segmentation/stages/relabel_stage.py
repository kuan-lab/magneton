# -*- coding: utf-8 -*-
"""
Pass 4: apply the LUT, one core at a time.

Fragments not in the LUT keep their own id (a fragment nothing merged with is
already a segment), which together with root = smallest fragment id makes the
mapping idempotent: re-running a core, or running in place, is safe.
"""
import os

import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from magneton.instance_segmentation.stages.fragments_stage import build_grid
from magneton.instance_segmentation.utils.graph_utils import read_lut_npz, apply_lut
from magneton.instance_segmentation.state.checkpoint import mark_local_done, is_local_done


def _ensure_output_volume(fragments_path, output_path):
    src = CloudVolume(fragments_path, mip=0, bounded=False, progress=False)
    info = CloudVolume.create_new_info(
        num_channels=1, layer_type="segmentation", data_type="uint64",
        encoding="raw", resolution=list(map(int, src.resolution)),
        voxel_offset=list(map(int, src.voxel_offset)),
        volume_size=list(map(int, src.info["scales"][0]["size"])),
        chunk_size=list(map(int, src.chunk_size)),
    )
    vol = CloudVolume(output_path, info=info, compress=False, progress=False)
    vol.commit_info()
    vol.commit_provenance()
    return vol


def relabel_core(index, block, *, fragments_path, output_path, lut_path, ckpt_dir):
    nodes, roots = read_lut_npz(lut_path)
    z1, z2, y1, y2, x1, x2 = block["core"]
    fvol = CloudVolume(fragments_path, mip=0, bounded=False, progress=False,
                       fill_missing=True)
    frag = np.asarray(fvol[x1:x2, y1:y2, z1:z2])[..., 0].astype(np.uint64)
    out = apply_lut(frag, nodes, roots)
    ovol = CloudVolume(output_path, mip=0, progress=False, compress=False)
    ovol[x1:x2, y1:y2, z1:z2] = out[..., np.newaxis]
    # Segment sizes for the Neuroglancer property list. Counted here because this
    # core already holds the labels; the separate segment-props stage merges the
    # per-core files into one list (keeps array tasks independent).
    ids, cnt = np.unique(out, return_counts=True)
    keep = ids > 0
    if ckpt_dir:
        np.savez_compressed(os.path.join(ckpt_dir, f"counts_{index:04d}.npz"),
                            ids=ids[keep], counts=cnt[keep])
        mark_local_done(ckpt_dir, index)
    return {"index": int(index), "n_ids": int(keep.sum())}


def relabel_blocks(global_cfg, stage_cfg, lut_path, indices=None, workers=1,
                   restart=False):
    fragments_path = global_cfg["paths"]["fragments"]
    in_place = bool(stage_cfg.get("in_place", False))
    output_path = fragments_path if in_place else global_cfg["paths"]["output"]
    ckpt_dir = global_cfg["checkpoint"].get("relabel_dir")

    aff_vol = CloudVolume(global_cfg["paths"]["input"], mip=0, bounded=False,
                          progress=False)
    blocks, _ = build_grid(global_cfg, aff_vol)
    if not in_place:
        _ensure_output_volume(fragments_path, output_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    todo = list(range(len(blocks))) if indices is None else [int(i) for i in indices]
    if not restart and ckpt_dir:
        todo = [i for i in todo if not is_local_done(ckpt_dir, i)]
    if not todo:
        print("[INFO] relabel: nothing to do.")
        return
    print(f"[INFO] relabel: {len(todo)}/{len(blocks)} cores -> {output_path}"
          f"{' (IN PLACE)' if in_place else ''}")

    kw = dict(fragments_path=fragments_path, output_path=output_path,
              lut_path=lut_path, ckpt_dir=ckpt_dir)
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(relabel_core, i, blocks[i], **kw) for i in todo]
            for f in as_completed(futs):
                f.result()
    else:
        for i in tqdm(todo, desc="relabel"):
            relabel_core(i, blocks[i], **kw)

    print("[DONE] relabel stage finished.")
