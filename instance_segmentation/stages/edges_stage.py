# -*- coding: utf-8 -*-
"""
Pass 2 of the graph pipeline: region-graph edges, including across core faces.

Each block reads FRAGMENTS + AFFINITY over core+halo. Because pass 1 already
wrote every core, the halo holds the neighbours' fragments, so waterz sees a
single fragment array and returns edges that straddle a core face as ordinary
edges -- scored by the same merge function used inside a block.

This is the step that replaces overlap voting: the question becomes "how strong
is the boundary between these two fragments" (answerable from the affinity)
instead of "did two independent labelings of the same voxels agree".
"""
import os

import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from waterz import agglomerate

from magneton.instance_segmentation.waterz_block import getScoreFunc
from magneton.instance_segmentation.stages.fragments_stage import build_grid
from magneton.instance_segmentation.utils.graph_utils import (
    MergeTree, contact_counts, write_edges_npz, fragment_id_bump,
)
from magneton.instance_segmentation.state.checkpoint import is_local_done


def _read_block(fragments_path, input_path, read, mip):
    z1, z2, y1, y2, x1, x2 = read
    fvol = CloudVolume(fragments_path, mip=0, bounded=False, progress=False,
                       fill_missing=True)
    frag = np.asarray(fvol[x1:x2, y1:y2, z1:z2])[..., 0]
    frag = np.ascontiguousarray(np.transpose(frag, (2, 1, 0)).astype(np.uint64))

    avol = CloudVolume(input_path, mip=mip, bounded=False, progress=False,
                       fill_missing=True)
    aff = np.asarray(avol[x1:x2, y1:y2, z1:z2])
    aff = np.transpose(aff, (3, 2, 1, 0)).astype(np.float32)   # (c, z, y, x)
    if aff.max() > 1.0:
        aff /= 255.0
    if aff.shape[0] == 1:
        aff = np.concatenate([aff, aff, aff], axis=0)
    return frag, np.ascontiguousarray(aff)


def edges_for_block(index, block, core_size, *, input_path, fragments_path, mip,
                    stage_cfg, edges_dir):
    """Region graph for one core's read region -> edges_<index>.npz."""
    frag, aff = _read_block(fragments_path, input_path, block["read"], mip)

    # waterz memory scales with max label -> relabel locally, keep the way back
    ids = np.unique(frag)
    local = np.searchsorted(ids, frag).astype(np.uint64)
    back = ids.astype(np.uint64)                       # local id -> global id
    if ids[0] != 0:
        local += np.uint64(1)
        back = np.concatenate([[np.uint64(0)], back])
    del frag

    # waterz.agglomerate MUTATES `fragments` in place, so measure contact area
    # first -- afterwards `local` holds agglomerated labels, not fragments.
    contacts = contact_counts(local)

    threshold = float(stage_cfg.get("score_threshold", 0.45))
    aff_thr = stage_cfg.get("aff_thresholds", [0.00001, 0.99999])
    gen = agglomerate(
        aff, [0, threshold],
        fragments=np.ascontiguousarray(local),
        aff_threshold_low=aff_thr[0], aff_threshold_high=aff_thr[1],
        scoring_function=getScoreFunc(stage_cfg.get("merge_function", "aff50_his256")),
        discretize_queue=256,
        return_merge_history=True, return_region_graph=True,
    )
    _, _, initial_rag = next(gen)                      # edges + initial scores
    _, merge_history, _ = next(gen)                    # merges up to threshold
    for _ in gen:
        pass

    tree = MergeTree()
    for m in merge_history:
        tree.merge(m["a"], m["b"], m["c"], m["score"])

    # keep edges with at least one endpoint owned by THIS core; the neighbour
    # writes the rest, and dedupe_edges collapses whatever overlaps
    cap = int(np.prod(np.asarray(core_size, dtype=np.uint64)))
    lo_own = int(fragment_id_bump(index, core_size))
    hi_own = lo_own + cap

    u_out, v_out, s_out, c_out = [], [], [], []
    for e in initial_rag:
        lu, lv = int(e["u"]), int(e["v"])
        if lu == 0 or lv == 0:
            continue
        gu, gv = int(back[lu]), int(back[lv])
        if gu == gv:
            continue
        if not (lo_own < gu <= hi_own or lo_own < gv <= hi_own):
            continue
        # The merge-tree score is waterz's VERDICT: the threshold at which these
        # two regions actually became one. If waterz refused to merge them, the
        # pair must be recorded as not-merged (inf) -- NOT as e["score"], which is
        # the pre-agglomeration fragment-vs-fragment score and ignores the
        # re-scoring of the growing region that caused the refusal. Falling back
        # to e["score"] resurrects every merge waterz deliberately rejected.
        merged_at = tree.find_merge(lu, lv)
        score = float(merged_at) if merged_at is not None else float("inf")
        a, b = (lu, lv) if lu < lv else (lv, lu)
        u_out.append(min(gu, gv))
        v_out.append(max(gu, gv))
        s_out.append(score)
        c_out.append(contacts.get((a, b), 0))

    cross = sum(1 for a, b in zip(u_out, v_out)
                if (a - 1) // cap != (b - 1) // cap)
    out = os.path.join(edges_dir, f"edges_{index:04d}.npz")
    write_edges_npz(out, u_out, v_out, s_out, c_out)
    print(f"[INFO] core {index}: {len(u_out)} edges kept ({cross} cross-core) "
          f"(rag {len(initial_rag)}, merges {len(merge_history)}) -> {out}")
    return {"index": int(index), "n_edges": len(u_out), "path": out}


def edges_blocks(global_cfg, stage_cfg, indices=None, workers=1, restart=False):
    """Run pass 2. Requires pass 1 to have finished for ALL cores."""
    input_path = global_cfg["paths"]["input"]
    fragments_path = global_cfg["paths"]["fragments"]
    mip = stage_cfg.get("mip", 0)
    edges_dir = stage_cfg.get("edges_dir", "magneton/metadata/edges")
    os.makedirs(edges_dir, exist_ok=True)

    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    blocks, core_size = build_grid(global_cfg, aff_vol)

    frag_ckpt = global_cfg["checkpoint"]["fragments_dir"]
    missing = [i for i in range(len(blocks)) if not is_local_done(frag_ckpt, i)]
    if missing and stage_cfg.get("require_all_fragments", True):
        raise RuntimeError(
            f"pass 1 incomplete: {len(missing)} cores missing (e.g. {missing[:5]}). "
            "Cross-core edges need the neighbours' fragments on disk; "
            "set edges_stage.require_all_fragments=false to override.")

    todo = list(range(len(blocks))) if indices is None else [int(i) for i in indices]
    if not restart:
        todo = [i for i in todo
                if not os.path.exists(os.path.join(edges_dir, f"edges_{i:04d}.npz"))]
    if not todo:
        print("[INFO] edges: nothing to do.")
        return
    print(f"[INFO] edges: {len(todo)}/{len(blocks)} cores, workers={workers}")

    kw = dict(input_path=input_path, fragments_path=fragments_path, mip=mip,
              stage_cfg=stage_cfg, edges_dir=edges_dir)
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(edges_for_block, i, blocks[i], core_size, **kw)
                    for i in todo]
            for f in as_completed(futs):
                f.result()
    else:
        for i in tqdm(todo, desc="edges"):
            edges_for_block(i, blocks[i], core_size, **kw)
    print("[DONE] edges stage finished.")
