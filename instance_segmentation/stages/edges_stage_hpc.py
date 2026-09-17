# -*- coding: utf-8 -*-
"""SLURM drivers for pass 2 (edges) and pass 4 (relabel)."""
import os

from cloudvolume import CloudVolume

from magneton.instance_segmentation.stages.fragments_stage import build_grid
from magneton.instance_segmentation.stages.fragments_stage_hpc import submit_array
from magneton.instance_segmentation.state.checkpoint import is_local_done


def _grid(global_cfg, stage_cfg):
    aff_vol = CloudVolume(global_cfg["paths"]["input"], mip=stage_cfg.get("mip", 0),
                          bounded=False, progress=False)
    return build_grid(global_cfg, aff_vol)[0]


def edges_blocks_hpc(global_cfg, stage_cfg, restart=False, dry_run=False,
                     config_path=None):
    blocks = _grid(global_cfg, stage_cfg)
    frag_ckpt = global_cfg["checkpoint"]["fragments_dir"]
    missing = [i for i in range(len(blocks)) if not is_local_done(frag_ckpt, i)]
    if missing and stage_cfg.get("require_all_fragments", True):
        raise RuntimeError(
            f"pass 1 incomplete: {len(missing)} cores missing (e.g. {missing[:5]}); "
            "cross-core edges need every neighbour's fragments on disk.")
    edges_dir = stage_cfg.get("edges_dir", "magneton/metadata/edges")
    os.makedirs(edges_dir, exist_ok=True)
    pending = [i for i in range(len(blocks))
               if restart or not os.path.exists(
                   os.path.join(edges_dir, f"edges_{i:04d}.npz"))]
    submit_array(global_cfg, stage_cfg, pending, "edges",
                 "magneton.instance_segmentation.tools.run_edges_shard",
                 "magneton/jobs/edges", dry_run=dry_run, config_path=config_path)


def relabel_blocks_hpc(global_cfg, stage_cfg, lut_path, restart=False,
                       dry_run=False, config_path=None):
    blocks = _grid(global_cfg, stage_cfg)
    ckpt = global_cfg["checkpoint"].get("relabel_dir")
    pending = [i for i in range(len(blocks))
               if restart or not (ckpt and is_local_done(ckpt, i))]
    # the worker needs the LUT path, so extend the shared submit with an arg
    stage_cfg = dict(stage_cfg)
    stage_cfg.setdefault("hpc", {})
    submit_array(global_cfg, stage_cfg, pending, "relabel",
                 "magneton.instance_segmentation.tools.run_relabel_shard "
                 f"--lut {lut_path}",
                 "magneton/jobs/relabel", dry_run=dry_run, config_path=config_path)
