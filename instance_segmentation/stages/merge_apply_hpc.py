# -*- coding: utf-8 -*-
import os
import math
import subprocess
from pathlib import Path

from cloudvolume import CloudVolume

from magneton.instance_segmentation.config import load_config, load_global_config_path
from magneton.instance_segmentation.utils.meta_utils import load_index_meta
from magneton.instance_segmentation.stages.merge_apply import _ensure_output_volume


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _count_done_blocks(global_cfg, stage_cfg):
    """Count the blocks marked done in the segmentation metadata dir.

    merge_apply iterates the same set, so this gives the SLURM array size
    when divided by blocks_per_job.
    """
    metadata_dir = stage_cfg.get("metadata_dir", "./metadata/local_metadata")
    index_data = load_index_meta(metadata_dir)
    blocks_meta = [b for b in index_data.get("blocks", []) if b.get("done", False)]
    return len(blocks_meta)


def _slurm_script(cfg, stage_cfg, job_dir, num_tasks, blocks_per_job, batch_num):
    # Prefer hpc_apply (light, SLURM-array). Fall back to hpc for legacy
    # configs that have only one shared block.
    hpc = stage_cfg.get("hpc_apply") or stage_cfg["hpc"]
    python_bin = hpc.get("python_bin", "python")
    time = hpc.get("time", "04:00:00")
    mem = hpc.get("mem", "16G")
    cpus = hpc.get("cpus", "8")
    partition = hpc.get("partition", None)
    extra_modules = hpc.get("extra_modules", [])

    conda = hpc.get("conda", None)
    env = hpc.get("env", None)
    work_path = hpc.get("work_path", None)

    script_path = os.path.join(job_dir, "submit_slurm.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=merge_apply",
        f"#SBATCH --time={time}",
        f"#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem-per-cpu={mem}",
        f"#SBATCH --array=0-{num_tasks-1}%{batch_num}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
    ]
    if partition:   lines.append(f"#SBATCH --partition={partition}")

    for m in extra_modules:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append(f"export SLURM_EXPORT_ENV=ALL")
    if conda:       lines.append(f"source {conda}")
    if env:         lines.append(f"conda activate {env}")
    if work_path:   lines.append(f"cd {work_path}")

    global_cfgs = load_global_config_path("magneton/config.yaml")
    cfg_path = (
        global_cfgs.get("instance_segmentation", {})
                .get("main", "magneton/instance_segmentation/configs/config.yaml")
    )
    lines += [
        "set -e",
        f"{python_bin} -m magneton.instance_segmentation.stages.merge_apply "
        f"--config {cfg_path} "
        f"--task-id ${{SLURM_ARRAY_TASK_ID}} --blocks-per-task {blocks_per_job}"
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_local_hpc(global_cfg, stage_cfg, restart=False, dry_run=False):
    """
    Submit a SLURM array job that runs merge_apply on slices of the block
    list. Each task processes blocks_per_job blocks; cores are chunk-disjoint
    by construction (compute_core_region snap-down), so concurrent tasks
    write to non-overlapping output chunks without locking.
    """
    hpc = stage_cfg.get("hpc_apply") or stage_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] merge_stage hpc_apply.enable=false, HPC submission is disabled.")
        return

    scheduler = hpc.get("scheduler", "slurm").lower()
    job_dir = hpc.get("job_dir", "magneton/jobs/merge")
    blocks_per_job = int(hpc.get("blocks_per_job", 1))
    batch_num = int(hpc.get("batch_num", 64))

    n_blocks = _count_done_blocks(global_cfg, stage_cfg)
    if n_blocks == 0:
        print("[INFO] No done blocks in metadata; nothing to apply.")
        return
    num_tasks = math.ceil(n_blocks / blocks_per_job)
    print(f"[INFO] merge_apply: {n_blocks} blocks "
          f"-> {num_tasks} SLURM tasks (blocks_per_job={blocks_per_job}, "
          f"concurrency cap={batch_num})")

    # Pre-create the output segmentation volume so workers find it ready.
    _ensure_output_volume(
        global_cfg["paths"]["input"],
        global_cfg["paths"]["output"],
        stage_cfg.get("mip", 0),
    )

    if scheduler == "slurm":
        script_path = _slurm_script(global_cfg, stage_cfg, job_dir,
                                    num_tasks, blocks_per_job, batch_num)
        submit_cmd = ["sbatch", script_path]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    print(f"[INFO] Submit command: {' '.join(submit_cmd)}")
    if not dry_run:
        try:
            out = subprocess.check_output(submit_cmd, stderr=subprocess.STDOUT)
            out_msg = out.decode("utf-8", "ignore")
            print(f"[INFO] Submit Output: {out_msg}")
        except Exception as e:
            print(f"[WARN] Submission failed:{e}")
            print(f"[HINT] You can manually execute the command:{' '.join(submit_cmd)}")


def apply_pools_to_global_hpc(global_cfg, stage_cfg, restart=False, dry_run=False):
    submit_local_hpc(global_cfg, stage_cfg, restart=restart, dry_run=dry_run)
