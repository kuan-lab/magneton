# -*- coding: utf-8 -*-
import os
import math
import json
import subprocess
from pathlib import Path

from magneton.instance_segmentation.config import load_config, load_global_config_path
from magneton.instance_segmentation.utils.block_utils import generate_blocks_zyx
from cloudvolume import CloudVolume


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _pending_block_indices(cfg, restart=False):
    """Compute all blocks and filter out completed blocks (depending on checkpoints/local/*.done)"""
    input_path = cfg["paths"]["input"]
    mip = cfg.get("local_stage", {}).get("mip", 0)

    aff_vol = CloudVolume(input_path, mip=mip, bounded=False, progress=False)
    vol_size_xyz = tuple(aff_vol.info["scales"][0]["size"])
    vol_shape_zyx = (vol_size_xyz[2], vol_size_xyz[1], vol_size_xyz[0])

    block_size = tuple(cfg["block"]["size"])
    overlap = tuple(cfg["block"]["overlap"])
    blocks = generate_blocks_zyx(vol_shape_zyx, block_size, overlap)

    local_ckpt_dir = cfg["checkpoint"]["segmentation_dir"]
    os.makedirs(local_ckpt_dir, exist_ok=True)

    pending = []
    for i, _coords in enumerate(blocks):
        done_flag = os.path.join(local_ckpt_dir, f"block_{i:04d}.done")
        if os.path.exists(done_flag) or restart:
            pending.append(i)
    return pending


def _write_manifest(job_dir: str, indices, blocks_per_job: int):
    """List the indexes to be processed in manifest.txt (each line containing a comma-separated group of indexes)."""
    _ensure_dir(job_dir)
    chunks = [
        indices[i:i + blocks_per_job]
        for i in range(0, len(indices), blocks_per_job)
    ]
    manifest = os.path.join(job_dir, "manifest.txt")
    with open(manifest, "w") as f:
        for group in chunks:
            f.write(",".join(map(str, group)) + "\n")
    return manifest, len(chunks)


def _slurm_script(cfg, stage_cfg, job_dir, array_len):
    hpc = stage_cfg["hpc"]
    python_bin = hpc.get("python_bin", "python")
    workers_per_job = int(hpc.get("workers_per_job", 2))
    time = hpc.get("time", "04:00:00")
    mem = hpc.get("mem", "16G")
    cpus = hpc.get("cpus", "8")
    partition = hpc.get("partition", None)
    # account = hpc.get("account", None)
    # qos = hpc.get("qos", None)
    extra_modules = hpc.get("extra_modules", [])

    conda = hpc.get("conda", None)
    env = hpc.get("env", None)
    work_path = hpc.get("work_path", None)

    script_path = os.path.join(job_dir, "submit_slurm.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=segmentation_chunks",
        f"#SBATCH --time={time}",
        f"#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem-per-cpu={mem}",
        f"#SBATCH --array=0-{array_len-1}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
    ]
    if partition:   lines.append(f"#SBATCH --partition={partition}")
    # if account:     lines.append(f"#SBATCH --account={account}")
    # if qos:         lines.append(f"#SBATCH --qos={qos}")

    # module load
    for m in extra_modules:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append(f"export SLURM_EXPORT_ENV=ALL")
    if conda:       lines.append(f"source {conda}")
    if env:         lines.append(f"conda activate {env}")
    if work_path:   lines.append(f"cd {work_path}")

    manifest = os.path.join(job_dir, "manifest.txt")
    global_cfgs = load_global_config_path("magneton/config.yaml")
    # global_cfgs = cfg
    # cfg_path = global_cfgs.get("instance_segmentation/mian", "magneton/instance_segmentation/configs/config.yaml")
    cfg_path = (
        global_cfgs.get("instance_segmentation", {})
                .get("mian", "magneton/instance_segmentation/configs/config.yaml")
    )
    lines += [
        "set -e",
        f'echo \"Running: {manifest}\"',

        # Run the locally parallel scripts within each job (which will parse indices and run on a single node using ProcessPool).
        f"{python_bin} -m magneton.instance_segmentation.stages.merge_apply "
        f"--config {cfg_path}"
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_local_hpc(global_cfg, stage_cfg, restart=False, dry_run=False):
    """
    Generate job lists and submission scripts (Slurm job arrays).
    Process a set of block indices locally on nodes using instance_segmentation.tools.run_local_shard.
    """
    hpc = stage_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] local_stage.hpc.enable=false, HPC submission is disabled.")
        return

    scheduler = hpc.get("scheduler", "slurm").lower()
    job_dir = hpc.get("job_dir", "magneton/jobs/merge")
    blocks_per_job = int(hpc.get("blocks_per_job", 1))

    # Calculate the blocks to be processed
    pending = _pending_block_indices(global_cfg, restart=restart)
    if not pending:
        print("[INFO] No pending blocks (or all completed).")
        return

    manifest, n_chunks = _write_manifest(job_dir, pending, blocks_per_job)
    print(f"[INFO] {len(pending)} blocks pending processing, manifest: {manifest}.")

    # Generate Script
    if scheduler == "slurm":
        script_path = _slurm_script(global_cfg, stage_cfg, job_dir, 1)
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
