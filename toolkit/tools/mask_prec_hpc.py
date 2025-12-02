# -*- coding: utf-8 -*-
import os
import h5py
import argparse
import numpy as np
import tifffile as tiff
from cloudvolume import CloudVolume
from magneton.toolkit.utils.config import load_config, load_global_config_path
import gc
import subprocess
from pathlib import Path


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _slurm_script(cfg, stage_cfg, job_dir, array_len):
    hpc = stage_cfg["hpc"]
    python_bin = hpc.get("python_bin", "python")
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
        f"#SBATCH --job-name=mask_prec",
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

    global_cfgs = load_global_config_path("magneton/config.yaml")
    # cfg_path = global_cfgs.get("toolkit/mask_prec", "magneton/toolkit/configs/config_mask.yaml")
    cfg_path = (
        global_cfgs.get("toolkit", {})
                .get("mask_prec", "magneton/toolkit/configs/config_mask.yaml")
    )
    lines += [
        f"{python_bin} -m magneton.toolkit.tools.mask_prec "
        f"--config {cfg_path}"
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_local_hpc(global_cfg, stage_cfg, restart=False, dry_run=False):
    """
    Generate job lists and submission scripts (Slurm job arrays).
    Process a set of block indices locally on nodes using toolkit.tools.mask_prec.
    """
    hpc = stage_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] local_stage.hpc.enable=false, HPC submission is disabled.")
        return

    scheduler = hpc.get("scheduler", "slurm").lower()
    job_dir = hpc.get("job_dir", "./jobs/mask")

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

def mask_prec_hpc(global_cfg, restart=False, dry_run=False):
    submit_local_hpc(global_cfg, global_cfg, restart, dry_run)