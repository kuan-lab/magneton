# -*- coding: utf-8 -*-
import os
import h5py
import yaml
import argparse
import numpy as np
import tifffile as tiff
from cloudvolume import CloudVolume

from magneton.pytorch_connectomics.utils.config import load_config, load_global_config_path
import gc
import subprocess
from pathlib import Path


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _gen_chunk_configs(config_base, configs_save_path, input_floder):
    configs_to_run = []
    for fname in os.listdir(input_floder):
        # fpath = os.path.join(folder, fname)
        configs_to_run.append({
            "INFERENCE": {
                "IMAGE_NAME": f"{fname}",  
                "OUTPUT_NAME": f"{fname.split('.')[0]}.h5"
            },
            "DATASET": {
                "INPUT_PATH": f"{input_floder}",
            }
        })

    for i, new_params in enumerate(configs_to_run):
        with open(config_base) as f:
            config_data = yaml.safe_load(f)
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        update_dict(config_data, new_params)
        new_config_path = f"{configs_save_path}/temp_config_{i}.yaml"
        with open(new_config_path, "w") as f:
            yaml.safe_dump(config_data, f)


def _slurm_script(cfg, stage_cfg, job_dir, array_len):
    hpc = stage_cfg["hpc"]
    python_bin = hpc.get("python_bin", "python")
    time = hpc.get("time", "04:00:00")
    mem = hpc.get("mem-per-gpu", "16G")
    cpus = hpc.get("cpus", "8")
    gpus = hpc.get("gpus", "a40:1")
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
    global_cfgs = load_global_config_path("magneton/config.yaml")
    # cfg_path = global_cfgs.get("instance_segmentation/gen_mask", "magneton/instance_segmentation/configs/config_gen_mask.yaml")
    cfg_file_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("config_file", "magneton/pytorch_connectomics/configs/config_file.yaml")
    )
    cfg_base_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("config_base", "magneton/pytorch_connectomics/configs/config_base.yaml")
    )
    checkpoint_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("checkpoint", "magneton/pytorch_connectomics/configs/checkpoint.yaml")
    )

    mutil_jobs_flag = hpc.get("mutil_jobs", False)
    if mutil_jobs_flag and cfg.stage in ["inference-hpc",]:
        mutil_jobs_configs = hpc.get("mutil_jobs_configs", {})
        configs_save_path = mutil_jobs_configs.get("configs_save_path", '')
        input_floder = mutil_jobs_configs.get("input_floder", '')
        _ensure_dir(configs_save_path)
        jobs_num = len(os.listdir(input_floder))
        batch_num = mutil_jobs_configs.get("batch_num", 1)
        _gen_chunk_configs(cfg_base_path, configs_save_path, input_floder)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pytc",
            f"#SBATCH --time={time}",
            f"#SBATCH --ntasks=1 --nodes=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem-per-gpu={mem}",
            f"#SBATCH --gpus={gpus}",
            f"#SBATCH --array=0-{jobs_num-1}%{batch_num}",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]
        cfg_base_path = os.path.join(configs_save_path, "temp_config_${SLURM_ARRAY_TASK_ID}.yaml")
        
    else:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pytc",
            f"#SBATCH --time={time}",
            f"#SBATCH --ntasks=1 --nodes=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem-per-gpu={mem}",
            f"#SBATCH --gpus={gpus}",
            f"#SBATCH --array=0-{array_len-1}",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]
    if partition:   lines.append(f"#SBATCH --partition={partition}")

    # module load
    for m in extra_modules:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append(f"export SLURM_EXPORT_ENV=ALL")
    if conda:       lines.append(f"source {conda}")
    if env:         lines.append(f"conda activate {env}")
    if work_path:   lines.append(f"cd {work_path}")

        
    if cfg.stage == "pre-train-hpc":
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path}"
        ]
    elif cfg.stage == "fine-tune-hpc":
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path} --checkpoint {checkpoint_path}"
        ]
    else:
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path} --inference --checkpoint {checkpoint_path}"
        ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_local_hpc(global_cfg, hpc_cfg, restart=False, dry_run=False):
    """
    Generate job lists and submission scripts (Slurm job arrays).
    Process a set of block indices locally on nodes using instance_segmentation.tools.gen_mask.
    """
    hpc = hpc_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] local_stage.hpc.enable=false, HPC submission is disabled.")
        return

    scheduler = hpc.get("scheduler", "slurm").lower()
    job_dir = hpc.get("job_dir", "./jobs/pytc")

    # Generate Script
    if scheduler == "slurm":
        script_path = _slurm_script(global_cfg, hpc_cfg, job_dir, 1)
        submit_cmd = ["sbatch", script_path]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    print(f"[INFO] Submit command: {' '.join(submit_cmd)}")
    if not dry_run:
        try:
            out = subprocess.check_output(submit_cmd, stderr=subprocess.STDOUT)
            out_msg = out.decode("utf-8", "ignore")
            print(f"[INFO] Submit output: {out_msg}")
        except Exception as e:
            print(f"[WARN] Submission failed:{e}")
            print(f"[HINT] You can manually execute the command:{' '.join(submit_cmd)}")

def run_hpc(global_cfg, hpc_cfg, restart=False, dry_run=False):
    submit_local_hpc(global_cfg, hpc_cfg, restart, dry_run)