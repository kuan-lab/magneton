# -*- coding: utf-8 -*-
"""SLURM driver for pass 1 (fragments). Mirrors segmentation_stage_hpc.py."""
import os
import subprocess
from pathlib import Path

from cloudvolume import CloudVolume

from magneton.instance_segmentation.config import load_global_config_path
from magneton.instance_segmentation.stages.fragments_stage import build_grid
from magneton.instance_segmentation.state.checkpoint import is_local_done


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _clear_state(global_cfg):
    ckpt = global_cfg["checkpoint"]["fragments_dir"]
    removed = 0
    if os.path.isdir(ckpt):
        for fn in os.listdir(ckpt):
            if fn.endswith((".done", ".json")):
                os.remove(os.path.join(ckpt, fn))
                removed += 1
    print(f"[INFO] restart: cleared {removed} checkpoint/metadata files ({ckpt})")


def _pending(global_cfg, stage_cfg, restart=False):
    aff_vol = CloudVolume(global_cfg["paths"]["input"], mip=stage_cfg.get("mip", 0),
                          bounded=False, progress=False)
    blocks, _ = build_grid(global_cfg, aff_vol)
    ckpt = global_cfg["checkpoint"]["fragments_dir"]
    os.makedirs(ckpt, exist_ok=True)
    return [i for i in range(len(blocks)) if restart or not is_local_done(ckpt, i)]


def _write_manifest(job_dir, indices, blocks_per_job):
    _ensure_dir(job_dir)
    groups = [indices[i:i + blocks_per_job]
              for i in range(0, len(indices), blocks_per_job)]
    manifest = os.path.join(job_dir, "manifest.txt")
    with open(manifest, "w") as f:
        for g in groups:
            f.write(",".join(map(str, g)) + "\n")
    return manifest, len(groups)


def _slurm_script(stage_cfg, job_dir, array_len, job_name, worker_module,
                  config_path=None):
    hpc = stage_cfg["hpc"]
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={hpc.get('time', '04:00:00')}",
        "#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={hpc.get('cpus', 2)}",
        f"#SBATCH --mem-per-cpu={hpc.get('mem', '16G')}",
        # default to running every task at once: a missing hpc_num used to mean
        # "%1", which silently serialised a 12-task array one job at a time
        f"#SBATCH --array=0-{array_len - 1}%{hpc.get('hpc_num', array_len)}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
    ]
    if hpc.get("partition"):
        lines.append(f"#SBATCH --partition={hpc['partition']}")
    if hpc.get("gres"):
        lines.append(f"#SBATCH --gres={hpc['gres']}")
    for m in hpc.get("extra_modules", []):
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append("export SLURM_EXPORT_ENV=ALL")
    if hpc.get("conda"):
        lines.append(f"source {hpc['conda']}")
    if hpc.get("env"):
        lines.append(f"conda activate {hpc['env']}")
    if hpc.get("work_path"):
        lines.append(f"cd {hpc['work_path']}")

    manifest = os.path.join(job_dir, "manifest.txt")
    cfg_path = config_path
    if cfg_path is None:
        global_cfgs = load_global_config_path("magneton/config.yaml")
        cfg_path = (global_cfgs.get("instance_segmentation", {})
                    .get("main", "magneton/instance_segmentation/configs/config.yaml"))
    lines += [
        "set -e",
        f'INDICES=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {manifest})',
        'echo "Running shard indices: $INDICES"',
        f"{hpc.get('python_bin', 'python')} -m {worker_module} "
        f"--config {cfg_path} --indices \"$INDICES\" "
        f"--workers {hpc.get('workers_per_job', 2)} --debug",
    ]
    script_path = os.path.join(job_dir, "submit_slurm.sh")
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_array(global_cfg, stage_cfg, pending, job_name, worker_module,
                 default_job_dir, dry_run=False, config_path=None):
    """Shared submit path for the per-core stages (fragments / edges / relabel)."""
    hpc = stage_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print(f"[INFO] {job_name}: hpc.enable=false, submission disabled.")
        return
    if not pending:
        print(f"[INFO] {job_name}: no pending cores.")
        return
    job_dir = hpc.get("job_dir", default_job_dir)
    manifest, n = _write_manifest(job_dir, pending,
                                  int(hpc.get("blocks_per_job", 4)))
    print(f"[INFO] {len(pending)} cores pending, manifest: {manifest} -> {n} tasks")
    script = _slurm_script(stage_cfg, job_dir, n, job_name, worker_module,
                           config_path=config_path)
    print(f"[INFO] Submit command: sbatch {script}")
    if dry_run:
        return
    try:
        out = subprocess.check_output(["sbatch", script], stderr=subprocess.STDOUT)
        print(f"[INFO] Submit Output: {out.decode('utf-8', 'ignore')}")
    except Exception as e:
        print(f"[WARN] Submission failed: {e}")
        print(f"[HINT] Run manually: sbatch {script}")


def fragments_blocks_hpc(global_cfg, stage_cfg, restart=False, dry_run=False,
                         config_path=None):
    if restart:
        _clear_state(global_cfg)
    pending = _pending(global_cfg, stage_cfg, restart=restart)
    submit_array(global_cfg, stage_cfg, pending, "fragments",
                 "magneton.instance_segmentation.tools.run_fragments_shard",
                 "magneton/jobs/fragments", dry_run=dry_run, config_path=config_path)
