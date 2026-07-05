# -*- coding: utf-8 -*-
"""
merge-supervox HPC submitter.

merge_supervox is a map-REDUCE whose per-block passes (write cores, seam edges,
relabel cores) are parallelized ACROSS THE NODE'S CORES in one process
(ProcessPoolExecutor), while the dense-id reduce is small vectorized numpy. So it
runs as a SINGLE node job with many CPUs — NOT a SLURM array (an array would add
per-phase queue waits + a dependency chain around the unavoidable single reduce,
which for ~100-block volumes costs more than it saves; revisit only past ~1000
heavy blocks). This submitter requests `--cpus-per-task=cpus`; the stage reads
`merge_stage.supervox_workers` (default = cpu count) to size the pool.

Config: `merge_stage.hpc_supervox` (falls back to `merge_stage.hpc`). Give it enough
`mem` — the global RAG/positions/agg arrays scale with the supervoxel count.
"""
import os
import subprocess
from pathlib import Path

from magneton.instance_segmentation.config import load_global_config_path


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _slurm_script(stage_cfg, job_dir):
    hpc = stage_cfg.get("hpc_supervox") or stage_cfg.get("hpc", {})
    python_bin = hpc.get("python_bin", "python")
    time = hpc.get("time", "12:00:00")
    mem = hpc.get("mem", "8G")            # per-cpu
    cpus = hpc.get("cpus", "8")
    partition = hpc.get("partition", None)
    gres = hpc.get("gres", None)      # e.g. "gpu:1" to borrow CPU cores on a GPU-partition node
    extra_modules = hpc.get("extra_modules", [])
    conda = hpc.get("conda", None)
    env = hpc.get("env", None)
    work_path = hpc.get("work_path", None)

    script_path = os.path.join(job_dir, "submit_slurm.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=merge_supervox",
        f"#SBATCH --time={time}",
        "#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem-per-cpu={mem}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
    ]
    if partition:   lines.append(f"#SBATCH --partition={partition}")
    if gres:        lines.append(f"#SBATCH --gres={gres}")

    for m in extra_modules:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append("export SLURM_EXPORT_ENV=ALL")
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
        f"{python_bin} -m magneton.instance_segmentation.stages.merge_supervox "
        f"--config {cfg_path}",
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def merge_supervox_hpc(global_cfg, stage_cfg, restart=False, dry_run=False):
    """Submit merge_supervox as a single multi-core SLURM node job."""
    hpc = stage_cfg.get("hpc_supervox") or stage_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] merge_stage hpc.enable=false; HPC submission disabled.")
        return
    if hpc.get("scheduler", "slurm").lower() != "slurm":
        raise ValueError(f"Unknown scheduler: {hpc.get('scheduler')}")

    job_dir = hpc.get("job_dir_supervox") or os.path.join(
        hpc.get("job_dir", "magneton/jobs/merge"), "supervox")
    script_path = _slurm_script(stage_cfg, job_dir)
    submit_cmd = ["sbatch", script_path]
    print(f"[INFO] Submit command: {' '.join(submit_cmd)}")
    if not dry_run:
        try:
            out = subprocess.check_output(submit_cmd, stderr=subprocess.STDOUT)
            print(f"[INFO] Submit Output: {out.decode('utf-8', 'ignore')}")
        except Exception as e:
            print(f"[WARN] Submission failed: {e}")
            print(f"[HINT] Manually run: {' '.join(submit_cmd)}")
