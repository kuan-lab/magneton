"""
SLURM submitter for stage B (nnInteractive expand) — a single GPU job.

nnInteractive needs a GPU (~10GB VRAM); login nodes have none. This generates a
1-GPU sbatch that activates the nnInteractive env and runs the expand stage.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from magneton.proofreading.config import load_config, get_stage_config


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _slurm_script(cfg_path: str, job_dir: str, hpc: dict) -> str:
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)
    script_path = os.path.join(job_dir, "submit_expand.sh")

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=proofread_expand",
        f"#SBATCH --time={hpc.get('time', '01:00:00')}",
        "#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={hpc.get('cpus', 8)}",
        f"#SBATCH --mem-per-gpu={hpc.get('mem_per_gpu', '64G')}",
        f"#SBATCH --gpus={hpc.get('gpus', 1)}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --partition={hpc.get('partition', 'gpu')}",
    ]
    if hpc.get("constraint"):
        lines.append(f"#SBATCH --constraint=\"{hpc['constraint']}\"")
    lines += ["module load StdEnv", "export SLURM_EXPORT_ENV=ALL",
              "export PYTHONUNBUFFERED=1"]
    if hpc.get("hf_cache"):
        lines.append(f"export HF_HUB_CACHE={hpc['hf_cache']}")
    if hpc.get("conda"):
        lines.append(f"source {hpc['conda']}")
    if hpc.get("env"):
        lines.append(f"conda activate {hpc['env']}")
    if hpc.get("work_path"):
        lines.append(f"cd {hpc['work_path']}")
    lines.append("set -e")
    lines.append(f"python -u -m magneton.proofreading.stages.expand --config {cfg_path}")

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit(cfg: dict, cfg_path: str = None, dry_run: bool = False):
    hpc = get_stage_config(cfg, "expand").get("hpc", {})
    if not hpc.get("enable", True):
        print("[proofreading.expand_hpc] expand_stage.hpc.enable=false; refusing to submit")
        return None
    # the HF cache should be reachable inside the job too
    hpc = {**hpc, "hf_cache": get_stage_config(cfg, "expand").get("hf_cache", hpc.get("hf_cache"))}
    job_dir = hpc.get("job_dir", "magneton/jobs/proofreading_expand")
    _ensure_dir(job_dir)
    script = _slurm_script(cfg_path, job_dir, hpc)
    print(f"[proofreading.expand_hpc] generated -> {script}")
    print(f"[proofreading.expand_hpc] submit command: sbatch {script}")
    if dry_run:
        return script
    try:
        out = subprocess.check_output(["sbatch", "--parsable", script], stderr=subprocess.STDOUT)
        jobid = out.decode().strip().split(";")[0]
        print(f"[proofreading.expand_hpc] submitted expand as job {jobid}")
        return jobid
    except Exception as e:
        print(f"[proofreading.expand_hpc][WARN] submission failed: {e}")
        print(f"[HINT] submit manually: sbatch {script}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading stage B — submit GPU expand job")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    submit(cfg, cfg_path=args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
