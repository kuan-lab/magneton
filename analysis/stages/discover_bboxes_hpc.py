"""
SLURM submitter for stage A (discover bboxes).

The discover stage does `read_full(mip) + scipy.ndimage.find_objects`, which
loads the whole discover-mip volume into RAM. At mip-2 that's small (~GB, fine
on a login node), but at mip-1/mip-0 it's tens-to-hundreds of GB — too big for
login. This submits a single 1-task, big-memory job so discover runs on a
compute node (a 480G `day` node holds the ~150G mip-0 array → full completeness).

Optionally chains stages B+C: after discover writes bboxes.parquet on the node,
the job runs `--stage submit-bc`, which sbatches the instance array + queues
reduce. That lets `all-hpc` run end-to-end on the cluster without the
chicken-and-egg of needing bboxes.parquet to size the array at submit time.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from magneton.analysis.config import load_config, get_stage_config


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _slurm_script(cfg_path: str, job_dir: str, hpc: dict, chain_bc: bool) -> str:
    python_bin = hpc.get("python_bin", "python")
    time_      = hpc.get("time", "01:00:00")
    mem        = hpc.get("mem", "240G")        # TOTAL (not per-cpu) — discover is one big-mem task
    cpus       = hpc.get("cpus", 2)
    partition  = hpc.get("partition", None)
    qos        = hpc.get("qos", None)
    extra_mods = hpc.get("extra_modules", [])
    conda      = hpc.get("conda", None)
    env        = hpc.get("env", None)
    work_path  = hpc.get("work_path", None)

    script_path = os.path.join(job_dir, "submit_discover.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=analysis_discover",
        f"#SBATCH --time={time_}",
        "#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
    ]
    if partition: lines.append(f"#SBATCH --partition={partition}")
    if qos:       lines.append(f"#SBATCH --qos={qos}")

    for m in extra_mods:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append("export SLURM_EXPORT_ENV=ALL")
    if conda:     lines.append(f"source {conda}")
    if env:       lines.append(f"conda activate {env}")
    if work_path: lines.append(f"cd {work_path}")

    lines.append("export PYTHONUNBUFFERED=1")   # stream discover's phase prints to the .out live (else block-buffered → nothing until exit)
    lines.append("set -e")
    lines.append(f"{python_bin} -m magneton.analysis.stages.discover_bboxes --config {cfg_path}")
    if chain_bc:
        # bboxes.parquet now exists on disk → safe to size + submit the array + queue reduce
        lines.append(f"{python_bin} -m magneton.analysis.main --stage submit-bc --config {cfg_path}")

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit(cfg: dict, cfg_path: str = None, dry_run: bool = False, chain_bc: bool = False) -> str | None:
    """Submit the discover stage as a big-mem SLURM job. Returns the job id
    (or the script path if dry_run). chain_bc=True makes the job also submit
    stages B+C after discover completes (used by all-hpc)."""
    hpc = get_stage_config(cfg, "discover").get("hpc", {})
    if not hpc.get("enable", True):
        print("[analysis.discover_hpc] discover_stage.hpc.enable=false; refusing to submit")
        return None
    job_dir = hpc.get("job_dir", "magneton/jobs/analysis_discover")
    _ensure_dir(job_dir)

    script = _slurm_script(cfg_path, job_dir, hpc, chain_bc)
    print(f"[analysis.discover_hpc] generated -> {script}")
    print(f"[analysis.discover_hpc] submit command: sbatch {script}")
    if dry_run:
        return script
    try:
        out = subprocess.check_output(["sbatch", "--parsable", script], stderr=subprocess.STDOUT)
        jobid = out.decode().strip().split(";")[0]
        print(f"[analysis.discover_hpc] submitted discover as job {jobid}"
              + (" (chains stages B+C on completion)" if chain_bc else ""))
        return jobid
    except Exception as e:
        print(f"[analysis.discover_hpc][WARN] submission failed: {e}")
        print(f"[HINT] you can submit manually: sbatch {script}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="analysis stage A — submit discover as a big-mem SLURM job")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chain-bc", action="store_true", help="also submit stages B+C after discover")
    args = ap.parse_args()
    cfg = load_config(args.config)
    submit(cfg, cfg_path=args.config, dry_run=args.dry_run, chain_bc=args.chain_bc)


if __name__ == "__main__":
    main()
