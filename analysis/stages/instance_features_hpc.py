"""
SLURM array submitter for stage B (per-mito feature math).

Mirrors `magneton/instance_segmentation/stages/merge_apply_hpc.py:_slurm_script`.
Reads bboxes.parquet to learn N, partitions into row-ranges of
`instance_stage.mitos_per_task`, writes manifest.txt + submit_slurm.sh, sbatches.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pandas as pd

from magneton.analysis.config import (
    load_config,
    get_stage_config,
    load_global_config_path,
    strip_file_prefix,
)
from magneton.analysis.lib.manifest import make_ranges, write_manifest


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _slurm_script(cfg_path_in_root: str, job_dir: str, num_tasks: int, hpc: dict) -> str:
    python_bin = hpc.get("python_bin", "python")
    time_      = hpc.get("time", "01:00:00")
    mem        = hpc.get("mem", "4G")
    cpus       = hpc.get("cpus", 4)
    partition  = hpc.get("partition", None)
    qos        = hpc.get("qos", None)
    extra_mods = hpc.get("extra_modules", [])
    conda      = hpc.get("conda", None)
    env        = hpc.get("env", None)
    work_path  = hpc.get("work_path", None)
    batch_num  = int(hpc.get("batch_num", 16))

    script_path = os.path.join(job_dir, "submit_slurm.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=analysis_instance",
        f"#SBATCH --time={time_}",
        f"#SBATCH --ntasks=1 --nodes=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem-per-cpu={mem}",
        f"#SBATCH --array=0-{num_tasks - 1}%{batch_num}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
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

    manifest_path = os.path.join(job_dir, "manifest.txt")
    lines += [
        "set -e",
        f'RANGE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {manifest_path})',
        f"{python_bin} -m magneton.analysis.stages.instance_features "
        f"--config {cfg_path_in_root} --range \"$RANGE\" --task-id ${{SLURM_ARRAY_TASK_ID}}",
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit(cfg: dict, cfg_path: str = None, dry_run: bool = False) -> str | None:
    """
    Submit the SLURM array. `cfg_path` MUST be the same YAML path the caller
    loaded `cfg` from — it's embedded into the generated submit_slurm.sh so
    array tasks load the same per-volume config. Defaults to the root
    `magneton/config.yaml` `analysis.main:` pointer only as a last resort.
    """
    paths    = get_stage_config(cfg, "paths")
    stage    = get_stage_config(cfg, "instance")
    out_dir  = strip_file_prefix(paths["output"])
    hpc      = stage.get("hpc", {})
    if not hpc.get("enable", True):
        print("[analysis.instance_hpc] instance_stage.hpc.enable=false; refusing to submit")
        return None

    mitos_per_task = int(stage.get("mitos_per_task", 100))
    job_dir = hpc.get("job_dir", "magneton/jobs/analysis")

    bboxes_path = os.path.join(out_dir, "bboxes.parquet")
    if not os.path.isfile(bboxes_path):
        raise FileNotFoundError(
            f"bboxes.parquet not found at {bboxes_path}; run the discover stage first"
        )
    n = len(pd.read_parquet(bboxes_path, columns=["seg_id"]))
    if n == 0:
        print("[analysis.instance_hpc] bboxes.parquet has 0 rows; nothing to do")
        return None

    ranges = make_ranges(n, mitos_per_task)
    print(f"[analysis.instance_hpc] {n} mitos -> {len(ranges)} SLURM array tasks "
          f"(mitos_per_task={mitos_per_task}, batch_num={hpc.get('batch_num', 16)})")

    _ensure_dir(job_dir)
    manifest_path = os.path.join(job_dir, "manifest.txt")
    write_manifest(manifest_path, ranges)
    print(f"[analysis.instance_hpc] wrote manifest -> {manifest_path}")

    # Use the actual config path the caller passed via --config; fall back to
    # the root config's analysis.main pointer only if not provided. This is
    # the fix for the 2026-05-28 bug where two volumes' arrays both got the
    # root pointer baked in and trampled each other's output.
    if cfg_path:
        cfg_path_for_script = cfg_path
    else:
        global_cfgs = load_global_config_path("magneton/config.yaml")
        cfg_path_for_script = (
            global_cfgs.get("analysis", {})
            .get("main", "magneton/analysis/configs/config.yaml")
        )

    script = _slurm_script(cfg_path_for_script, job_dir, len(ranges), hpc)
    print(f"[analysis.instance_hpc] generated -> {script}")
    print(f"[analysis.instance_hpc] submit command: sbatch {script}")
    if dry_run:
        return script
    try:
        out = subprocess.check_output(["sbatch", script], stderr=subprocess.STDOUT)
        print(f"[analysis.instance_hpc] sbatch output: {out.decode().strip()}")
    except Exception as e:
        print(f"[analysis.instance_hpc][WARN] submission failed: {e}")
        print(f"[HINT] you can submit manually: sbatch {script}")
    return script


def main():
    import argparse
    ap = argparse.ArgumentParser(description="analysis stage B — submit SLURM array")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    submit(cfg, cfg_path=args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
