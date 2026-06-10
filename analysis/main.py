#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis module — argparse CLI + interactive (Rich) menu.

Per-instance mito morphometrics pipeline:
  Stage A  discover       Read high-mip volume, find each mito's bbox.
  Stage B  instance       Per-mito feature math (single-process; debug).
  Stage B' instance-hpc   Submit SLURM array for Stage B.
  Stage C  reduce         Concat task shards into morphometrics.parquet.
  all-hpc                 Run discover + submit array + queue reduce.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from magneton.analysis.config import (
    load_config,
    load_global_config_path,
)


console = Console()


# ----------------------------------------------------------------- dispatch ---

def _resolve_cfg_path(cli_path: str | None) -> str:
    """Resolve --config from CLI or from root config.yaml's analysis.main pointer."""
    if cli_path:
        return cli_path
    root = load_global_config_path("magneton/config.yaml")
    return root.get("analysis", {}).get("main", "magneton/analysis/configs/config.yaml")


def _run_discover(cfg_path: str):
    from magneton.analysis.stages.discover_bboxes import discover_bboxes
    cfg = load_config(cfg_path)
    discover_bboxes(cfg)


def _run_instance(cfg_path: str, rng: str | None, task_id: int):
    from magneton.analysis.stages.instance_features import process_range
    cfg = load_config(cfg_path)
    if rng is None:
        # Default: process every row in bboxes.parquet (single-task debug run)
        import pandas as pd
        from magneton.analysis.config import get_stage_config, strip_file_prefix
        out_dir = strip_file_prefix(get_stage_config(cfg, "paths")["output"])
        n = len(pd.read_parquet(os.path.join(out_dir, "bboxes.parquet"), columns=["seg_id"]))
        start, end = 0, n
    else:
        start, end = (int(x) for x in rng.split(","))
    process_range(cfg, start, end, task_id)


def _run_instance_hpc(cfg_path: str, dry_run: bool):
    from magneton.analysis.stages.instance_features_hpc import submit
    cfg = load_config(cfg_path)
    submit(cfg, cfg_path=cfg_path, dry_run=dry_run)


def _run_reduce(cfg_path: str):
    from magneton.analysis.stages.reduce_features import reduce_features
    cfg = load_config(cfg_path)
    reduce_features(cfg)


def _run_cluster(cfg_path: str):
    from magneton.analysis.stages.cluster import cluster
    cfg = load_config(cfg_path)
    cluster(cfg)


def _run_discover_hpc(cfg_path: str, dry_run: bool):
    """Submit the discover stage as a big-mem SLURM job (for mip-1/mip-0 where
    read_full+find_objects won't fit on login)."""
    from magneton.analysis.stages.discover_bboxes_hpc import submit
    cfg = load_config(cfg_path)
    submit(cfg, cfg_path=cfg_path, dry_run=dry_run, chain_bc=False)


def _run_all_hpc(cfg_path: str):
    """Submit discover as a big-mem SLURM job that, on completion, chains stages
    B+C (instance array + reduce). Runs end-to-end on the cluster — safe for
    mip-1/mip-0 discover that can't run on login."""
    from magneton.analysis.stages.discover_bboxes_hpc import submit
    cfg = load_config(cfg_path)
    submit(cfg, cfg_path=cfg_path, dry_run=False, chain_bc=True)


def _run_match(cfg_path: str):
    """Relational stage 1 — match instances across volumes (uses a relational config)."""
    from magneton.analysis.stages.match_stage import run_matching
    cfg = load_config(cfg_path)
    run_matching(cfg)


def _run_relational(cfg_path: str):
    """Relational stage 2 — stats + plots from the match tables + morphometrics."""
    from magneton.analysis.stages.relational import relational
    cfg = load_config(cfg_path)
    relational(cfg)


def _submit_bc(cfg_path: str):
    """Submit stage B (instance array) + queue stage C (reduce, afterok).
    Assumes bboxes.parquet already exists. Called by the discover SLURM job
    (chain_bc) after discover finishes, or manually."""
    from magneton.analysis.stages.instance_features_hpc import submit
    cfg = load_config(cfg_path)
    script = submit(cfg, cfg_path=cfg_path, dry_run=True)   # generate but don't sbatch yet
    if script is None:
        return
    try:
        out = subprocess.check_output(["sbatch", "--parsable", script], stderr=subprocess.STDOUT)
        array_jobid = out.decode().strip().split(";")[0]
        console.print(f"[bold green][analysis.all][/bold green] stage B submitted as job {array_jobid}")
    except Exception as e:
        console.print(f"[red][analysis.all] failed to sbatch stage B: {e}[/red]")
        console.print(f"[yellow]Manual: sbatch {script}[/yellow]")
        return
    # Stage C as a dependent 1-cpu job
    # (Inline shell here keeps things simple; could become a stages/reduce_features_hpc.py later.)
    reduce_script = os.path.join(os.path.dirname(script), "submit_reduce.sh")
    # Inherit partition + conda from the instance_stage.hpc section so the reduce
    # job lands on the same cluster as the array.
    hpc = cfg.get("instance_stage", {}).get("hpc", {})
    partition = hpc.get("partition", "day")
    with open(reduce_script, "w") as f:
        f.write(
            "#!/bin/bash\n"
            "#SBATCH --job-name=analysis_reduce\n"
            "#SBATCH --time=00:10:00\n"
            "#SBATCH --ntasks=1 --nodes=1\n"
            "#SBATCH --cpus-per-task=1\n"
            "#SBATCH --mem=4G\n"
            f"#SBATCH --output={os.path.dirname(script)}/logs/%x_%j.out\n"
            f"#SBATCH --error={os.path.dirname(script)}/logs/%x_%j.err\n"
            f"#SBATCH --partition={partition}\n"
            "module load StdEnv\n"
            "export SLURM_EXPORT_ENV=ALL\n"
        )
        if hpc.get("conda"):
            f.write(f"source {hpc['conda']}\n")
        if hpc.get("env"):
            f.write(f"conda activate {hpc['env']}\n")
        if hpc.get("work_path"):
            f.write(f"cd {hpc['work_path']}\n")
        f.write(f"set -e\n")
        f.write(f"python -m magneton.analysis.stages.reduce_features --config {cfg_path}\n")
    os.chmod(reduce_script, 0o755)
    try:
        out = subprocess.check_output(
            ["sbatch", f"--dependency=afterok:{array_jobid}", reduce_script],
            stderr=subprocess.STDOUT,
        )
        console.print(f"[bold green][analysis.all][/bold green] stage C queued: {out.decode().strip()}")
    except Exception as e:
        console.print(f"[red][analysis.all] failed to sbatch reduce: {e}[/red]")
        console.print(f"[yellow]Manual once array finishes: sbatch {reduce_script}[/yellow]")


# ---------------------------------------------------------- CLI entrypoint ----

def run(args, global_cfg=None):
    cfg_path = _resolve_cfg_path(getattr(args, "config", None))
    stage = args.stage
    if stage == "discover":
        _run_discover(cfg_path)
    elif stage == "instance":
        _run_instance(cfg_path, getattr(args, "range", None), getattr(args, "task_id", 0))
    elif stage == "instance-hpc":
        _run_instance_hpc(cfg_path, getattr(args, "dry_run", False))
    elif stage == "discover-hpc":
        _run_discover_hpc(cfg_path, getattr(args, "dry_run", False))
    elif stage == "submit-bc":
        _submit_bc(cfg_path)
    elif stage == "reduce":
        _run_reduce(cfg_path)
    elif stage == "cluster":
        _run_cluster(cfg_path)
    elif stage == "all-hpc":
        _run_all_hpc(cfg_path)
    elif stage == "match":
        _run_match(cfg_path)
    elif stage == "relational":
        _run_relational(cfg_path)
    elif stage == "relational-all":
        _run_match(cfg_path)
        _run_relational(cfg_path)
    else:
        raise SystemExit(f"unknown stage: {stage}")


def _build_parser():
    p = argparse.ArgumentParser(prog="magneton.analysis.main", description="mito morphometrics pipeline")
    p.add_argument("--stage", choices=["discover", "discover-hpc", "instance", "instance-hpc",
                                       "submit-bc", "reduce", "cluster", "all-hpc",
                                       "match", "relational", "relational-all"],
                   required=False, help="pipeline stage to run")
    p.add_argument("--config", required=False, default=None, help="per-volume YAML config")
    p.add_argument("--range", required=False, default=None, help="row range for stage B: 'start,end'")
    p.add_argument("--task-id", type=int, default=0, help="task id for stage B output file")
    p.add_argument("--dry-run", action="store_true", help="for instance-hpc: don't actually sbatch")
    return p


# --------------------------------------------------------- interactive menu --

def _menu_table():
    t = Table(show_header=True, box=box.SIMPLE, border_style="white",
              title_style="bold bright_white", header_style="bright_white")
    t.add_column("Option", justify="center", style="white")
    t.add_column("Function", style="white")
    t.add_column("Description", style="white")
    t.add_row("1", "Discover Bboxes",                    "Read high-mip volume, find each mito's bbox")
    t.add_row("2", "Discover Bboxes [HPC]",              "Submit discover as a big-mem SLURM job (mip-1/0)")
    t.add_row("3", "Per-Instance Features",              "Run per-mito feature math (single-process, debug)")
    t.add_row("4", "Per-Instance Features [HPC]",        "Submit SLURM array for per-mito features")
    t.add_row("5", "Reduce / Concat",                    "Concat task partials → morphometrics.parquet")
    t.add_row("6", "All [HPC]",                          "Discover[HPC] → array → reduce, end-to-end on cluster")
    t.add_row("7", "Embed (PCA + UMAP)",                 "Z-score features → PCA + UMAP, save embedding + plots")
    t.add_row("8", "Relational (cross-volume)",          "Match mito/synapse → bouton, then relational stats + plots (relational config)")
    t.add_row("9", "View Current Config",                "Print the resolved analysis config")
    t.add_row("0", "Return",                             "Back to magneton main menu")
    return t


def run_interactive():
    """Rich-table interactive sub-menu, mirroring instance_segmentation.run_interactive."""
    cfg_path = _resolve_cfg_path(None)
    choice_pool = [str(i) for i in range(10)]
    while True:
        console.rule("[bold bright_white]Mito Analysis Menu[/bold bright_white]", style="bold white")
        console.print(f"[white] Config:[/white] {cfg_path}")
        console.print(_menu_table())
        choice = Prompt.ask("[bright_white]> Select stage[/bright_white]", default="0").strip()
        if choice not in choice_pool:
            console.print("[red]Invalid selection.[/red]")
            continue

        if choice == "0":
            console.print("[yellow]Exit Mito Analysis.[/yellow]")
            break

        if choice == "9":
            cfg = load_config(cfg_path)
            t = Table(box=box.SIMPLE, header_style="bright_white")
            t.add_column("Section"); t.add_column("Key"); t.add_column("Value")
            for sec, sub in cfg.items():
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        t.add_row(str(sec), str(k), str(v))
                else:
                    t.add_row(str(sec), "-", str(sub))
            console.print(t)
            input("Press Enter to return menu.\n> ")
            continue

        class Args:
            pass
        a = Args()
        a.config = cfg_path
        a.range = None
        a.task_id = 0
        a.dry_run = False
        mapping = {"1": "discover", "2": "discover-hpc", "3": "instance", "4": "instance-hpc",
                   "5": "reduce", "6": "all-hpc", "7": "cluster", "8": "relational-all"}
        a.stage = mapping[choice]
        try:
            run(a)
        except Exception as e:
            console.print(f"[red]Error during {a.stage}: {e}[/red]")
        input("\nPress Enter to return menu.\n> ")


def main():
    parser = _build_parser()
    if len(sys.argv) == 1:
        run_interactive()
        return
    args = parser.parse_args()
    if args.stage is None:
        run_interactive()
        return
    run(args)


if __name__ == "__main__":
    main()
