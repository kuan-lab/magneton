#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proofreading module — argparse CLI + interactive (Rich) menu.

Skeleton-driven proofreading / GT-bootstrap loop:
  Stage A  skeletonize   Instance seg -> skeletons.nml (kimimaro).
                         [ upload via /wk, correct in WebKnossos, download NML ]
  Stage B  expand        Corrected NML + EM -> dense segments (nnInteractive).
  Stage B' expand-hpc    Submit stage B as a 1-GPU SLURM job (the usual path).
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from magneton.proofreading.config import load_config, load_global_config_path

console = Console()


def _resolve_cfg_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    root = load_global_config_path("magneton/config.yaml")
    return root.get("proofreading", {}).get(
        "main", "magneton/proofreading/configs/config_fib_b_neuron_fennel.yaml")


def _run_skeletonize(cfg_path: str):
    from magneton.proofreading.stages.skeletonize import skeletonize
    skeletonize(load_config(cfg_path))


def _run_expand(cfg_path: str):
    from magneton.proofreading.stages.expand import expand
    expand(load_config(cfg_path))


def _run_expand_hpc(cfg_path: str, dry_run: bool):
    from magneton.proofreading.stages.expand_hpc import submit
    submit(load_config(cfg_path), cfg_path=cfg_path, dry_run=dry_run)


def run(args, global_cfg=None):
    cfg_path = _resolve_cfg_path(getattr(args, "config", None))
    stage = args.stage
    if stage == "skeletonize":
        _run_skeletonize(cfg_path)
    elif stage == "expand":
        _run_expand(cfg_path)
    elif stage == "expand-hpc":
        _run_expand_hpc(cfg_path, getattr(args, "dry_run", False))
    else:
        raise SystemExit(f"unknown stage: {stage}")


def _build_parser():
    p = argparse.ArgumentParser(prog="magneton.proofreading.main",
                                description="skeleton-driven proofreading pipeline")
    p.add_argument("--stage", choices=["skeletonize", "expand", "expand-hpc"],
                   required=False, help="pipeline stage to run")
    p.add_argument("--config", required=False, default=None, help="per-volume YAML config")
    p.add_argument("--dry-run", action="store_true", help="expand-hpc: generate script, don't sbatch")
    return p


def _menu_table():
    t = Table(show_header=True, box=box.SIMPLE, border_style="white",
              header_style="bright_white")
    t.add_column("Option", justify="center", style="white")
    t.add_column("Function", style="white")
    t.add_column("Description", style="white")
    t.add_row("1", "Skeletonize",          "Instance seg -> skeletons.nml (kimimaro)")
    t.add_row("2", "Expand (nnInteractive)", "Corrected NML + EM -> dense segments (needs GPU)")
    t.add_row("3", "Expand [HPC]",          "Submit expand as a 1-GPU SLURM job")
    t.add_row("9", "View Current Config",   "Print the resolved proofreading config")
    t.add_row("0", "Return",                "Back to magneton main menu")
    return t


def run_interactive():
    cfg_path = _resolve_cfg_path(None)
    pool = ["0", "1", "2", "3", "9"]
    while True:
        console.rule("[bold bright_white]Proofreading Menu[/bold bright_white]", style="bold white")
        console.print(f"[white] Config:[/white] {cfg_path}")
        console.print(_menu_table())
        choice = Prompt.ask("[bright_white]> Select stage[/bright_white]", default="0").strip()
        if choice not in pool:
            console.print("[red]Invalid selection.[/red]")
            continue
        if choice == "0":
            console.print("[yellow]Exit Proofreading.[/yellow]")
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
        a.dry_run = False
        a.stage = {"1": "skeletonize", "2": "expand", "3": "expand-hpc"}[choice]
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
