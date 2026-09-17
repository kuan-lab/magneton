#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instance segmentation — graph pipeline (non-overlapping cores + region graph).

Four passes, each its own run over the data:

  1 fragments     watershed per core (halo = context only), crop to the core,
                  globally unique ids -> ONE supervoxel volume
  2 edges         per core, waterz over core+halo on those fragments -> region
                  graph edges (incl. pairs meeting across a core face), scored
                  by the same merge function used inside a core
  3 global merge  concat edges -> threshold + min contact -> connected
                  components -> LUT fragment id -> segment id
  4 relabel       apply the LUT per core -> final segmentation

No block ever decides that two fragments are one cell; every merge is made once,
globally, in pass 3. The legacy overlap-vote pipeline lives in legacy_main.py.
"""
import argparse
import logging
import os
import shutil
import time

import yaml
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box

console = Console()

from magneton.instance_segmentation.config import (
    load_config,
    get_stage_config,
    load_global_config_path,
)
from magneton.instance_segmentation.legacy_main import (
    edit_stage_config,
    load_global_config,
    modify_global_config,
)
from magneton.instance_segmentation.stages.fragments_stage import fragments_blocks
from magneton.instance_segmentation.stages.fragments_stage_hpc import fragments_blocks_hpc
from magneton.instance_segmentation.stages.edges_stage import edges_blocks
from magneton.instance_segmentation.stages.edges_stage_hpc import (
    edges_blocks_hpc,
    relabel_blocks_hpc,
)
from magneton.instance_segmentation.stages.global_merge import global_merge
from magneton.instance_segmentation.stages.relabel_stage import relabel_blocks
from magneton.instance_segmentation.stages.segment_props_stage import segment_properties
from magneton.instance_segmentation.utils.interrupts import InterruptController

STAGES = [
    "fragments", "fragments-hpc",
    "edges", "edges-hpc",
    "global-merge",
    "relabel", "relabel-hpc",
    "segment-props",
    "status", "clean",
]


def _seg_cfg_path(global_cfg):
    return (global_cfg.get("instance_segmentation", {})
            .get("main", "magneton/instance_segmentation/configs/config.yaml"))


def _resolve_lut(cfg, stage_cfg):
    """Explicit lut_path, else the newest LUT in lut_dir."""
    if stage_cfg.get("lut_path"):
        return stage_cfg["lut_path"]
    lut_dir = (cfg.get("global_merge_stage", {})
               .get("lut_dir", "magneton/checkpoints/lut"))
    if not os.path.isdir(lut_dir):
        raise FileNotFoundError(f"no LUT directory {lut_dir}; run global-merge first")
    luts = [os.path.join(lut_dir, f) for f in os.listdir(lut_dir) if f.endswith(".npz")]
    if not luts:
        raise FileNotFoundError(f"no LUT in {lut_dir}; run global-merge first")
    newest = max(luts, key=os.path.getmtime)
    print(f"[INFO] using LUT {newest}")
    return newest


def _status(cfg):
    frag_ckpt = cfg["checkpoint"]["fragments_dir"]
    relabel_ckpt = cfg["checkpoint"].get("relabel_dir", "")
    edges_dir = cfg.get("edges_stage", {}).get("edges_dir", "")
    lut_dir = cfg.get("global_merge_stage", {}).get("lut_dir", "")

    def count(d, suffix):
        return (len([f for f in os.listdir(d) if f.endswith(suffix)])
                if d and os.path.isdir(d) else 0)

    t = Table(box=box.SIMPLE, header_style="bright_white")
    t.add_column("Pass"); t.add_column("Artifact"); t.add_column("Count")
    t.add_row("1 fragments", frag_ckpt, str(count(frag_ckpt, ".done")))
    t.add_row("2 edges", edges_dir, str(count(edges_dir, ".npz")))
    t.add_row("3 global merge", lut_dir, str(count(lut_dir, ".npz")))
    t.add_row("4 relabel", relabel_ckpt, str(count(relabel_ckpt, ".done")))
    t.add_row("  (counts)", relabel_ckpt, str(count(relabel_ckpt, ".npz")))
    props = os.path.join(cfg["paths"]["output"].replace("file://", ""), "segment_properties")
    t.add_row("5 segment props", props, "written" if os.path.exists(os.path.join(props, "info")) else "-")
    console.print(t)


def run(args, global_cfg):
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    seg_cfg_path = _seg_cfg_path(global_cfg)
    workers = int(getattr(args, "workers", 1) or 1)
    restart = bool(getattr(args, "restart", False))

    def confirm(stage_name):
        print(f"\nStarting stage: {stage_name}")
        print("Press Enter to continue or type 'q' to cancel.")
        if input("> ").strip().lower() == "q":
            console.print(f"[bold red]▶ Canceled stage: {stage_name}.[/bold red]\n")
            return False
        return True

    def pause():
        print("Press Enter to return menu.")
        input("> ")

    try:
        if args.stage == "fragments":
            if not confirm("Pass 1 - Fragments"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Fragments Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                fragments_blocks(cfg, get_stage_config(cfg, "fragments"),
                                 restart=restart, workers=workers)
            pause()

        elif args.stage == "fragments-hpc":
            if not confirm("Pass 1 - Fragments [HPC]"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Fragments Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                fragments_blocks_hpc(cfg, get_stage_config(cfg, "fragments"),
                                     restart=restart, config_path=cfg_path)
            pause()

        elif args.stage == "edges":
            if not confirm("Pass 2 - Edges"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Edges Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                edges_blocks(cfg, get_stage_config(cfg, "edges"),
                             restart=restart, workers=workers)
            pause()

        elif args.stage == "edges-hpc":
            if not confirm("Pass 2 - Edges [HPC]"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Edges Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                edges_blocks_hpc(cfg, get_stage_config(cfg, "edges"),
                                 restart=restart, config_path=cfg_path)
            pause()

        elif args.stage == "global-merge":
            if not confirm("Pass 3 - Global Merge"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Global Merge Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                global_merge(cfg, get_stage_config(cfg, "global_merge"))
            pause()

        elif args.stage == "relabel":
            if not confirm("Pass 4 - Relabel"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Relabel Stage")
            cfg = load_config(cfg_path)
            stage_cfg = get_stage_config(cfg, "relabel")
            lut = _resolve_lut(cfg, stage_cfg)
            with InterruptController():
                relabel_blocks(cfg, stage_cfg, lut, restart=restart, workers=workers)
            pause()

        elif args.stage == "relabel-hpc":
            if not confirm("Pass 4 - Relabel [HPC]"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Relabel Stage")
            cfg = load_config(cfg_path)
            stage_cfg = get_stage_config(cfg, "relabel")
            lut = _resolve_lut(cfg, stage_cfg)
            with InterruptController():
                relabel_blocks_hpc(cfg, stage_cfg, lut, restart=restart,
                                   config_path=cfg_path)
            pause()

        elif args.stage == "segment-props":
            if not confirm("Pass 5 - Segment Properties"):
                return
            cfg_path = edit_stage_config(seg_cfg_path, "Segment Properties Stage")
            cfg = load_config(cfg_path)
            with InterruptController():
                segment_properties(cfg, get_stage_config(cfg, "segment_props"))
            pause()

        elif args.stage == "status":
            _status(load_config(seg_cfg_path))
            pause()

        elif args.stage == "clean":
            if not confirm("Clean graph-pipeline state"):
                return
            cfg = load_config(seg_cfg_path)
            targets = [cfg["checkpoint"]["fragments_dir"],
                       cfg["checkpoint"].get("relabel_dir"),
                       cfg.get("edges_stage", {}).get("edges_dir"),
                       cfg.get("global_merge_stage", {}).get("lut_dir")]
            console.print("[yellow]This removes checkpoints, edges and LUTs "
                          "(NOT the fragments/output volumes):[/yellow]")
            for p in targets:
                console.print(f"  {p}")
            if Prompt.ask("[white]> Proceed? (y/n)[/white]", default="n").lower().startswith("y"):
                for p in targets:
                    if p and os.path.exists(p):
                        shutil.rmtree(p)
                        print(f"[INFO] Cleaned: {p}")
            pause()

        console.print(f"[bold green]▶ Stage {args.stage} completed.[/bold green]\n")

    except KeyboardInterrupt:
        print("\nExecution interrupted abruptly by user.")
    finally:
        logging.shutdown()


def run_interactive():
    console.print("\n[bold bright_white] Instance Segmentation — graph pipeline"
                  "[/bold bright_white]\n")
    cfg_path = "magneton/config.yaml"
    cfg, cfg_path = load_global_config(cfg_path)
    mapping = {"1": "fragments", "2": "fragments-hpc", "3": "edges", "4": "edges-hpc",
               "5": "global-merge", "6": "relabel", "7": "relabel-hpc",
               "8": "segment-props", "9": "status", "10": "clean"}

    while True:
        console.rule("[bold bright_white]Instance Segmentation Menu[/bold bright_white]",
                     style="bold white")
        t = Table(show_header=True, box=box.SIMPLE, border_style="white",
                  header_style="bright_white")
        t.add_column("Option", justify="center", style="white")
        t.add_column("Function", style="white")
        t.add_column("Description", style="white")
        t.add_row("1", "Pass 1 - Fragments", "Watershed supervoxels on non-overlapping cores")
        t.add_row("2", "Pass 1 - Fragments [HPC]", "Same, as a SLURM array")
        t.add_row("3", "Pass 2 - Edges", "Region-graph edges (incl. across core faces)")
        t.add_row("4", "Pass 2 - Edges [HPC]", "Same, as a SLURM array")
        t.add_row("5", "Pass 3 - Global Merge", "Threshold the graph -> LUT (fast, sweepable)")
        t.add_row("6", "Pass 4 - Relabel", "Apply the LUT -> final segmentation")
        t.add_row("7", "Pass 4 - Relabel [HPC]", "Same, as a SLURM array")
        t.add_row("8", "Pass 5 - Segment Properties", "Merge per-core counts -> Neuroglancer segment list")
        t.add_row("9", "Status", "Per-pass artifact counts")
        t.add_row("10", "Clean", "Remove checkpoints / edges / LUTs")
        t.add_row("11", "Modify Global Config", "Edit magneton/config.yaml")
        t.add_row("0", "Return", "Return to main menu")
        console.print(t)

        choice = Prompt.ask("[bright_white]> Select stage[/bright_white]",
                            default="0").strip().lower()
        if choice == "0":
            console.print("[yellow]Exit Instance Segmentation Pipeline.[/yellow]")
            break
        if choice == "11":
            cfg, cfg_path = modify_global_config(cfg, cfg_path)
            continue
        if choice not in mapping:
            console.print("[red]Invalid selection. Try again.[/red]")
            continue

        cfg, cfg_path = load_global_config(cfg_path)

        class Args:
            pass

        args = Args()
        args.stage = mapping[choice]
        args.debug = False
        args.restart = False
        args.workers = 1
        if args.stage in ("fragments", "edges", "relabel"):
            args.workers = int(Prompt.ask("[white]> Local workers[/white]", default="1"))
        if args.stage in ("fragments", "fragments-hpc", "edges", "edges-hpc",
                          "relabel", "relabel-hpc"):
            args.restart = Prompt.ask("[white]> Restart? (y/n)[/white]",
                                      default="n").lower().startswith("y")

        console.print(f"\n[green]▶ Executing stage:[/] [cyan]{args.stage}[/cyan]")
        run(args, cfg)


def main():
    t1 = time.time()
    parser = argparse.ArgumentParser(description="Graph-based instance segmentation")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(args, load_global_config_path("magneton/config.yaml"))
    print(f"Total runtime: {time.time() - t1:.2f}s")


if __name__ == "__main__":
    main()
