#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connectomics training/inference module — unified CLI and interactive interface.

Provides:
- main(): command-line interface for training/inference
- run(args, global_cfg): external entrypoint
- run_interactive(): interactive menu
"""

import argparse
import logging
import os
import socket
import yaml
import signal
import time
import torch
import warnings


from .tools.run import run as pytc_run
from .tools.run_hpc import run_hpc as pytc_run_hpc
from .tools.vis import launch_tensorboard
from .utils.interrupts import InterruptController
from .utils.config import load_config

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

console = Console()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# Helper: config file editing (same style)
# ==========================================================

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
    
def edit_stage_config(config_path: str, stage_name: str):
    print(f"\nStage: {stage_name}")
    print(f"Config path: {config_path}")

    if not os.path.exists(config_path):
        print("Config file not found. Skipping modification.")
        return config_path

    if not Prompt.ask("[white]> Modify this stage config before running? (y/n)[/white]", default="n").lower().startswith("y"):
        return config_path

    # Read configuration file
    with open(config_path, "r") as f:
        cfg_data = yaml.safe_load(f)

    # Display Configuration Items (Single Layer)
    flat_keys = []
    flat_sections = []
    print("\nAvailable parameters in config:")
    config_table = Table(
            box=box.SIMPLE,
            title_style="bold bright_white",
            header_style="bright_white",
            show_header=True
        )
    config_table.add_column("Index", justify="center", style="white")
    config_table.add_column("Section", style="white")
    config_table.add_column("Parameter", style="white")
    config_table.add_column("Value", style="white")
    
    i = 1
    for section, sub in cfg_data.items():
        # console.print(f"[bold]{section}[/bold]:")
        
        if isinstance(sub, dict):
            for k, v in sub.items():
                # console.print(f"   - {k}: [cyan]{v}[/cyan]")
                flat_keys.append(k)
                flat_sections.append(section)
                config_table.add_row(f"{i}", f"{section}", f"{k}", f"[cyan]{v}[/cyan]")
                i += 1
        else:
            # console.print(f"   - [cyan]{sub}[/cyan]")
            flat_keys.append(section)
            flat_sections.append("-")
            config_table.add_row(f"{i}", f"{section}", f"{section}",  f"[cyan]{sub}[/cyan]")
            i += 1
        
    console.print(config_table)

    while True:
        choice = input("> Select parameter to modify (number, ENTER to finish): ").strip()
        if choice == "":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(flat_keys)):
            print("Invalid selection.")
            continue

        key = flat_keys[int(choice) - 1]
        section_key = flat_sections[int(choice) - 1]
        print(f"Current value for {section_key}/{key}: {cfg_data[section_key][key]}")
        new_val = input("New value: ").strip()
        if new_val == "":
            print("No change.")
            continue
        try:
            if "." in new_val:
                new_val = float(new_val)
            else:
                new_val = int(new_val)
        except ValueError:
            pass
        cfg_data[section_key][key] = new_val
        print(f"Updated {key} → {new_val}")

    tmp_path = config_path
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg_data, f, sort_keys=False)
    print(f"Modified config updated: {tmp_path}")
    return tmp_path


# ==========================================================
# Unified CLI entrypoint
# ==========================================================
def run(args, global_cfg):
    """Unified CLI-compatible entrypoint with interrupt handling and config editing."""
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    cfg_file_path = global_cfg.get("affinity_prediction", {}).get("config_file", "configs/network.yaml")
    cfg_base_path = global_cfg.get("affinity_prediction", {}).get("config_base", "configs/base.yaml")
    cfg_checkpoint_path = global_cfg.get("affinity_prediction", {}).get("checkpoint", "configs/checkpoint_xxxxx.pth.tar")
    hpc_cfg_path = global_cfg.get("affinity_prediction", {}).get("hpc", "configs/hpc.yaml")
    

    # Confirm stage
    def confirm_stage(stage_name):
        print(f"\nStarting stage: {stage_name}")
        print("Press Enter to continue or type 'q' to cancel.")
        resp = input("> ").strip().lower()
        if resp == "q":
            print(f"Canceled stage: {stage_name}")
            return False
        return True
    
    if args.stage in ["pre-train", "pre-train-hpc"]:
        args.checkpoint = None
    elif args.stage in ["fine-tune", "fine-tune-hpc", "inference", "inference-hpc"]:
        args.checkpoint = cfg_checkpoint_path
    else:
        args.checkpoint = None

    try:
        if args.stage in ["pre-train", "fine-tune", "inference"]:
            if not confirm_stage(f"{args.stage}"):
                return
            cfg_file_path = edit_stage_config(cfg_file_path, f"{args.stage}, config_file")
            cfg_base_path = edit_stage_config(cfg_base_path, f"{args.stage}, config_base")
            # Load and initialize
            args.config_file = cfg_file_path
            args.config_base = cfg_base_path
            with InterruptController():
                pytc_run(args)
            print("Press Enter to return menu.")
            input("> ").strip().lower()
        elif args.stage in ["pre-train-hpc", "fine-tune-hpc", "inference-hpc"]:
            if not confirm_stage(f"{args.stage}"):
                return
            cfg_file_path = edit_stage_config(cfg_file_path, f"{args.stage}, config_file")
            cfg_base_path = edit_stage_config(cfg_base_path, f"{args.stage}, config_base")
            # Load and initialize
            args.config_file = cfg_file_path
            args.config_base = cfg_base_path
            hpc_cfg_path = edit_stage_config(hpc_cfg_path, f"{args.stage}, hpc")
            with InterruptController():
                hpc_cfg = load_config(hpc_cfg_path)
                pytc_run_hpc(args, hpc_cfg)
            print("Press Enter to return menu.")
            input("> ").strip().lower()
        elif args.stage  == "status":
            if not confirm_stage(f"Tool: {args.stage}"):
                return
            with InterruptController():
                # print("To do ...")
                pass
            print("Press Enter to return menu.")
            input("> ").strip().lower()
        else:
            pass
    except KeyboardInterrupt:
        print("\nExecution interrupted abruptly by user.")
    finally:
        logging.shutdown()

    print(f"Stage {args.stage} completed.")

# ==========================================================
# Interactive CLI mode
# ==========================================================
def load_global_config(path="magneton/config.yaml"):
    """Load YAML config file."""
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"\nLoaded global config from: {path}")
        return cfg, path
    except FileNotFoundError:
        print(f"\nGlobal config not found at {path}. Using defaults.")
        return {}, path


def save_global_config(cfg, path):
    """Save updated YAML config."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved updated global config to: {path}")


def modify_global_config(cfg, cfg_path):
    """Interactive menu to modify config path or values."""
    print("\nModify Global Config")

    # new_path = input(f"> Enter new config file path (blank to keep {cfg_path}): ").strip()
    # if new_path:
    #     cfg, cfg_path = load_global_config(new_path)

    flat_items = []
    # print("\nAvailable config parameters:")
    idx = 1
    # for section, sub in cfg.items():
    #     if isinstance(sub, dict):
    #         for k, v in sub.items():
    #             flat_items.append((f"{section}/{k}", v))
    #             print(f"{idx}. {section}/{k}: {v}")
    #             idx += 1
    #     else:
    #         flat_items.append((section, sub))
    #         print(f"{idx}. {section}: {sub}")
    #         idx += 1
    console.rule("[bold bright_white]Available Config Parameters[/bold bright_white]", style="bright_cyan")
    config_table = Table(
            box=box.SIMPLE,
            title_style="bold bright_white",
            header_style="bright_white",
            show_header=True
        )
    config_table.add_column("Index", justify="center", style="white")
    config_table.add_column("Section", style="white")
    config_table.add_column("Parameter", style="white")
    config_table.add_column("Value", style="white")

    for section, sub in cfg.items():
        # console.print(f"[bold]{section}[/bold]:")
        if isinstance(sub, dict):
            for k, v in sub.items():
                # console.print(f"   - {k}: [cyan]{v}[/cyan]")
                flat_items.append((f"{section}/{k}", v))
                config_table.add_row(f"{idx}", f"{section}", f"{k}", f"[cyan]{v}[/cyan]")
                idx += 1
        else:
            # console.print(f"   - [cyan]{sub}[/cyan]")
            flat_items.append((section, sub))
            config_table.add_row("-", "-", "-", f"[cyan]{sub}[/cyan]")
            idx += 1
        
    console.print(config_table)
    while True:
        choice = input("> Select parameter to modify (number, or ENTER to finish): ").strip()
        if choice == "":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(flat_items)):
            print("Invalid selection.")
            continue

        key_path, old_val = flat_items[int(choice) - 1]
        # print('\n')
        print(f"Current value for {key_path}: {old_val}")
        new_val = input("> New value: ").strip()
        if new_val == "":
            print("No change made.")
            continue

        # Apply modification
        parts = key_path.split("/")
        target = cfg
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = new_val
        print(f"Updated {key_path} → {new_val}")

    if Prompt.ask("[white]> Save changes to file? (y/n)[/white]", default="n").lower().startswith("y"):
        save_global_config(cfg, cfg_path)
    else:
        print("Using in-memory config (not saved).")

    return cfg, cfg_path

def run_interactive():
    """Interactive CLI interface (consistent with instance_segmentation)."""
    # print("\n[Interactive Mode] Connectomics Training/Inference")
    console.print("\n[bold bright_white] Instance Segmentation Interactive Mode[/bold bright_white]\n")

    cfg_path = "magneton/config.yaml"
    cfg, cfg_path = load_global_config(cfg_path)

    choice_pool = [str(i) for i in range(10)] + ["h", "help"]


    while True:
        console.rule("[bold bright_white]Affinity Map Prediction Menu[/bold bright_white]", style="bold white")

        table = Table(show_header=True, box=box.SIMPLE, border_style="white", 
                      title_style="bold bright_white",header_style="bright_white",)
        
        table.add_column("Option", justify="center", style="white")
        table.add_column("Function", style="white")
        table.add_column("Description", style="white")
        table.add_row("1", "Pre-train", "Pre-train a model using configuration settings")
        table.add_row("2", "Pre-train [HPC]", "Pre-train a model using configuration settings by using HPC resources")
        table.add_row("3", "Fine-tune", "Fine-tune a model loop using configuration settings")
        table.add_row("4", "Fine-tune [HPC]", "Fine-tune a model loop using configuration settings by using HPC resources")
        table.add_row("5", "Inference", "Perform prediction on data")
        table.add_row("6", "Inference [HPC]", "Perform prediction on data by using HPC resources")
        table.add_row("7", "Status", "Visualization of the training status")
        table.add_row("8", "Modify Global Config", "Modify the global configuration files for each module")
        table.add_row("9", "View Current Config", "View the global configuration files for each module")
        table.add_row("0", "Return", "Return to main menu")
        # table.add_row("h", "Help", "Function description")
        console.print(table)

        # choice = input("> Select stage: ").strip().lower()
        choice = Prompt.ask("[bright_white]> Select stage[/bright_white]", default="0").strip().lower()
        
        if choice not in choice_pool:
            console.print("[red]Invalid selection. Try again.[/red]")
            continue

        if choice == "0":
            console.print("[yellow]Exit Affinity Map Prediction Pipeline.[/yellow]")
            break
        if choice == "7":
            cfg_base_path = cfg.get("affinity_prediction", {}).get("config_base", "configs/base.yaml")
            # print(cfg_base_path)
            default_dir = load_config(cfg_base_path).get("DATASET", {}).get("OUTPUT_PATH", "/")
            vis_dir = Prompt.ask("[bright_white]> Input log path: [/bright_white]", default=default_dir).strip().lower()
            
            with InterruptController():
                tb = launch_tensorboard(vis_dir)
                print("Press Enter to stop.")
                input("> ").strip().lower()
                if hasattr(tb, "_server") and tb._server:
                    try:
                        tb._server.shutdown()
                        print("[INFO] TensorBoard shut down successfully.")
                    except Exception as e:
                        print(f"[WARN] TensorBoard shutdown failed: {e}")
                else:
                    print("[WARN] TensorBoard server not running or not initialized.")
                # if hasattr(tb, "_server") and tb._server is not None:
                # tb._server.shutdown()
                continue
            # pass
        if choice == "8":
            cfg, cfg_path = modify_global_config(cfg, cfg_path)
            print("Press Enter to return menu.")
            input("> ").strip().lower()
            continue

        if choice == "9":
            console.rule("[bold bright_white]Current Global Config[/bold bright_white]", style="bright_cyan")
            config_table = Table(
                    box=box.SIMPLE,
                    title_style="bold bright_white",
                    header_style="bright_white",
                    show_header=True
                )
            config_table.add_column("Index", justify="center", style="white")
            config_table.add_column("Section", style="white")
            config_table.add_column("Parameter", style="white")
            config_table.add_column("Value", style="white")
    
            for section, sub in cfg.items():
                # console.print(f"[bold]{section}[/bold]:")
                i = 1
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        # console.print(f"   - {k}: [cyan]{v}[/cyan]")
                        config_table.add_row(f"{i}", f"{section}", f"{k}", f"[cyan]{v}[/cyan]")
                        i += 1
                else:
                    # console.print(f"   - [cyan]{sub}[/cyan]")
                    config_table.add_row("-", "-", "-", f"[cyan]{sub}[/cyan]")
                    i += 1
                
            console.print(config_table)
            print("Press Enter to return menu.")
            input("> ").strip().lower()
            continue
        args = argparse.Namespace(stage="affinity prediction")
        args.inference = (choice == "5" or choice == "6")
        args.distributed = False
        args.manual_seed = None
        args.local_world_size = 1
        args.local_rank = None
        args.debug = False
        args.opts = []
        mapping = {
            "1": "pre-train",
            "2": "pre-train-hpc",
            "3": "fine-tune",
            "4": "fine-tune-hpc",
            "5": "inference",
            "6": "inference-hpc",
        }

        args.stage = mapping.get(choice)
        # args.stage = "train" if choice == "1" else "inference"
        cfg, cfg_path = load_global_config(cfg_path)
        # === Tools Submenu ===
        
        console.print(f"\n[green]▶ Executing stage:[/] [cyan]{args.stage}[/cyan]")
        
        run(args, cfg)


# ==========================================================
# main()
# ==========================================================
def main():
    t1 = time.time()
    parser = argparse.ArgumentParser(description="Connectomics training/inference CLI")
    parser.add_argument("--stage", choices=["train", "inference"], required=True)
    parser.add_argument("--config", required=False, help="Path to training config")
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--local_rank", type=int, default=None)
    args = parser.parse_args()

    global_cfg = load_global_config()[0]
    run(args, global_cfg)
    print(f"Total runtime: {time.time() - t1:.2f}s")


if __name__ == "__main__":
    main()
