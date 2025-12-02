#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KUAN LAB — Neuron Segmentation Pipeline
Interactive CLI entrypoint
"""

import sys
import os
import warnings
import argparse
import platform
import yaml
from datetime import datetime

warnings.filterwarnings("ignore")

# ==== Optional rich UI ====
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box
except ImportError:
    print("  Optional dependency 'rich' not found. Install with `pip install rich` for better UI.")
    # fallback to plain console
    class DummyConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("=" * 60)
    Console = DummyConsole
    Table = None
    Prompt = None
    box = None

console = Console()

# ==== Modules ====
import magneton.instance_segmentation as ins_segmentation
import magneton.pytorch_connectomics as aff_inference
import magneton.toolkit as toolkit

# ---------------------------------------------------
# Config loader
# ---------------------------------------------------
def load_global_config(path="config.yaml"):
    """Load global YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold yellow] Global config not found:[/bold yellow] {path}, using defaults.")
        return {}


# ---------------------------------------------------
# KUAN LAB Banner
# ---------------------------------------------------
def show_banner():
    banner = r"""
  ███╗   ███╗ █████╗  ██████╗ ███╗   ██╗███████╗████████╗ ██████╗ ███╗   ██╗
  ████╗ ████║██╔══██╗██╔════╝ ████╗  ██║██╔════╝╚══██╔══╝██╔═══██╗████╗  ██║
  ██╔████╔██║███████║██║  ███╗██╔██╗ ██║█████╗     ██║   ██║   ██║██╔██╗ ██║
  ██║╚██╔╝██║██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ██║   ██║██║╚██╗██║
  ██║ ╚═╝ ██║██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ╚██████╔╝██║ ╚████║
  ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝
"""

    console.print(banner, style="bright_white")


# ---------------------------------------------------
# Environment Summary
# ---------------------------------------------------
def show_environment_info(config_path):
    import torch

    # console.rule("[bold bright_white]Environment Information[/bold bright_white]")
    console.print(f"[bold white] Config file:[/bold white] {os.path.abspath(config_path)}")

    console.print(f"[bold white] Python version:[/bold white] {platform.python_version()}")
    console.print(f"[bold white] PyTorch version:[/bold white] {torch.__version__}")

    if torch.cuda.is_available():
        console.print(f"[bold green]  GPU:[/bold green] {torch.cuda.get_device_name(0)}")
        console.print(f"[bold white]  CUDA devices:[/bold white] {torch.cuda.device_count()}")
    else:
        console.print("[bold yellow]  Using CPU (no CUDA available)[/bold yellow]")

    console.print(f"[bold white] Working directory:[/bold white] {os.getcwd()}")
    # console.print(f"[bold white] Tool:[/bold white] Three-in-One Neuron Segmentation CLI (Magneton)")
    console.print(f"[bold white] Started at:[/bold white] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # console.rule()


# ---------------------------------------------------
# Menu
# ---------------------------------------------------
def show_menu():
    console.rule("[bold bright_white]Neuron Segmentation CLI[/bold bright_white]", style="bold white", characters="=")
    table = Table(
        box=box.SIMPLE,
        title_style="bold bright_white",
        header_style="bright_white",
    )
    table.add_column("Option", justify="center", style="white")
    table.add_column("Module", style="white")
    table.add_column("Description", style="white")

    table.add_row("1", "Processing Toolkit", "Data pre- and post-processing toolkit")
    table.add_row("2", "Affinity Map Inference", "Train / Infer affinity maps using deep learning models")
    table.add_row("3", "Instance Segmentation", "Perform block-based segmentation and merge across blocks")
    table.add_row("0", "Exit", "Close CLI")
    console.print(table)


# ---------------------------------------------------
# Main entrypoint
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="Magneton",
        description="Unified CLI for neuron segmentation workflows",
    )

    # === Interactive Mode ===
    if len(sys.argv) == 1:
        config_path = "config.yaml"
        show_banner()
        show_environment_info(config_path)

        while True:
            show_menu()
            choice = Prompt.ask("\n> Select a module", choices=["0", "1", "2", "3"], default="0")

            if choice == "1":
                console.rule("[bold bright_white]Pre- and Post-Processing Toolkit[/bold bright_white]", style="bold white", characters="=")
                toolkit.run_interactive()

            elif choice == "2":
                console.rule("[bold bright_white]Affinity Map Inference Module[/bold bright_white]", style="bold white", characters="=")
                aff_inference.run_interactive()

            elif choice == "3":
                console.rule("[bold bright_white]Instance Segmentation Module[/bold bright_white]", style="bold white", characters="=")
                ins_segmentation.run_interactive()

            elif choice == "0":
                console.print("\n[bold bright_red] Exiting Magneton... [/bold bright_red]\n")
                break

        return

    # === Command-line (non-interactive) Mode ===
    args = parser.parse_args()
    global_cfg = load_global_config(getattr(args, "config", "config.yaml"))
    if hasattr(args, "func"):
        args.func(args, global_cfg)
    else:
        console.print("[red]Error:[/red] No function specified in args.")
