import os
from pathlib import Path
import yaml


def _default_global_config_path() -> Path:
    return Path(__file__).with_name("magneton/config.yaml")


def load_global_config_path(path: str = None):
    """Load the root pointer config (./config.yaml by default)."""
    cfg_path = _default_global_config_path() if path is None else Path(path)
    if path is not None and not cfg_path.is_file():
        maybe = Path(__file__).resolve().parent / Path(path).name
        if maybe.is_file():
            cfg_path = maybe
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: str = None):
    """Load a proofreading per-volume YAML config."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        maybe = Path(__file__).resolve().parent / Path(path).name
        if maybe.is_file():
            cfg_path = maybe
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_stage_config(cfg, stage: str):
    if stage == "paths":
        return cfg.get("paths", {})
    if stage == "skeletonize":
        return cfg.get("skeletonize_stage", {})
    if stage == "expand":
        return cfg.get("expand_stage", {})
    if stage == "membrane":
        return cfg.get("membrane_stage", {})
    return {}


def strip_file_prefix(p: str) -> str:
    if p is None:
        return None
    return p[len("file://"):] if p.startswith("file://") else p
