import os
from pathlib import Path
import yaml


def _default_config_path() -> Path:
    return Path(__file__).with_name("magneton/analysis/configs/config.yaml")


def _default_global_config_path() -> Path:
    return Path(__file__).with_name("magneton/config.yaml")


def load_global_config_path(path: str = None):
    """
    Load global magneton config (the root pointer config).
    - path is None → read ./config.yaml
    - path given → read that file, with a fallback that resolves relative-to-package
    """
    if path is None:
        cfg_path = _default_global_config_path()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            maybe_pkg = Path(__file__).resolve().parent / Path(path).name
            if maybe_pkg.is_file():
                cfg_path = maybe_pkg
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: str = None):
    """
    Load an analysis per-volume YAML config.
    """
    if path is None:
        cfg_path = _default_config_path()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            maybe_pkg = Path(__file__).resolve().parent / Path(path).name
            if maybe_pkg.is_file():
                cfg_path = maybe_pkg
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_stage_config(cfg, stage: str):
    if stage == "discover":
        return cfg.get("discover_stage", {})
    elif stage == "instance":
        return cfg.get("instance_stage", {})
    elif stage == "features":
        return cfg.get("features", {})
    elif stage == "paths":
        return cfg.get("paths", {})
    return {}


def strip_file_prefix(p: str) -> str:
    """Strip 'file://' prefix that CloudVolume uses but plain file ops don't accept."""
    if p is None:
        return None
    if p.startswith("file://"):
        return p[len("file://"):]
    return p
