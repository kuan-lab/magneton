import os
from pathlib import Path
import yaml

def _default_config_path() -> Path:
    # config.yaml in the configs directory
    return Path(__file__).with_name("magneton/instance_segmentation/configs/config.yaml")

def _default_global_config_path() -> Path:
    # config.yaml in the configs directory
    return Path(__file__).with_name("magneton/config.yaml")


def load_global_config_path(path: str = None):
    """
    Load configuration path:
    - When path is None, read ./config.yaml
    - When path is a relative/absolute path, read the specified file
    """
    if path is None:
        cfg_path = _default_global_config_path()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            # Allow input similar to “magneton/config.yaml”"
            maybe_pkg = Path(__file__).resolve().parent / Path(path).name
            if maybe_pkg.is_file():
                cfg_path = maybe_pkg
    # print(cfg_path)
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_config(path: str = None):
    """
    Load configuration:
    - When path is None, read ./instance_segmentation/configs/config.yaml
    - When path is a relative/absolute path, read the specified file
    """
    if path is None:
        cfg_path = _default_config_path()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            # Allow input similar to “./instance_segmentation/configs/config.yaml”"
            maybe_pkg = Path(__file__).resolve().parent / Path(path).name
            if maybe_pkg.is_file():
                cfg_path = maybe_pkg
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

