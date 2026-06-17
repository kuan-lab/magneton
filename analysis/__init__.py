# -*- coding: utf-8 -*-
"""
analysis package initializer.
Per-instance morphometrics pipeline (mito, bouton, synapse, ...; bbox-driven architecture).
"""
from .main import run, run_interactive

__all__ = [
    "config",
    "stages",
    "lib",
    "get_version",
]

__version__ = "0.1.0"


def get_version() -> str:
    return __version__
