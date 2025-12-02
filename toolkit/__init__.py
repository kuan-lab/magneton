# -*- coding: utf-8 -*-
"""
toolkit package initializer.
Keep it lightweight; avoid importing large dependencies here to prevent introducing side effects during imports.
"""
from .main import run, run_interactive

__all__ = [
    "config",
    "tools",
    "utils",
    "get_version",
]

__version__ = "0.1.0"


def get_version() -> str:
    return __version__
