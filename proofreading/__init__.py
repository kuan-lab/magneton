# -*- coding: utf-8 -*-
"""
proofreading package initializer.
Skeleton-driven proofreading / GT-bootstrap loop (skeletonize -> WebKnossos
correction -> nnInteractive expansion).
"""
from .main import run, run_interactive

__all__ = ["config", "stages", "lib", "get_version"]

__version__ = "0.1.0"


def get_version() -> str:
    return __version__
