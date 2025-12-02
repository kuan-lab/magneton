# -*- coding: utf-8 -*-
"""
state: checkpoint/state management
"""
from .checkpoint import (
    load_merge_state, save_merge_state,
    local_done_path, mark_local_done, is_local_done,
)

__all__ = [
    "load_merge_state",
    "save_merge_state",
    "local_done_path",
    "mark_local_done",
    "is_local_done",
]
