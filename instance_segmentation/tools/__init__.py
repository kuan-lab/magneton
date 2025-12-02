# -*- coding: utf-8 -*-
"""
tools: Command-line tools and in-node sharded executors
"""
try:
    from .run_local_shard import main as run_local_shard_main
except Exception:
    run_local_shard_main = None


__all__ = ["run_local_shard_main", ]
