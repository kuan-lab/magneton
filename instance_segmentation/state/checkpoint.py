import os
import json

# ---------- General Tools ----------
def _load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def _save_json(path, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

# ---------- Segmentation stage ----------
def local_done_path(local_ckpt_dir: str, i: int) -> str:
    """Return the path to the .done file in the local checkpoint"""
    return os.path.join(local_ckpt_dir, f"block_{i:04d}.done")

def mark_local_done(local_ckpt_dir: str, i: int):
    """Mark a block as completed"""
    os.makedirs(local_ckpt_dir, exist_ok=True)
    open(local_done_path(local_ckpt_dir, i), "w").close()

def is_local_done(local_ckpt_dir: str, i: int) -> bool:
    """Check whether a block is complete"""
    return os.path.exists(local_done_path(local_ckpt_dir, i))

# ---------- Merge stage ----------
def load_merge_state(merge_ckpt_dir: str):
    """Loading merge state (state.json)"""
    state_path = os.path.join(merge_ckpt_dir, "state.json")
    return _load_json(state_path, default={"next_gid": 1, "merged_blocks": [], "num_pools": 0})

def save_merge_state(merge_ckpt_dir: str, state: dict):
    """Save merge state (state.json)"""
    state_path = os.path.join(merge_ckpt_dir, "state.json")
    _save_json(state_path, state)
