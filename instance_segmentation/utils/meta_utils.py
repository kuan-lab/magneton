import os
import re
import json

def block_meta_path(metadata_dir: str, i: int) -> str:
    """Return the metadata file path for a single block"""
    return os.path.join(metadata_dir, f"block_{i:04d}.json")

def index_meta_path(metadata_dir: str) -> str:
    """Return to index file path"""
    return os.path.join(metadata_dir, "index.json")

# ---------- Write ----------
def save_block_meta(metadata_dir: str, block_meta: dict):
    """
    Save metadata for an individual block to its own file.
    block_meta must contain:
      index: int
      coords: [z1,z2,y1,y2,x1,x2]
      path: str
      done: bool
      max_id: int
    """
    os.makedirs(metadata_dir, exist_ok=True)
    path = block_meta_path(metadata_dir, block_meta["index"])
    with open(path, "w") as f:
        json.dump(block_meta, f, indent=2)

def build_index(metadata_dir: str) -> dict:
    """Rebuild index.json from individual block_XXXX.json files."""
    if not os.path.isdir(metadata_dir):
        return {"blocks": []}
    pattern = re.compile(r"^block_\d{4}\.json$")
    blocks = []
    for fname in os.listdir(metadata_dir):
        if pattern.match(fname):
            with open(os.path.join(metadata_dir, fname), "r") as f:
                blocks.append(json.load(f))
    blocks.sort(key=lambda b: b["index"])
    index_data = {"blocks": blocks}
    index_path = index_meta_path(metadata_dir)
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"[INFO] Built index.json from {len(blocks)} block files in {metadata_dir}")
    return index_data

# ---------- Read ----------
def load_block_meta(metadata_dir: str, i: int) -> dict:
    """Read metadata for a single block"""
    path = block_meta_path(metadata_dir, i)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Block metadata not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_index_meta(metadata_dir: str) -> dict:
    """Load index.json if it exists, otherwise build it from block files."""
    path = index_meta_path(metadata_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return build_index(metadata_dir)

