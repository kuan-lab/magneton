import os
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
    Save metadata for individual blocks while updating index.json
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

    # Update index.json
    index_path = index_meta_path(metadata_dir)
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
    else:
        index_data = {"blocks": []}

    # Update or add the current block
    found = False
    for i, blk in enumerate(index_data["blocks"]):
        if blk["index"] == block_meta["index"]:
            index_data["blocks"][i] = block_meta
            found = True
            break
    if not found:
        index_data["blocks"].append(block_meta)

    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

# ---------- Read ----------
def load_block_meta(metadata_dir: str, i: int) -> dict:
    """Read metadata for a single block"""
    path = block_meta_path(metadata_dir, i)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Block metadata not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_index_meta(metadata_dir: str) -> dict:
    """Read index.json"""
    path = index_meta_path(metadata_dir)
    if not os.path.exists(path):
        return {"blocks": []}
    with open(path, "r") as f:
        return json.load(f)
