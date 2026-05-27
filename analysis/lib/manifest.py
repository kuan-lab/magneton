"""
manifest.txt read/write for the stage-B SLURM array.

Each line is `start,end` — half-open row range into bboxes.parquet.
SLURM array task k reads its line via `sed -n "$((SLURM_ARRAY_TASK_ID+1))p"`.
"""
from pathlib import Path
from typing import List, Tuple


def make_ranges(n_total: int, per_task: int) -> List[Tuple[int, int]]:
    if per_task <= 0:
        raise ValueError("per_task must be positive")
    ranges = []
    for start in range(0, n_total, per_task):
        end = min(start + per_task, n_total)
        ranges.append((start, end))
    return ranges


def write_manifest(path: str, ranges: List[Tuple[int, int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for start, end in ranges:
            f.write(f"{start},{end}\n")


def read_manifest_line(path: str, task_id: int) -> Tuple[int, int]:
    """1-indexed sed semantics: task_id=0 reads the first line."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == task_id:
                parts = line.strip().split(",")
                return int(parts[0]), int(parts[1])
    raise IndexError(f"task_id {task_id} out of range for manifest {path}")
