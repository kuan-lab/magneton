import sys
import os
import re
import igneous.task_creation as tc
from taskqueue import TaskQueue
import argparse
from magneton.toolkit.utils.config import load_config
from multiprocessing import Process
import time
import traceback


def parse_memory_string(mem_str):
    """Parse memory string like '32G', '4096M', '4096' to bytes."""
    if isinstance(mem_str, (int, float)):
        return int(mem_str)

    mem_str = str(mem_str).strip().upper()
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$', mem_str)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    return int(value * multipliers.get(unit, 1))


def calculate_memory_target(num_workers):
    """
    Calculate memory_target based on environment.
    - SLURM job: (SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK / num_workers) * 0.75
    - Non-SLURM: 5GB default
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if slurm_job_id:
        mem_per_cpu_str = os.environ.get("SLURM_MEM_PER_CPU", "4096")
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

        # SLURM_MEM_PER_CPU is always in MB (plain number, no unit suffix)
        if re.match(r'^\d+$', mem_per_cpu_str):
            mem_per_cpu_str = mem_per_cpu_str + 'M'
        mem_per_cpu = parse_memory_string(mem_per_cpu_str)
        if mem_per_cpu is None:
            mem_per_cpu = 4 * 1024 * 1024 * 1024

        total_memory = mem_per_cpu * cpus
        memory_target = int((total_memory / num_workers) * 0.75)
        GiB = 1024**3
        print(f"[INFO] SLURM job detected (ID: {slurm_job_id})")
        print(f"[INFO] Memory: {mem_per_cpu/GiB:.1f}GB x {cpus} CPUs = {total_memory/GiB:.1f}GB total")
        print(f"[INFO] Memory target: {total_memory/GiB:.1f}GB / {num_workers} workers x 0.75 = {memory_target/GiB:.1f}GB per task")
        return memory_target
    else:
        print("[INFO] Non-SLURM mode: using 5GB memory target")
        return 5_000_000_000


def _restrict_tasks_to_bounds(tasks, bounds, shape):
    """Restrict an igneous task iterator to an ROI sub-region in-place.

    `create_meshing_tasks` returns a FinelyDividedTaskIterator whose grid is
    driven by `tasks.bounds` (the full volume's mip-bounds). The installed
    igneous (4.27) has no `bounds=` arg, so we override the iterator's bounds
    to the ROI: it then tiles only the ROI while the task() closure still emits
    correct absolute offsets. `bounds` is a cloudvolume Bbox in the SAME voxel
    space as `tasks.bounds` (the mesh mip). The ROI is snapped outward to the
    task-`shape` grid (relative to the volume origin) so cells align and fully
    cover the ROI, then clipped to the volume.
    """
    import numpy as np
    from cloudvolume import Bbox
    from igneous.task_creation.common import num_tasks

    sh = np.array(shape)
    vol_min = np.array(tasks.bounds.minpt)
    vol_max = np.array(tasks.bounds.maxpt)
    rmin = np.array(bounds.minpt)
    rmax = np.array(bounds.maxpt)
    mn = vol_min + ((rmin - vol_min) // sh) * sh
    mx = vol_min + np.ceil((rmax - vol_min) / sh).astype(int) * sh
    mn = np.maximum(mn, vol_min)
    mx = np.minimum(mx, vol_max)
    roi = Bbox(mn.tolist(), mx.tolist())
    tasks.bounds = roi
    tasks.start = 0
    tasks.end = num_tasks(roi, shape)
    print(f"[BOUNDS] meshing restricted to {roi} -> {tasks.end} tasks")
    return tasks


def create_meshing_queue(queuepath, source_path, mip, shape, simplification,
                         max_simplification_error, dust_threshold, bounds=None):
    """Create meshing tasks and enqueue them.

    bounds: optional cloudvolume Bbox (in mesh-`mip` voxel coords) to restrict
    meshing to an ROI. None = whole declared volume.
    """
    tq = TaskQueue('fq://' + queuepath)

    tasks = tc.create_meshing_tasks(
        source_path,
        mip=mip,
        shape=shape,
        simplification=simplification,
        max_simplification_error=max_simplification_error,
        dust_threshold=dust_threshold,
        fill_missing=True,
        encoding='precomputed',
        spatial_index=True,
        compress=False,
    )
    if bounds is not None:
        tasks = _restrict_tasks_to_bounds(tasks, bounds, shape)
    tq.insert(tasks)
    tq.rezero()
    total = len(tasks)
    print('Done adding {} meshing tasks to queue at {}'.format(total, queuepath))
    return total


def create_manifest_queue(queuepath, source_path):
    """Create mesh manifest tasks and enqueue them."""
    tq = TaskQueue('fq://' + queuepath)

    tasks = tc.create_mesh_manifest_tasks(source_path)
    tq.insert(tasks)
    tq.rezero()
    total = len(tasks)
    print('Done adding {} manifest tasks to queue at {}'.format(total, queuepath))
    return total


def run_tasks_from_queue(queuepath):
    tq = TaskQueue('fq://' + queuepath)
    print('Working on tasks from filequeue "{}"'.format(queuepath))
    tq.poll(
        verbose=True,
        lease_seconds=3000,
        tally=True,
    )
    print('Done')


def run_multiple_workers(queuepath, num_workers=8, idle_exit_seconds=120, file_idle_threshold=60, total_tasks=None):
    """
    Launch multiple workers and monitor the queue status.
    If the queue remains unchanged for an extended period, terminate all workers and exit.
    """
    print(f"[MAIN] Starting {num_workers} parallel workers for queue '{queuepath}'...")
    processes = []

    for i in range(num_workers):
        p = Process(target=run_tasks_from_queue, args=(queuepath,))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    queue_dir = os.path.join(os.path.abspath(queuepath), "queue")
    tq = TaskQueue('fq://' + queuepath)
    last_activity = time.time()
    last_queue_count = None

    def get_queue_count():
        try:
            return len([f for f in os.listdir(queue_dir) if f.endswith(".json")])
        except FileNotFoundError:
            return 0

    def queue_recently_active():
        now = time.time()
        try:
            files = [os.path.join(queue_dir, f) for f in os.listdir(queue_dir) if f.endswith(".json")]
        except FileNotFoundError:
            return False

        for f in files:
            try:
                if now - os.path.getmtime(f) < file_idle_threshold:
                    return True
            except FileNotFoundError:
                continue
        return False

    while True:
        alive = [p for p in processes if p.is_alive()]
        if not alive:
            print("[MAIN] All workers have exited on their own.")
            break

        current_queue_count = get_queue_count()
        completed = tq.completed

        if total_tasks is not None and completed >= total_tasks:
            print(f"[MAIN] All {completed}/{total_tasks} tasks completed.")
            for p in alive:
                if p.is_alive():
                    p.terminate()
            break

        if queue_recently_active() or (last_queue_count is not None and current_queue_count != last_queue_count):
            last_activity = time.time()
        last_queue_count = current_queue_count

        if time.time() - last_activity > idle_exit_seconds:
            print(f"[MAIN] {completed}/{total_tasks or '?'} tasks completed, "
                  f"{current_queue_count} in queue, no progress for {idle_exit_seconds}s -> terminating workers.")
            for p in alive:
                if p.is_alive():
                    p.terminate()
            break

        time.sleep(2)

    for p in processes:
        p.join(timeout=2)

    print("[MAIN] All workers cleaned up. Phase complete.")


def mesh_prec(cfg):
    mesh_cfg = cfg["mesh"]
    if not mesh_cfg.get("flag", True):
        print('mesh flag is false.')
        return

    source_path = mesh_cfg["source_path"]
    queuepath_base = mesh_cfg.get("queuepath", "magneton/igneous_tasks")
    mip = mesh_cfg.get("mip", 0)
    shape = tuple(mesh_cfg.get("shape", [448, 448, 448]))
    simplification = mesh_cfg.get("simplification", True)
    max_simplification_error = mesh_cfg.get("max_simplification_error", 40)
    dust_threshold = mesh_cfg.get("dust_threshold", None)
    num_workers = mesh_cfg.get("num_workers", 16)

    # Optional ROI to restrict meshing. Project-standard ZYX [z1,z2,y1,y2,x1,x2]
    # in mesh-`mip` voxel coords (matches inference ROI / block.roi). None=full.
    bounds_cfg = mesh_cfg.get("bounds", None)
    bounds = None
    if bounds_cfg is not None:
        from cloudvolume import Bbox
        z1, z2, y1, y2, x1, x2 = bounds_cfg
        bounds = Bbox([x1, y1, z1], [x2, y2, z2])  # igneous is XYZ

    vol_name = os.path.basename(source_path.rstrip("/"))
    queuepath_mesh = os.path.join(queuepath_base, vol_name + "_mesh")
    queuepath_manifest = os.path.join(queuepath_base, vol_name + "_mesh_manifest")

    # Phase 1: Meshing (heavy — marching cubes per block)
    print(f"\n{'='*60}")
    print(f"Phase 1: Meshing — {source_path}")
    print(f"  mip={mip}, shape={shape}, simplification={simplification}")
    print(f"  max_simplification_error={max_simplification_error}, dust_threshold={dust_threshold}")
    print(f"{'='*60}\n")

    total_mesh = create_meshing_queue(
        queuepath_mesh, source_path, mip, shape,
        simplification, max_simplification_error, dust_threshold,
        bounds=bounds,
    )
    run_multiple_workers(
        queuepath=queuepath_mesh,
        num_workers=num_workers,
        total_tasks=total_mesh,
        idle_exit_seconds=300,
    )

    # Phase 2: Manifest (lightweight — groups fragments by segment ID)
    print(f"\n{'='*60}")
    print(f"Phase 2: Mesh manifest")
    print(f"{'='*60}\n")

    total_manifest = create_manifest_queue(queuepath_manifest, source_path)
    manifest_workers = min(num_workers, total_manifest, 4)
    run_multiple_workers(
        queuepath=queuepath_manifest,
        num_workers=manifest_workers,
        total_tasks=total_manifest,
        idle_exit_seconds=120,
    )

    print(f"\n[DONE] Meshing complete for {source_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate meshes for Neuroglancer precomputed segmentation volumes.")
    parser.add_argument("--config", default="config_mesh.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mesh_prec(cfg)


if __name__ == '__main__':
    main()
