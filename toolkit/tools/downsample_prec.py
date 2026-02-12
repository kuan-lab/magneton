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
    Calculate memory_target for igneous based on environment.
    - SLURM job: (SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK / num_workers) * 0.75
    - Non-SLURM: 5GB default
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if slurm_job_id:
        # Running in SLURM - get memory allocation from environment
        mem_per_cpu_str = os.environ.get("SLURM_MEM_PER_CPU", "4096")  # Default 4GB in MB
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

        # SLURM_MEM_PER_CPU is always in MB (plain number, no unit suffix)
        # Append 'M' so parse_memory_string treats it correctly
        if re.match(r'^\d+$', mem_per_cpu_str):
            mem_per_cpu_str = mem_per_cpu_str + 'M'
        mem_per_cpu = parse_memory_string(mem_per_cpu_str)
        if mem_per_cpu is None:
            mem_per_cpu = 4 * 1024 * 1024 * 1024  # 4GB fallback

        total_memory = mem_per_cpu * cpus
        memory_target = int((total_memory / num_workers) * 0.75)
        GiB = 1024**3
        print(f"[INFO] SLURM job detected (ID: {slurm_job_id})")
        print(f"[INFO] Memory: {mem_per_cpu/GiB:.1f}GB × {cpus} CPUs = {total_memory/GiB:.1f}GB total")
        print(f"[INFO] Memory target: {total_memory/GiB:.1f}GB / {num_workers} workers × 0.75 = {memory_target/GiB:.1f}GB per task")
        return memory_target
    else:
        print("[INFO] Non-SLURM mode: using 5GB memory target")
        return 5_000_000_000  # 5GB default for non-HPC


def create_task_queue(queuepath, source_path, mip, num_mips, factor, memory_target=None):
    bounds = None  # None will use full bounds
    tq = TaskQueue('fq://'+queuepath)

    kwargs = {
        'mip': mip,
        'num_mips': num_mips,
        'bounds': bounds,
        'factor': factor,
        'compress': False,
        'fill_missing': True,
    }
    if memory_target is not None:
        kwargs['memory_target'] = memory_target

    tasks = tc.create_downsampling_tasks(source_path, **kwargs)
    tq.insert(tasks)
    print('Done adding {} tasks to queue at {}'.format(len(tasks), queuepath))

def run_tasks_from_queue(queuepath):
    tq = TaskQueue('fq://'+queuepath)
    print('Working on tasks from filequeue "{}"'.format(queuepath))
    tq.poll(
        verbose=True, # prints progress
        lease_seconds=3000,
        tally=True, # makes tq.completed work, logs 1 byte per completed task
    )
    print('Done')


def run_multiple_workers(queuepath, num_workers=8, idle_exit_seconds=300, file_idle_threshold=60):
    """
    Launch multiple workers and monitor the queue status.
    If the queue remains unchanged for an extended period (even if not emptied), the main process will terminate all workers and exit.
    """
    print(f"[MAIN] Starting {num_workers} parallel workers for queue '{queuepath}'...")
    processes = []

    # Start worker
    for i in range(num_workers):
        p = Process(target=run_tasks_from_queue, args=(queuepath,))
        p.start()
        processes.append(p)
        time.sleep(0.5)  # small stagger

    queue_dir = os.path.join(os.path.abspath(queuepath), "queue")
    last_activity = time.time()
    last_queue_count = None

    def get_queue_count():
        """Return count of .json files in the queue directory"""
        try:
            return len([f for f in os.listdir(queue_dir) if f.endswith(".json")])
        except FileNotFoundError:
            return 0

    def queue_recently_active():
        """Determine whether queue/ contains any recently modified files"""
        now = time.time()
        try:
            files = [os.path.join(queue_dir, f) for f in os.listdir(queue_dir) if f.endswith(".json")]
        except FileNotFoundError:
            return False

        for f in files:
            try:
                if now - os.path.getmtime(f) < file_idle_threshold:
                    return True  # There have been recent document updates.
            except FileNotFoundError:
                continue
        return False

    # Main Loop: Monitor worker and queue status
    while True:
        alive = [p for p in processes if p.is_alive()]
        if not alive:
            print("[MAIN] All workers have exited on their own.")
            break

        current_queue_count = get_queue_count()

        # Check if the queue has any activity (file modifications or count changes)
        if queue_recently_active() or (last_queue_count is not None and current_queue_count != last_queue_count):
            last_activity = time.time()
        last_queue_count = current_queue_count

        # Queue remains inactive for an extended period → Terminate all workers
        if time.time() - last_activity > idle_exit_seconds:
            if current_queue_count == 0:
                print(f"[MAIN] Queue is empty and idle for {idle_exit_seconds}s → terminating all workers.")
            else:
                print(f"[MAIN] Queue has {current_queue_count} remaining tasks but no progress for {idle_exit_seconds}s (tasks may be stuck) → terminating workers.")
            for p in alive:
                if p.is_alive():
                    p.terminate()
            break

        time.sleep(2)

    # Wait and clean up worker
    for p in processes:
        p.join(timeout=2)

    print("[MAIN] All workers cleaned up. Job complete.")

def main():
    parser = argparse.ArgumentParser(description="Downsample Neuroglancer Precomputed data.")
    parser.add_argument("--config", default="config_downsample.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    queuepath = cfg["downsample"]["queuepath"]
    source_path = cfg["downsample"]["source_path"]
    mip = cfg["downsample"]["mip"]
    num_mips = cfg["downsample"]["num_mips"]
    factor = cfg["downsample"]["factor"]
    num_workers = cfg["downsample"]["num_workers"]

    if cfg["downsample"]["flag"]:
        memory_target = calculate_memory_target(num_workers)
        create_task_queue(queuepath, source_path, mip, num_mips, factor, memory_target)
        run_multiple_workers(queuepath=queuepath, num_workers=num_workers)
    else:
        print('downsample flag is false.')


def downsample_prec(cfg):
    queuepath = cfg["downsample"]["queuepath"]
    source_path = cfg["downsample"]["source_path"]
    mip = cfg["downsample"]["mip"]
    num_mips = cfg["downsample"]["num_mips"]
    factor = cfg["downsample"]["factor"]
    num_workers = cfg["downsample"]["num_workers"]

    if cfg["downsample"]["flag"]:
        memory_target = calculate_memory_target(num_workers)
        create_task_queue(queuepath, source_path, mip, num_mips, factor, memory_target)
        run_multiple_workers(queuepath=queuepath, num_workers=num_workers)
    else:
        print('downsample flag is false.')


if __name__ == '__main__':
    main()