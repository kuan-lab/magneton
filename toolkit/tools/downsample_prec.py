import sys
import os
import igneous.task_creation as tc
from taskqueue import TaskQueue
import argparse
from magneton.toolkit.utils.config import load_config
from multiprocessing import Process
import time
import traceback


def create_task_queue(queuepath, source_path, mip, num_mips, factor):
    bounds = None  # None will use full bounds
    tq = TaskQueue('fq://'+queuepath)
    # source_path=input('Cloud Path:')
    tasks = tc.create_downsampling_tasks(
        source_path,
        mip=mip,       # Starting mip
        num_mips=num_mips,  # Final mip to downsample to
        bounds=bounds,
        factor=factor,  # Downsample all 3 axes
        compress=False,
        fill_missing=True,
    )
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


def run_multiple_workers(queuepath, num_workers=8, idle_exit_seconds=60, file_idle_threshold=60):
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

        # Check if the queue has any activity
        if queue_recently_active():
            last_activity = time.time()

        # Queue remains inactive for an extended period → Terminate all workers
        if time.time() - last_activity > idle_exit_seconds:
            print(f"[MAIN] Queue has been idle for {idle_exit_seconds}s → terminating all workers.")
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
        create_task_queue(queuepath, source_path, mip, num_mips, factor)
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
        create_task_queue(queuepath, source_path, mip, num_mips, factor)
        run_multiple_workers(queuepath=queuepath, num_workers=num_workers)
    else:
        print('downsample flag is false.')


if __name__ == '__main__':
    main()