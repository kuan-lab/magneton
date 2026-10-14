# -*- coding: utf-8 -*-
"""Small SLURM helpers for driving the pipeline without the interactive menu."""
import re
import subprocess
import time

_JOBID = re.compile(r"Submitted batch job (\d+)")


def parse_job_id(sbatch_output: str):
    """Pull the job id out of `sbatch` stdout, or None if it isn't there."""
    m = _JOBID.search(sbatch_output or "")
    return m.group(1) if m else None


def job_active(job_id: str) -> bool:
    """True while any task of the array is still pending or running."""
    try:
        out = subprocess.check_output(
            ["squeue", "-j", str(job_id), "-h", "-o", "%i"],
            stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    except subprocess.CalledProcessError:
        # squeue errors once the job has left the queue entirely
        return False
    return bool(out.strip())


def job_states(job_id: str):
    """{state: count} from sacct, for the post-mortem after a wait."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id), "-n", "-X", "-P", "-o", "State"],
            stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            states[s] = states.get(s, 0) + 1
    return states


def wait_for_job(job_id: str, poll: int = 60, timeout: int = None) -> bool:
    """
    Block until the array leaves the queue. Returns True if every task ended
    COMPLETED, False otherwise (the caller decides whether to continue).
    """
    if not job_id:
        print("[WARN] no job id to wait on")
        return False
    t0 = time.time()
    print(f"[INFO] waiting on job {job_id} (poll {poll}s)")
    while job_active(job_id):
        if timeout and time.time() - t0 > timeout:
            print(f"[WARN] wait timed out after {timeout}s; job {job_id} still queued")
            return False
        time.sleep(poll)
    states = job_states(job_id)
    elapsed = time.time() - t0
    print(f"[INFO] job {job_id} left the queue after {elapsed / 60:.1f} min: {states}")
    return bool(states) and set(states) == {"COMPLETED"}
