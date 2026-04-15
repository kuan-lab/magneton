#!/usr/bin/env python3
"""Analyze a SLURM job (or array job) and report run metrics + concurrency.

Pulls task-level data from sacct and computes:
- Wall clock (first start -> last end, across all array tasks)
- Total CPU-time accrued across tasks
- Task count by final state
- MaxRSS distribution (approximate memory usage)
- Concurrency timeline: % of wall clock spent at N concurrent tasks

Usage:
    analyze.py <jobid> [<jobid> ...]
"""
import subprocess
import sys
from collections import Counter
from datetime import datetime


SACCT_FIELDS = [
    "JobID", "JobName", "Start", "End", "Elapsed",
    "State", "MaxRSS", "ReqCPUS", "ReqMem", "ExitCode",
]


def parse_sacct(jobid):
    """Fetch task-level rows. MaxRSS lives on .batch steps, so query without -X
    and fold the step-level MaxRSS back onto the parent task row."""
    out = subprocess.run(
        ["sacct", "-j", jobid, "-o", ",".join(SACCT_FIELDS), "-P", "-n"],
        capture_output=True, text=True,
    )
    parents = {}  # JobID -> dict
    step_mems = {}  # parent JobID -> max RSS seen across steps

    for line in out.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != len(SACCT_FIELDS):
            continue
        row = dict(zip(SACCT_FIELDS, parts))
        jid = row["JobID"]

        if "." in jid:
            # Step row (e.g., 12345.batch) — scavenge MaxRSS
            parent = jid.split(".", 1)[0]
            mem = parse_mem_kb(row["MaxRSS"])
            if mem > step_mems.get(parent, 0):
                step_mems[parent] = mem
        else:
            parents[jid] = row

    for pid, row in parents.items():
        if step_mems.get(pid, 0) > 0:
            row["MaxRSS_kb"] = step_mems[pid]
        else:
            row["MaxRSS_kb"] = parse_mem_kb(row.get("MaxRSS", ""))

    return list(parents.values())


def to_dt(s):
    if not s or s in ("Unknown", "None", ""):
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def parse_elapsed(s):
    """Parse SLURM Elapsed string (DD-HH:MM:SS or HH:MM:SS) to seconds."""
    if not s:
        return 0
    days = 0
    if "-" in s:
        days_s, s = s.split("-", 1)
        days = int(days_s)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = "0", parts[0], parts[1]
    else:
        return 0
    return days * 86400 + int(h) * 3600 + int(m) * 60 + int(float(sec))


def parse_mem_kb(s):
    """MaxRSS is reported with suffix K/M/G. Return KB as float."""
    if not s:
        return 0.0
    s = s.strip()
    if not s:
        return 0.0
    suffix = s[-1].upper()
    try:
        num = float(s[:-1]) if suffix in "KMGT" else float(s)
    except ValueError:
        return 0.0
    return {"K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}.get(suffix, 1) * num


def fmt_duration(sec):
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    if sec < 86400:
        h, rem = divmod(sec, 3600)
        return f"{h}h {rem // 60}m"
    d, rem = divmod(sec, 86400)
    h = rem // 3600
    return f"{d}d {h}h"


def fmt_mem_kb(kb):
    if kb >= 1024**2:
        return f"{kb / 1024**2:.1f}G"
    if kb >= 1024:
        return f"{kb / 1024:.1f}M"
    return f"{kb:.0f}K"


def concurrency(tasks):
    """Compute concurrency timeline. Returns (max_concurrent, duration_at_level)."""
    events = []
    for t in tasks:
        start = to_dt(t["Start"])
        end = to_dt(t["End"])
        if start and end and end > start:
            events.append((start, +1))
            events.append((end, -1))
    events.sort()

    current = 0
    max_c = 0
    prev_t = None
    durations = Counter()  # {concurrent_count: total_seconds}

    for ts, delta in events:
        if prev_t is not None:
            durations[current] += (ts - prev_t).total_seconds()
        current += delta
        max_c = max(max_c, current)
        prev_t = ts

    return max_c, durations


def analyze(jobid):
    tasks = parse_sacct(jobid)
    if not tasks:
        print(f"[{jobid}] no sacct data (too recent? too old?)")
        return

    name = tasks[0]["JobName"]
    print(f"\n=== {jobid}  {name}  ({len(tasks)} tasks) ===")

    # Wall clock
    starts = [to_dt(t["Start"]) for t in tasks]
    ends = [to_dt(t["End"]) for t in tasks]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    if starts and ends:
        wall = (max(ends) - min(starts)).total_seconds()
        print(f"Wall clock:     {fmt_duration(wall)}")
        print(f"First start:    {min(starts).isoformat(sep=' ', timespec='seconds')}")
        print(f"Last end:       {max(ends).isoformat(sep=' ', timespec='seconds')}")
    else:
        wall = 0
        print("Wall clock:     n/a (job may still be running or have no recorded start)")

    # Elapsed / CPU time
    total_elapsed = sum(parse_elapsed(t["Elapsed"]) for t in tasks)
    req_cpus = [int(t["ReqCPUS"]) for t in tasks if t["ReqCPUS"].isdigit()]
    cpu_seconds = sum(parse_elapsed(t["Elapsed"]) * (int(t["ReqCPUS"]) if t["ReqCPUS"].isdigit() else 1) for t in tasks)
    print(f"Total task time: {fmt_duration(total_elapsed)}  (summed across all tasks)")
    print(f"CPU-time:       {fmt_duration(cpu_seconds)}  (task time × cpus per task)")

    # State breakdown
    states = Counter(t["State"].split()[0] for t in tasks)
    print(f"States:         {dict(states)}")

    # Exit codes
    bad = [(t["JobID"], t["ExitCode"]) for t in tasks if t["ExitCode"] not in ("0:0", "")]
    if bad:
        print(f"Non-zero exits: {len(bad)} tasks")
        for jid, code in bad[:5]:
            print(f"    {jid}  exit={code}")
        if len(bad) > 5:
            print(f"    ... and {len(bad) - 5} more")

    # Memory (scavenged from .batch step rows)
    mems = [t.get("MaxRSS_kb", 0) for t in tasks]
    mems = [m for m in mems if m > 0]
    if mems:
        mems.sort()
        print(f"MaxRSS:         min {fmt_mem_kb(mems[0])}, p50 {fmt_mem_kb(mems[len(mems)//2])}, max {fmt_mem_kb(mems[-1])}")
    req_mems = Counter(t["ReqMem"] for t in tasks if t["ReqMem"])
    if req_mems:
        print(f"Requested mem:  {dict(req_mems)}")

    # Concurrency
    if wall > 0:
        max_c, durations = concurrency(tasks)
        total_busy = sum(v for k, v in durations.items() if k > 0)
        print(f"Max concurrent: {max_c}")
        if total_busy > 0:
            for level in sorted(durations.keys(), reverse=True):
                if level == 0:
                    continue
                pct = 100.0 * durations[level] / wall
                bar_len = int(pct / 2)
                bar = "#" * bar_len
                print(f"    {level:4d}x: {pct:5.1f}% of wall  {bar}")
            idle_pct = 100.0 * durations.get(0, 0) / wall
            if idle_pct > 0.1:
                print(f"    idle:  {idle_pct:5.1f}% of wall")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for jid in sys.argv[1:]:
        analyze(jid)


if __name__ == "__main__":
    main()
