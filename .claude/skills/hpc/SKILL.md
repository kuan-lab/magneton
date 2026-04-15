---
name: hpc
description: Inspect Yale YCRC HPC state and estimate job wait time. Two sub-commands — `/hpc info [partition...]` shows partition availability and queue depth; `/hpc time [cpu|gpu]` uses `sbatch --test-only` to estimate wait time for a standard probe job profile. Invoked as `/hpc <subcommand> [args]`.
---

# /hpc

Two read-only sub-commands for HPC planning on the Yale YCRC cluster. Neither actually submits work.

## Partitions of interest

`day`, `week`, `gpu`, `bigmem`. If the user's English argument names one or more partitions, filter to those. Otherwise, show all four.

## `/hpc info [partition...]`

Snapshot of partition availability and current queue depth.

### Steps

1. **Parse the argument**: extract any of `day`, `week`, `gpu`, `bigmem` mentioned in the argument. If none are mentioned, default to all four.

2. **Run sinfo** for the selected partitions:
   ```bash
   sinfo -o "%20P %5a %.11l %.6D %.15F %.15C %25G" -p <comma-separated>
   ```
   Columns: PARTITION, AVAIL, TIMELIMIT, NODES, NODES(A/I/O/T), CPUS(A/I/O/T), GRES.
   - `NODES(A/I/O/T)` = Allocated / Idle / Other / Total
   - `CPUS(A/I/O/T)` = same breakdown for CPUs
   - `GRES` = generic resources (shows GPU type + count for gpu partition)

3. **Run squeue** to count queue depth per partition:
   ```bash
   squeue -p <partition> -h -t R  | wc -l   # running
   squeue -p <partition> -h -t PD | wc -l   # pending
   ```
   Do this once per partition in the filter.

4. **Present a compact table** with one row per partition (or per GPU-type row for the gpu partition, which appears multiple times in sinfo output):

   ```
   Partition  Nodes (idle/total)  CPUs (free/total)  GPUs         Queue (R / PD)
   day        6/14                675/896            —            30 / 45
   week       2/10                573/640            —            12 / 8
   bigmem     1/2                 111/128            —            5 / 2
   gpu/h100   0/14                354/672            gpu:h100:4   18 / 22
   gpu/h200   0/4                 96/192             gpu:h200:4   4 / 0
   ...
   ```

   Note: the `gpu` partition has **multiple sinfo rows**, one per GPU type. Preserve that breakdown — don't collapse them, since GPU type matters for choosing where to submit.

5. **Highlight anything notable** in a short one-liner after the table:
   - Any partition with 0 idle nodes (capacity-bound)
   - Any partition with PD > 2×R (backlog-bound)
   - Any partition with low free CPUs relative to queue depth

   Keep the commentary to a sentence or two — don't over-explain.

## `/hpc time [cpu|gpu]`

Measure real wait time for a standard probe job profile by actually submitting a trivial `sleep 1` job, polling `squeue` until it transitions PENDING → RUNNING, and reporting how long that took. The helper script at `.claude/skills/hpc/probe.sh` handles submission, polling, cleanup, and reporting.

**Why not `sbatch --test-only`?** It returns SLURM's main-schedule estimate and ignores backfill eligibility. For small backfill-eligible jobs on a partition with idle CPUs, `--test-only` gives wildly pessimistic estimates (e.g., reporting "1.5 day wait" when real wait is 10 seconds). Only real submission + polling gives the truth.

### Profiles

Two hardcoded probe profiles drawn from existing magneton configs. If these defaults drift (because the source configs changed), update them here.

**`cpu` profile** — drawn from `instance_segmentation/configs/config_30tb.yaml` segmentation stage (right-sized for production waterz):
- `--partition=day`
- `--time=5:00:00`
- `--cpus-per-task=2`
- `--mem=12G`

**`gpu` profile** — drawn from `pytorch_connectomics/configs/hpc_f.yaml` (pytc inference):
- `--partition=gpu`
- `--time=4:00:00`
- `--cpus-per-task=8`
- `--mem-per-gpu=64G`
- `--gres=gpu:h100:1`

### Steps

1. **Parse the argument**:
   - No arg → run both `cpu` and `gpu` probes in parallel (two Bash tool calls in one message).
   - `cpu` → run just the CPU probe.
   - `gpu` → run just the GPU probe.

2. **For each profile**, invoke the probe helper:
   ```bash
   .claude/skills/hpc/probe.sh cpu --partition=day --time=5:00:00 --cpus-per-task=2 --mem=12G
   .claude/skills/hpc/probe.sh gpu --partition=gpu --time=4:00:00 --cpus-per-task=8 --mem-per-gpu=64G --gres=gpu:h100:1
   ```
   The script:
   - Submits `sbatch <flags> --wrap="sleep 1"` (a real job that finishes on its own in 1 second once it starts).
   - Polls `squeue` every 2 seconds until state is RUNNING, or the job has already left the queue.
   - If the job left the queue between polls, recovers the actual Start time via `sacct`.
   - If it doesn't transition within 300 seconds (`TIMEOUT_SEC` env override available), calls `scancel` on the pending job and reports the pending reason.
   - Also traps SIGINT/SIGTERM to scancel cleanly if interrupted.

3. **Expected output** from the script, one line per probe:
   ```
   [cpu] submitted 1502962 (flags: --partition=day --time=5:00:00 --cpus-per-task=2 --mem=12G)
   [cpu] started after ~10s
   ```
   or on timeout:
   ```
   [cpu] still PENDING after 300s (reason: Resources); cancelled
   ```

4. **Running both in parallel**: when `/hpc time` is invoked with no arg, issue both Bash tool calls in the same message so they run concurrently. The script on each side handles its own lifecycle independently.

5. **Do not report on `--test-only` output anywhere** — it's misleading for small jobs. This skill uses real submission exclusively.

## Output style

- `/hpc info` and `/hpc time` are read-only diagnostics. No draft-review loop, no approval step. Run the commands, format the table, return.
- Keep the output compact. This skill is about quick situational awareness.
- Don't mix the two sub-commands in one response unless the user explicitly asks for both.

## Keeping this skill current

The `cpu` and `gpu` probe profiles are **hardcoded defaults** drawn from specific magneton configs. If those source configs change (e.g., `config_30tb.yaml` seg stage is re-tuned, or `hpc_f.yaml` GPU type changes from h100), this skill's probe profiles become stale.

**Claude's responsibility**: whenever a session edits `instance_segmentation/configs/config_30tb.yaml` or `pytorch_connectomics/configs/hpc_f.yaml` in a way that changes the HPC resource block, proactively ask the user: "Should I update the `/hpc time` probe profiles to match the new resources?"

New partitions on YCRC (if YCRC adds one) or new GPU types also require updating this skill — ask the user when you see such a change.
