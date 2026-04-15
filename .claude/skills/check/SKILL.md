---
name: check
description: Check the status and results of a SLURM job referenced in conversation — running state, log errors, memory usage, wall-clock time, array concurrency breakdown, and whether output was produced. Also handles historical lookups and side-by-side job comparisons. Invoked explicitly as `/check <english>`, e.g. "/check how's my seg job going" or "/check compare the last two fennel merge jobs".
---

# /check

Answers questions about SLURM jobs the user submitted. Parses stage + intent from the English argument, locates the right `jobs/<suffix>/` directory and job ID(s), queries `squeue`/`sacct`, runs the analyzer, scans logs, and reports.

## Stage → job_dir mapping

Job directories are **suffixed per submission**, not just by stage. A single stage keyword usually matches multiple directories. The full list at scan time is `ls jobs/` — match against it dynamically using the table below as the keyword legend.

| English keyword(s) | `jobs/` dir prefix |
|---|---|
| `seg`, `segmentation`, `waterz`, `mito seg` | `seg`, `seg_*` |
| `merge` (instance seg merge) | `merge`, `merge_*` |
| `pytc`, `inference`, `training`, `affinity` | `pytc` |
| `convert`, `prec`, `to precomputed` | `convert` |
| `downsample`, `mip`, `igneous` | `downsample` |
| `crop` | `crop` |
| `split` | `split` |
| `resize` | `resize` |
| `gen_mask`, `generate mask` | `gen_mask` |
| `mask_prec` | `mask_prec` |
| `mask_tif` | `mask_tif` |

Note: `jobs/merge/` is the *toolkit* merge-volumes tool. `jobs/merge_*/` suffixed dirs are almost always *instance_segmentation* merge stage (from `merge_stage.hpc.job_dir` in the instance seg config). The keyword "merge" alone is ambiguous — infer from context (if the user is talking about segmentation/waterz flow, it's instance seg merge; if they just converted a volume, it's toolkit merge).

## Job directory layout

Every `jobs/<suffix>/` contains:
- `submit_slurm.sh` — the SBATCH script. Contains `--job-name`, `--array=0-N%C`, `#SBATCH --output=...%x_%A_%a.out`, and the exact `python ... --config /path/to/config.yaml` line that was launched.
- `manifest.txt` — list of shard indices, one per array task.
- `logs/` — directory with per-task log files.

## Log filename format

`<job_name>_<arrayJobID>_<arrayTaskIdx>.{out,err}`

Example: `segmentation_chunks_1454510_6.out` = job name `segmentation_chunks`, SLURM array job ID `1454510`, array task `6`.

To get the array job ID from a jobs dir, `ls logs/` and parse the middle field of the newest filename. Multiple distinct job IDs in the same dir means the user resubmitted — each resubmit gets its own `%A`.

## Invocation flow

### Step 1 — Parse the English

From the argument, extract:
- **Stage** — match against the keyword table above. If none found, look at `squeue -u yf354` for the user's currently-running jobs and see if one jumps out ("top" of the queue = highest array job ID, or newest by submit time).
- **Recency** — "just submitted" / "currently running" / "how's it going" → current. "last week" / "the fennel one" / "compare to older" → historical. Default: current.
- **Comparison intent** — "compare X and Y" / "vs the older one" / "how does this compare" → multi-job comparison mode.
- **Key phrases** — dataset names (`fennel`, `fib_b`, `30tb`, `bouton_c`, `wztest`), config suffixes, anything that could narrow which `jobs/<suffix>/` dir is meant.

If any of stage / intent is unclear, ask a short clarifying question (don't assume).

### Step 2 — Locate the jobs directory

1. `ls jobs/` and filter by the stage keyword's dir prefix(es).
2. Narrow using key phrases from the English (e.g., "fennel" → `seg_fennel`, `seg_fennel_wztest`).
3. If still multiple candidates:
   - For **current** queries: pick the one with the newest file in `logs/` (most recent activity).
   - For **historical** queries with multiple candidates: list them compactly and ask which.
4. For comparison mode: collect all matching dirs; proceed to analyze each.

### Step 3 — Find the job ID(s)

In the chosen `jobs/<suffix>/logs/`:
- Parse array job IDs from log filenames (`<name>_<jobid>_<task>.out`). Collect the set of distinct `<jobid>` values.
- If the user asked about "the most recent", pick the newest `<jobid>` by log file mtime.
- If there are multiple distinct job IDs (resubmits), and the user asked about history, list them briefly and ask which unless the English specifies.

### Step 4 — Current state check

For each job ID:
1. `squeue -u yf354 -j <jobid> -h -o "%T %M %l %C %m %R"` — state, elapsed, timelimit, cpus, mem, reason/nodelist. If running/pending, show these.
2. If not in `squeue`, it's finished or too old → proceed to sacct analysis.

### Step 5 — Historical analysis

Run the analyzer helper:
```bash
python3 .claude/skills/check/analyze.py <jobid> [<jobid> ...]
```

The helper reports:
- Wall clock (first start → last end across array tasks)
- Total task time (summed elapsed)
- CPU-time (task time × requested CPUs)
- State breakdown
- Non-zero exit codes with per-task details (first 5)
- MaxRSS distribution (min / p50 / max) — scavenged from `.batch` step rows
- Requested memory
- **Concurrency timeline**: max concurrent, and % of wall time spent at each concurrency level

For multi-job comparison, pass all job IDs at once — each is reported in its own block. Then the skill should extract the key metrics (wall clock, max concurrent, MaxRSS max, state) and add a side-by-side summary table after the analyzer output.

### Step 6 — Error scan

For each job, scan logs for error patterns:
```bash
grep -nEi "traceback|error|failed|cancelled|killed|out of memory|oom" \
     jobs/<suffix>/logs/*_<jobid>_*.{out,err} 2>/dev/null | head -30
```

Report findings:
- If zero matches: "no errors detected in logs".
- If matches: show the first ~10 lines with file:line context, grouped by task index.
- Be careful — some normal output contains "Error" as a substring (e.g., "Error-free completion"). Show context, let the user judge.

### Step 7 — Output integrity check

1. Read `jobs/<suffix>/submit_slurm.sh` and grep for `--config` to find the config file path used.
2. Read that config to resolve output path(s). Stage-specific keys:
   - **instance_segmentation**: `paths.output` (global), `paths.output_local_base` (per-block), `checkpoint.segmentation_dir`, `seg_metadata_*/`
   - **pytc**: `INFERENCE.OUTPUT_PATH` + `INFERENCE.OUTPUT_NAME` (the h5 file) or precomputed dir
   - **toolkit prec/convert**: `paths.output`
   - **toolkit split**: `split.output`
   - **toolkit merge (volumes)**: `merge.output`
   - **toolkit downsample**: `downsample.source_path` — mip levels written in place
   - **toolkit crop**: `crop.output`
   - **instance_segmentation merge_stage**: `paths.output`, `checkpoint.merge_dir`
3. Strip any `file://` prefix and check the path:
   - Exists?
   - Non-empty (file size > 0, or directory has entries)?
   - Precomputed dirs: has an `info` JSON and at least one mip subdir with chunks?
   - h5 outputs: file exists, non-zero size
   - Per-block outputs (instance seg): count directories under `output_local_base` parent and compare against expected block count from `block.size` / total volume (if derivable); otherwise just report count.
4. Report concisely: "output written to `<path>`: <N files, <size>, info.json ✓" or "output path empty".

### Step 8 — Report

Present a single cohesive block, not a wall of headers. Structure:
1. One-line summary (state + wall time + task count + location)
2. Any errors, or "no errors in logs"
3. Analyzer output (already formatted)
4. Output integrity one-liner
5. Log tail (last ~20 lines of the newest `.out` file) only if explicitly asked or if there were errors

## Comparison mode

When the user says "compare X with Y" or similar:
1. Resolve all referenced jobs per Steps 2–3.
2. Run analyzer with all job IDs in one call.
3. After the analyzer blocks, produce a side-by-side summary table:
   ```
   JobID     Suffix                Wall      Tasks  MaxConc  MaxRSS  State
   1454510   seg_fennel_wztest     2m 14s    8      3        1.5G    COMPLETED
   1401234   seg_fennel            4m 30s    8      4        2.1G    COMPLETED
   ```
4. Add a one-line commentary on the main difference (wall clock, concurrency utilization, memory, or state).

## Gotchas

- **`merge` ambiguity**: `jobs/merge/` is toolkit merge-volumes; `jobs/merge_<suffix>/` is instance_segmentation merge stage. Check context.
- **Array vs plain**: non-array jobs have log filenames like `<name>_<jobid>.out` (no `_<task>` suffix). Parser must handle both.
- **Resubmits**: a single jobs dir can hold multiple distinct `<arrayJobID>` values from multiple submissions. Always check the set of IDs, don't assume one.
- **MaxRSS missing**: `sacct -X` excludes step rows where MaxRSS lives. The analyzer already removes `-X` and folds step-level MaxRSS back onto parent rows — do not "fix" it back.
- **Wall clock vs total task time**: wall clock is `max(End) - min(Start)` across all array tasks. Total task time is the sum. If wall clock ≈ total task time and there are many tasks, concurrency is near 1 (serial). If wall clock << total task time, concurrency is high. The analyzer's concurrency timeline makes this explicit.
- **`idle` concurrency**: the analyzer reports "idle: X%" when events leave a gap. This usually means tasks didn't start back-to-back — scheduler delay or intentional throttling via `%N` array cap.

## Keeping this skill current

The stage→dir keyword table is static. When a new pipeline stage is added to magneton, the keywords and `jobs/` prefix won't be in the table and the skill will fail to locate the job.

**Claude's responsibility**: whenever a session adds a new pipeline stage (new `hpc.job_dir` value in a config, new tool in `toolkit/tools/`, new stage script), proactively ask the user: "Should I update the `/check` skill's stage→dir table since we added `<thing>`?" (This is also recorded in the auto-memory under `feedback_skill_update_prompts.md`.)

The analyzer script (`analyze.py`) is stable and stage-agnostic — it only needs updating if SLURM's sacct field set or output format changes.
