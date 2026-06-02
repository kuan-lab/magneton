---
name: globus
description: Transfer a dataset between Globus endpoints (defaults — source = Yale CRC Misha, destination = KuanLab nginx storage at `/nginx_share/marmoset/`). Parses the natural-language argument to identify which precomputed dataset to move, builds the exact globus CLI command, and confirms with the user before submitting. Invoked as `/globus <english>`, e.g. "/globus fib_b mito instances v3" or "/globus move the new bouton affinity over".
---

# /globus

End-to-end natural-language wrapper around `globus transfer`. The user types `/globus <english>` and the skill:

1. Resolves the source dataset path (default endpoint = Misha)
2. Resolves the destination path (default endpoint = KuanLab nginx at `/nginx_share/marmoset/`)
3. Confirms with the user via `AskUserQuestion` (showing src→dst), unless the request is unambiguous and small (<1 GB)
4. Submits the transfer using the CLI, optionally waits + reports

## Endpoint UUIDs

Hardcoded — both shared via gpfs in the user's `~/.globus/cli/`:

- **Yale CRC Misha** (source): `48c1bca8-7510-4b82-90e1-d38a42a98372`
- **KuanLab nginx** (destination): `036f47b0-a5d9-4e7c-82e7-6e0c0a3a6f20`

These are also documented in the auto-memory under `reference_globus_endpoints.md`. If the user names a different endpoint in the English (e.g., "transfer to misha", "from radev to nginx"), parse and use that instead.

## CLI access

`globus` binary lives at `/gpfs/radev/home/yf354/.conda/envs/yf354/bin/globus`. Use the full path — not on default PATH on radev. Auth tokens are in `~/.globus/cli/` (gpfs-shared, so radev can run transfers using the user's misha login).

## Invocation flow

### Step 1 — Parse the English

Identify the dataset name(s) and any endpoint or path overrides. Examples:

| English | Interpretation |
|---|---|
| "fib_b mito instances v3" | source path `precomputed_outputs/fib_b_mito_instances_v3` → dest `/nginx_share/marmoset/fib_b_mito_instances_v3` |
| "the new bouton affinity" | look at most recent inference output dir matching `*bouton*v2*` |
| "move my latest mito test over" | look at most recent `*mito*test*` output |
| "from radev to misha" | source = `radev` (gpfs share), dest = Misha endpoint |
| "fib_c neuron instances v2 to my home" | dest = a path under `/gpfs/radev/home/yf354/` |

For ambiguous paths, list candidates and ask via `AskUserQuestion` before transfer.

### Step 2 — Resolve source path

Default search root: `/gpfs/marilyn/pi/kuan/shared/marmoset_project/precomputed_outputs/`. Use `ls`/`find` to find an exact match for the dataset token. If multiple candidates exist (e.g., `fib_b_mito_instances_iso_v1` vs `fib_b_mito_instances_iso_v1_e4`), pick the newest by mtime and confirm with the user.

If user mentions "the new <X>" or "the latest <X>", sort matches by mtime descending and pick the most recent.

### Step 3 — Resolve destination path

Default: `/nginx_share/marmoset/<basename_of_source>`. If a different parent dir is implied by the English, use that. Always preserve the source basename unless explicitly told otherwise — that's the convention used for everything else in `marmoset/`.

### Step 4 — Confirm via `AskUserQuestion`

Build a single confirmation question with the actual paths, e.g.:

> "Transfer `fib_b_mito_instances_v3` (114 GB) from Misha:`/gpfs/marilyn/.../precomputed_outputs/fib_b_mito_instances_v3` → KuanLab:`/nginx_share/marmoset/fib_b_mito_instances_v3`?"

Options: **Yes / Yes (and wait for completion) / No / Adjust path**.

Skip the confirmation only if (a) the transfer is small (<1 GB), (b) the source/dest are unambiguous, and (c) the dest doesn't already exist at the destination — when all three hold, just go.

If the dest already exists at the destination, surface that in the confirmation. Options become **Yes (overwrite/skip-existing) / No / Adjust path**.

### Step 5 — Submit + report

Build the command:

```bash
~/.conda/envs/yf354/bin/globus transfer \
  <src_uuid>:<src_path> \
  <dst_uuid>:<dst_path> \
  --recursive \
  --label "<dataset_name>"
```

Capture the task ID from output (first line: `Task ID: <uuid>`).

If the user opted for "wait for completion": run `globus task wait <task_id> --timeout <appropriate>` in the background, then `globus task show` for the final stats (Status / Files / Bytes Transferred / Faults). Report concisely.

If not waiting: just print the Task ID and tell the user how to monitor: `globus task show <task_id>` or `globus task list`.

## Common helpers

- **Check dataset existence** before transfer: `ls -ld <source>` to confirm and get size with `du -sh <source>`.
- **Check dest collision**: `globus ls <dst_uuid>:<dst_path>` — returns 404 if not present (clean transfer), or contents if it exists (warn user).
- **Estimate wall time**: at observed ~160–300 MB/s steady state, GB / 0.2 = approx seconds. Use this for the `--timeout` if waiting.

## Gotchas

- **Per-collection consent**: first time touching a new collection on this account, you'll hit `consent required` errors. Surface that to the user with the exact `globus session consent ...` command — they need a browser to run it. After that, it's persistent in their Globus Auth account.
- **Path-prefix differences**: Misha sees `/gpfs/marilyn/pi/...` from its endpoint root; radev sees the same data via `/gpfs/radev/.marilyn/pi/...`. Use the Misha-flavor path when transferring FROM Misha endpoint.
- **Big transfers (>50 GB)**: don't wait synchronously — submit + give Task ID. The user can monitor with `globus task show`.
- **Existing partial transfers**: if a previous transfer failed mid-way, Globus's `--sync-level checksum` will only re-send changed/missing files. Suggest it when re-running a failed transfer.

## Saved-memory references

- `reference_globus_endpoints.md` — endpoint UUIDs, CLI install pattern, transfer command template
- `reference_data_layout.md` — where datasets live (precomputed_outputs root, naming conventions)
