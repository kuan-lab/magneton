---
name: wk
description: Upload an EM volume (+ optional instance segmentation and/or skeleton NML) to webknossos.org. Parses the natural-language argument to find the image/seg/skeleton files, resolves the target WKS folder, confirms with the user, then runs upload_webknossos.py. Invoked as `/wk <english>`, e.g. "/wk upload the fennel neuron GT with its skeleton to the Skeleton-Test folder" or "/wk push the basil EM + seg".
---

# /wk

End-to-end natural-language wrapper around `/gpfs/radev/home/yf354/magneton/upload_webknossos.py`. The user types `/wk <english>` and the skill:

1. Resolves the EM (`--img-files`), optional segmentation (`--seg-files`), and optional skeleton NML (`--skel-files`)
2. Resolves the dataset name (`--names`) and target WKS folder (`--remote-folder`)
3. Confirms via `AskUserQuestion` (uploading publishes to webknossos.org — always confirm)
4. Runs the upload in the **yf354** env, then reports the dataset + annotation URLs

## Environment & auth

- **Run in the `yf354` conda env** (not magneton) — webknossos-libs lives there. Use `conda run -n yf354 python ...`.
- **Token**: pass `--token "$WEBKNOSSOS_TOKEN"`. If `WEBKNOSSOS_TOKEN` is unset, ask the user to paste it (via `AskUserQuestion` or plain prompt) — NEVER hardcode a token in this skill or any file. It's a secret.
- Instance defaults to webknossos.org.

## What the script does (already handles the gotchas)

- `--img-files` → a **color layer** "volume". `--seg-files` → a uint32 **volume annotation** layer "segmentation", and every label is **pre-registered as a segment** (named `neuron_<id>`) so the Segments panel is populated. `--skel-files` → an NML **skeleton** attached to the same annotation.
- TIFs are read with `moveaxis (z,y,x)→(x,y,z)`; keep skeleton NML coords in (x,y,z) voxels.
- It auto-aligns `skel.dataset_name`/`dataset_id` to the uploaded dataset (the NML `<experiment name>` is otherwise read by WKS as a dataset name → 400 error).
- `--names` overrides the dataset name (default = EM filename stem). `--out-dir` is a local scratch dir for the wkw build (default `wk_upload_out`).

## Invocation flow

### Step 1 — Parse the English

Identify the EM, seg, skeleton, dataset name, and folder. Examples:

| English | Interpretation |
|---|---|
| "fennel neuron GT with skeleton" | EM + `_neuron_seg_v2.tif` from `wk_crops/neuron_myelin_gt/...fennel...`, skeleton = most recent `*.nml` in repo root |
| "basil EM + seg" | basil crop's `_em.tif` + `_seg.tif`; no skeleton |
| "just the EM for <crop>" | only `--img-files` |
| "...to the Skeleton-Test folder" | resolve folder (see Step 3) |
| "...named fennel_test_v4" | `--names fennel_test_v4` |

### Step 2 — Resolve files

Default search roots (use `find`/`ls`):
- GT crops: `/gpfs/marilyn/pi/kuan/shared/marmoset_project/wk_crops/` and `/gpfs/marilyn/pi/kuan/shared/marmoset_project/model_ground_truth/fibsem/{em,labels}/`
- Precomputed-derived TIFs / skeletons produced this session: repo root `/gpfs/radev/home/yf354/magneton/*.nml`

Pairing: an `_em.tif` pairs with the same-stem `_seg.tif` / `_neuron_seg_v2.tif` (prefer the proofread `_v2`). Confirm the voxel size from the README (`model_ground_truth/fibsem/README.md`: mip0 = 4 nm, mip1 = 8 nm) — pass `--voxel-sizes "(x,y,z)"`. If ambiguous, list candidates and ask via `AskUserQuestion`.

Note: nnInteractive/precomputed inputs may need conversion to TIF first; the script reads `.tif/.tiff`, `.h5/.hdf5` (key `exported_data`), and `.nii.gz`.

### Step 3 — Resolve the WKS folder

`--remote-folder` accepts a path OR a folder id. **Folder id from a dashboard URL**: `https://webknossos.org/dashboard/datasets/<Name>-<24hexid>` → the trailing 24-hex string is the id. If the user pastes a URL, extract the id. If they name a folder ("Skeleton-Test"), pass the name as the path (the script falls back to `get_by_id` on `KeyError`). If omitted, upload goes to the org default location.

### Step 4 — Confirm via `AskUserQuestion`

Show what will upload, e.g.:

> "Upload dataset `fennel_neuron_skel_test_v4` to folder Skeleton-Test? Layers: EM (color), seg (208 segments), skeleton (210 trees)."

Options: **Yes / No / Adjust (name/folder)**. Always confirm — this publishes to webknossos.org.

Heads-up to surface: dataset names must be **unique per org** — if the chosen name already exists, the upload 400s; bump the name or have the user delete the old one in the UI.

### Step 5 — Run + report

```bash
conda run -n yf354 python /gpfs/radev/home/yf354/magneton/upload_webknossos.py \
  --img-files  <em.tif> \
  --seg-files  <seg.tif> \
  --skel-files <skel.nml> \
  --voxel-sizes "(8,8,8)" \
  --names      <dataset_name> \
  --remote-folder <folder-id-or-path> \
  --out-dir    /gpfs/radev/home/yf354/magneton/wk_upload_out \
  --token      "$WEBKNOSSOS_TOKEN"
```

Run in the background (downsample + upload takes ~30–60s for a GT crop; longer for big volumes). Report the two URLs from the output: `Uploaded '<name>' to <dataset-url>` and `Annotation: <annotation-url>`. Omit `--seg-files`/`--skel-files` if not requested (both optional; `--img-files` is required).

## Gotchas

- Background shells don't auto-load conda — when running detached, source `/gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh` first, or use `conda run -n yf354` in a foreground call.
- The webknossos-libs version (3.1.0) prints an "outdated" notice — harmless; filter it from output.
- A failed annotation upload can leave an EM-only dataset orphaned remotely — note it for cleanup in the UI.

## Saved-memory references

- `reference_webknossos_upload.md` — script args, folder-id-from-URL, NML experiment-name=dataset-name gotcha, segment-list registration
- `feedback_webknossos_env.md` — use the yf354 env for webknossos
- `reference_data_layout.md` — where EM/seg/GT crops live
