# Magneton

Connectomics pipeline for neuron/mitochondria segmentation on HPC (Yale YCRC clusters via SLURM).

## Project Structure

```
magneton/
├── config.yaml                  # Master config pointing to sub-configs
├── toolkit/                     # Data processing tools
│   ├── main.py                  # CLI entry point for toolkit
│   ├── configs/                 # YAML configs for each tool
│   └── tools/                   # split, merge, convert_prec, downsample_prec, etc.
├── pytorch_connectomics/        # Affinity prediction (PyTC training/inference)
│   ├── main.py                  # CLI entry point for PyTC
│   └── configs/                 # HPC and model configs
├── instance_segmentation/       # Segmentation pipeline (waterz or binary_watershed)
│   ├── configs/config.yaml      # Main segmentation config
│   ├── stages/                  # segmentation_stage, merge_apply, merge_pools (+HPC variants)
│   ├── tools/                   # run_local_shard.py, etc.
│   ├── waterz_block.py          # Neuron mode: waterz agglomeration
│   └── mito_block.py            # Mito mode: binary_watershed
├── jobs/                        # SLURM job scripts and logs
│   ├── pytc/                    # PyTC inference jobs
│   ├── merge/                   # Merge stage jobs
│   ├── downsample/              # Downsampling jobs
│   └── convert/                 # H5-to-precomputed conversion jobs
├── claude_notes/                # Daily session logs (log_MM_DD_YYYY.md)
└── checkpoints/                 # Segmentation/merge checkpoint state
```

## Pipeline Flow

1. **Affinity prediction** (`pytorch_connectomics/`) - UNet predicts voxel affinities from EM images
2. **Convert to precomputed** (`toolkit/tools/convert_prec.py`) - H5 output to Neuroglancer precomputed format
3. **Instance segmentation** (`instance_segmentation/`) - Two modes:
   - `neuron`: waterz agglomeration (dense, fills volume)
   - `mito`: binary_watershed (sparse objects, respects background)
4. **Merge** (`instance_segmentation/stages/merge_*`) - Stitch block-level segments into global IDs
5. **Downsample** (`toolkit/tools/downsample_prec.py`) - Generate mip levels via igneous

## Key Conventions

- **Data format**: Neuroglancer precomputed (CloudVolume). Resolution metadata lives in `info` JSON files.
- **HPC**: All heavy computation runs on SLURM. Configs have `hpc:` sections with partition, mem, time, etc.
- **YAML gotcha**: YAML 1.1 interprets unquoted `HH:MM:SS` as sexagesimal integers (e.g., `10:00:00` → `36000`). Always quote time strings or rely on `_fix_yaml_time()` in `pytorch_connectomics/main.py`.
- **SLURM memory**: `SLURM_MEM_PER_CPU` is always a plain number in MB (no unit suffix). Code in `downsample_prec.py` appends `'M'` before parsing.
- **Session logs**: Written to `claude_notes/log_MM_DD_YYYY.md` after each session. Record only code/script changes, not manual file operations.

## Common Commands

```bash
# Toolkit operations (split, merge, convert, downsample)
python toolkit/main.py --config config.yaml

# PyTC inference
python pytorch_connectomics/main.py --config config.yaml

# Instance segmentation
python instance_segmentation/main.py --config instance_segmentation/configs/config.yaml
```

---

# Session Logs

## February 2, 2026

### Summary
Added mitochondria segmentation mode using `binary_watershed` as an alternative to waterz, which is better suited for sparse objects like mitochondria.

### Changes Made

**New File: `instance_segmentation/mito_block.py`**
- `binary_watershed()` - Seeds from high-confidence regions (prob > seed_threshold), foreground mask from moderate-confidence regions (prob > foreground_threshold), watershed grows from seeds bounded by mask
- `run_mito_block()` - Entry point matching `run_waterz_block` signature

**Modified: `instance_segmentation/configs/config.yaml`**
- Added `mode` section with `type: "neuron"` or `"mito"` and mito-specific parameters (seed_threshold, foreground_threshold, min_segment_size, seed_min_size, remove_small_mode)

**Modified: `instance_segmentation/stages/segmentation_stage.py`**
- Added mode config parsing and branching: mito mode calls `run_mito_block`, neuron mode calls `run_waterz_block`

**Modified: `instance_segmentation/tools/run_local_shard.py`**
- Added mode config parsing, updated `_process_block` call to pass `mode_cfg`

**Modified: `instance_segmentation/waterz_block.py`**
- Added single-channel affinity handling (duplicates to 3 channels for waterz compatibility)

---

## February 3, 2026

### Summary
Fixed downsample queue monitoring bug that caused premature job termination, added dynamic memory allocation for igneous, and fixed typo bugs causing config paths to be ignored.

### Changes Made

**Modified: `toolkit/tools/downsample_prec.py`**
- Queue monitoring: added `get_queue_count()`, activity timer now resets on file modifications OR queue count changes, increased idle timeout from 60s to 300s
- New `parse_memory_string()` and `calculate_memory_target()` for dynamic SLURM-aware memory allocation

**Modified: `instance_segmentation/stages/segmentation_stage_hpc.py`, `merge_apply_hpc.py`, `merge_pools_hpc.py`**
- Typo fix: `cfg.get("mian", ...)` → `cfg.get("main", ...)`

**Modified: `instance_segmentation/tools/run_local_shard.py`**
- Config key fix: `get_stage_config(cfg, "segmentation_stage")` → `get_stage_config(cfg, "segmentation")`

---

## February 4, 2026

### Summary
Fixed YAML sexagesimal bug that corrupted HPC time values (e.g., `10:00:00` → `36000`).

### Changes Made

**Modified: `pytorch_connectomics/main.py`**
- New `_fix_yaml_time()`: detects integer `time` values in config dicts, converts seconds back to HH:MM:SS format, recursively processes nested dicts
- Updated `edit_stage_config()` to silently repair corrupted time values on every run

**Modified: `pytorch_connectomics/configs/hpc_f_bouton.yaml`**
- Restored `time: "10:00:00"` (was corrupted to `time: 36000`)

---

## February 5, 2026

### Summary
Fixed SLURM memory detection showing 0.0GB in downsample jobs, and corrected display units from decimal GB to binary GiB.

### Changes Made

**Modified: `toolkit/tools/downsample_prec.py`**
- Bug fix: `SLURM_MEM_PER_CPU` is a plain number in MB but `parse_memory_string` treated it as bytes. Fix: append `'M'` when value has no unit suffix
- Bug fix: Display divided by `1e9` (decimal GB) instead of `1024**3` (binary GiB), causing 32G to show as 34.4GB. Fixed to use `1024**3`
