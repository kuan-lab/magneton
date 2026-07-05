# Magneton

**Magneton** is a connectomics segmentation and analysis pipeline developed by [Kuan Lab](https://www.kuanlab.org/) for neuron and organelle (mitochondria, bouton, synapse) processing of large-scale 3D EM data. It runs as a command-line toolbox on Linux + NVIDIA GPUs, with every heavy step submittable to an HPC (SLURM) cluster.

Full rendered docs: [magneton.readthedocs.io](https://magneton.readthedocs.io/en/latest/) (source under [`docs/source/`](docs/source/)).

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration) — how every tool is driven
- [Usage](#usage) — running modules locally or on HPC
- [The processing toolkit](#the-processing-toolkit)
- [Optional: Claude Code assistant tools](#optional-claude-code-assistant-tools)
- [Repository layout](#repository-layout)

## Overview

The pipeline is organized into five modules. Each is runnable from one unified CLI (`python -m magneton`) or from its own entry point.

| # | Module | What it does |
|---|--------|--------------|
| 1 | **Processing toolkit** | Split / merge / convert / downsample / mask / resize / crop volumes, and generate meshes. The glue around the ML steps. |
| 2 | **Affinity-map inference** | Train / fine-tune a model and predict voxel **affinity maps** from EM, block-wise over large volumes. Built on [PyTorch Connectomics](https://connectomics.readthedocs.io/en/latest/index.html). |
| 3 | **Instance segmentation** | Turn affinity maps into labeled instances per block, then merge blocks into global IDs. Modes: `neuron` (waterz), `mito`/`synapse` (binary watershed), `bouton` (watershed + neuron-membrane gating). |
| 4 | **Analysis** | Per-instance morphometrics for any sparse organelle (mito/bouton/synapse): discover bounding boxes → compute features → `morphometrics.parquet`, plus embedding and cross-volume relational analysis. |
| 5 | **Proofreading** | Human-in-the-loop ground-truth correction in [WebKnossos](https://webknossos.org/): the **membrane** entry (crop EM + affinity → binary membrane → upload) and the **skeletonize** entry (instance seg → NML). |

Typical flow:

```
EM (precomputed)
  └─ 2. Affinity inference          →  fib_X_inference_<model>_v<N>/
       └─ 1. convert / downsample
       └─ 3. Instance segmentation  →  per-block outputs
            └─ 3. merge             →  fib_X_<model>_instances_v<N>/
                 └─ 1. downsample / mesh
                 └─ 4. Analysis     →  morphometrics.parquet
                 └─ 5. Proofreading →  WebKnossos correction
```

Data is stored as Neuroglancer **precomputed** volumes (via CloudVolume); resolution and geometry live in each volume's `info` JSON.

## Installation

The pipeline uses **conda environments**. On the lab's Yale cluster these already exist — you mostly just `conda activate` them. To build from scratch:

### Main environment (`magneton`)
Runs the toolkit, affinity inference, instance segmentation, analysis, and the crop/threshold/skeletonize halves of proofreading.

```bash
conda create -y -n magneton python=3.9
conda activate magneton

# PyTorch with the matching CUDA (H100/A100 → 12.4):
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# conda pulls in MKL 2025.x, which breaks `import torch`
# (ImportError: undefined symbol: iJIT_NotifyEvent). Pin MKL back to 2024.0:
conda install -y "mkl=2024.0" -c conda-forge
# boost headers are required to compile the waterz extension:
conda install -y boost -c conda-forge

git clone https://github.com/kuan-lab/magneton.git
cd magneton
pip install --editable .                 # magneton + toolkit + analysis + proofreading
cd pytorch_connectomics && pip install --editable . && cd ..   # affinity inference (also installs Cython)
# waterz's setup.py imports Cython + numpy at build time, so build it WITHOUT pip's
# build isolation — the previous step already put Cython + numpy in the env:
cd waterz && pip install --editable . --no-build-isolation && cd ..   # instance segmentation

# skeletonize proofreading needs wknml (not pulled in by anything above):
pip install wknml
# wknml's install bumps numpy to 2.x; monai and the waterz/mahotas C-extensions
# need numpy<2, so re-pin numpy LAST:
pip install "numpy==1.26.4"
```

Key packages that end up in this env: `torch`, `cloudvolume`, `waterz`, `igneous` (downsample/mesh), `kimimaro` + `wknml` (skeletonize), `connectomics`, `tifffile`.

> The first time a `neuron`-mode segmentation runs, `waterz` JIT-compiles its C++ backend (with `boost` + your system `g++`) into `~/.cython/inline/` — expect a one-off compile on the first block, then it is cached.

**Other build errors that can show up on some systems** (the steps above already cover the ones you are certain to hit):

| Error | Fix |
|-------|-----|
| `libstdc++.so.6: version GLIBCXX_3.4.32 not found` | `conda install -c conda-forge libstdcxx-ng` |
| `'PyDataType_ELSIZE' was not declared in this scope` (waterz build, numpy too old) | `pip install --upgrade numpy && pip install numpy==1.26.4` |
| `No module named 'mahotas'` (only if `pip install --editable .` was skipped) | `pip install mahotas` |

### WebKnossos / upload environment
The WebKnossos upload step needs `webknossos` and `SimpleITK`, which are **not** in the `magneton` env. Create a separate environment for it (name it whatever you like — `wks` below):

```bash
conda create -y -n wks python=3.11
conda activate wks
pip install webknossos SimpleITK cloudvolume tifffile
```

The upload tools refer to this env **by name**: set `upload.env` in the proofreading config to your env's name (the membrane stage runs `conda run -n <upload.env> …`), and likewise for `upload_webknossos.py` / the `/wk` skill.

> A separate GPU env (`nninteractive`) exists for the experimental skeleton-expand step, which is **not currently functional on EM** — ignore it unless you are specifically working on that.

## Quick start

```bash
conda activate magneton
python -m magneton          # interactive menu: pick a module (1–5), then a stage
```

The menu shows the five modules; each module has its own sub-menu of stages, and most stages offer a **local** vs **HPC** option. Before anything runs, you choose (or edit) the config that drives it — see below.

## Configuration

**Everything in Magneton is driven by YAML config files**, and a single root file decides which config is "active" for each tool.

### The root `config.yaml` is a switchboard
[`config.yaml`](config.yaml) at the repo root does **not** contain settings itself — it is a set of **pointers** to the config file currently in use for each module/tool:

```yaml
toolkit:
  split:      magneton/toolkit/configs/config_split_fib_f_neuron.yaml
  downsample: magneton/toolkit/configs/config_downsample_fib_f_synapse_instances_v3.yaml
  crop:       magneton/marmoset_project/bouton/configs/crop_f.yaml
  # … one entry per toolkit tool …
affinity_prediction:
  config_base: .../pytorch_connectomics/configs/Isotropic-Synapse-v3-fib_f.yaml
  config_file: .../pytorch_connectomics/configs/blank.yaml
  checkpoint:  .../checkpoint_125000.pth.tar
  hpc:         magneton/pytorch_connectomics/configs/hpc_f_synapse.yaml
instance_segmentation:
  main: .../instance_segmentation/configs/config_fib_f_synapse_v3.yaml
analysis:
  main: magneton/analysis/configs/config_fib_b_relational.yaml
proofreading:
  main: magneton/proofreading/configs/config_fib_b_membrane.yaml
```

When you run a stage, the CLI reads the relevant pointer, loads that YAML, and runs with its settings.

### Where the actual configs live
Each module keeps its config files in its own `configs/` folder:

| Module | Config folder | Notes |
|--------|---------------|-------|
| Toolkit | `toolkit/configs/` | one file per tool |
| Affinity inference | `pytorch_connectomics/configs/` | a model config (`config_base`) + an `hpc_*` file; `blank.yaml` is the no-op override placeholder |
| Instance segmentation | `instance_segmentation/configs/` | **one file holds both** `segmentation_stage` and `merge_stage` |
| Analysis | `analysis/configs/` | per-volume, plus a relational (cross-volume) shape |
| Proofreading | `proofreading/configs/` | one file holds `membrane_stage`, `skeletonize_stage`, `expand_stage` |

The per-volume config files are **gitignored** — they are customized per dataset/run and not committed. Treat them as your working scratch.

### Changing what runs
Two ways to point a tool at different data/parameters:
1. **Edit the active config** that the pointer references, or
2. **Re-point** the root `config.yaml` entry to a different config file.

Each config also carries an `hpc:` block (partition, memory, time, etc.) used when you run the HPC variant of a stage.

> **YAML gotcha:** quote time strings (`time: "10:00:00"`). Unquoted, YAML 1.1 reads `10:00:00` as the integer `36000`.

The `/config` Claude skill (below) automates this whole loop — it interviews you, writes a new config from a reference template, and patches the root pointer for you.

## Usage

### Unified menu
```bash
python -m magneton
```
Pick a module (1–5) → pick a stage → choose **local** (runs now, on the current machine) or **HPC** (submits a SLURM job). Heavy work (inference, segmentation, merging, large analysis) should go to HPC; small tests and quick tools can run locally.

### Per-module entry points
Each module is also scriptable directly with a config (handy for scripts and reproducibility):

```bash
# Toolkit (split / merge / convert / downsample / crop / mesh / mask)
python toolkit/main.py --config config.yaml

# Affinity inference (train / fine-tune / infer)
python pytorch_connectomics/main.py --config config.yaml

# Instance segmentation (segmentation + merge stages)
python instance_segmentation/main.py --config instance_segmentation/configs/config.yaml

# Analysis (per-stage: discover / instance / reduce / embed / relational)
python -m magneton.analysis.main --stage discover --config <analysis config>

# Proofreading (per-stage: membrane / skeletonize)
python -m magneton.proofreading.main --stage membrane --config <proofreading config>
```

### Where outputs and job logs go
- Derived precomputed volumes are written to shared storage (paths set in each config's `paths:`).
- SLURM jobs land in `jobs/<stage>_<suffix>/` with `submit_slurm.sh`, a `manifest.txt`, and per-task logs under `logs/`.
- Long-running stages checkpoint, so re-running resumes rather than restarting.

## The processing toolkit

The toolkit (`python toolkit/main.py`, or module 1 in the menu) bundles the data-wrangling steps around the ML stages. Every tool has a **local** and an **HPC** variant, and each is driven by its pointer under `toolkit:` in the root config.

| Tool | Purpose |
|------|---------|
| **Split volume** | Split a large tif/precomputed volume into overlapping blocks (for block-wise inference). |
| **Merge blocks** | Stitch h5 inference blocks back into a single volume. |
| **Convert prec** | Convert tif/h5 → precomputed (Neuroglancer) format. |
| **Downsample prec** | Build lower-resolution mip levels with igneous. |
| **Generate mask** | Make a binary mask from an affinity map. |
| **Mask prec / Mask tif** | Apply a mask to a precomputed / tif volume. |
| **Resize tif** | Resample a tif to a new voxel size or shape. |
| **Crop volume** | Extract an ROI from a tif/h5/precomputed volume. |
| **Mesh prec** | Generate 3D meshes from a segmentation volume (for Neuroglancer). |

## Optional: Claude Code assistant tools

This repo ships a set of [Claude Code](https://claude.com/claude-code) **skills** under [`.claude/skills/`](.claude/skills/) that wrap common operational chores in natural language. They are **optional conveniences** — the pipeline runs completely without them — but they speed up day-to-day work if you use Claude Code in this repo. Invoke a skill by typing `/<name> <plain-English request>`.

| Skill | What it does |
|-------|--------------|
| `/config` | Generate a new pipeline config from a description: picks a reference template, interviews for missing fields, writes the file, and patches the root `config.yaml` pointer. |
| `/check` | Report on a SLURM job referenced in conversation — state, errors, memory, wall-clock, array concurrency, and whether output was produced. |
| `/hpc` | Inspect Yale YCRC cluster state and estimate job wait time (`/hpc info`, `/hpc time`). |
| `/globus` | Build and run a Globus transfer of a precomputed dataset between endpoints (defaults: Yale Misha → KuanLab storage). |
| `/wk` | Upload an EM volume (+ optional segmentation / skeleton) to webknossos.org via `upload_webknossos.py`. |
| `/report` | Write end-of-session docs: a code-change log (`claude_notes/`, committed) and/or a lab-notebook entry (synced to Notion). |

## Repository layout

```
magneton/
├── config.yaml              # root switchboard: pointers to the active config per module/tool
├── toolkit/                 # data-processing tools (+ configs/)
├── pytorch_connectomics/    # affinity prediction: training & inference (+ configs/)
├── instance_segmentation/   # waterz / binary_watershed + merge (+ configs/)
├── analysis/                # per-instance morphometrics (+ configs/)
├── proofreading/            # membrane- and skeleton-driven WebKnossos proofreading (+ configs/)
├── waterz/                  # waterz agglomeration backend
├── jobs/                    # SLURM submit scripts + per-job logs
├── docs/                    # Sphinx documentation (readthedocs)
└── .claude/skills/          # optional Claude Code assistant skills
```
