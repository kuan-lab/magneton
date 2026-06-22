# Magneton

**Magneton** is a connectomics segmentation and analysis pipeline developed by [Kuan Lab](https://www.kuanlab.org/) for neuron and organelle (mitochondria, bouton, synapse) processing of large-scale 3D EM data. It uses a chunk-based mode for processing large volumes and is available as a CLI tool, supporting both local and HPC (SLURM) jobs.

> Based on Linux machines with NVIDIA GPUs.

## Documentation

Full docs: [magneton.readthedocs.io](https://magneton.readthedocs.io/en/latest/) (source in [`docs/source/`](docs/source/)). This README mirrors the essentials.

## Modules

Magneton consists of 5 modules, each runnable from the unified CLI (`python -m magneton`):

### 1. Processing toolkit
Data pre- and post-processing:
- Split a large tif volume into blocks / merge inference blocks back into a volume
- Convert tif/h5 to precomputed (Neuroglancer) format
- Downsample a precomputed volume (mip pyramid, via igneous)
- Generate and apply masks for affinity maps (tif or precomputed)
- Resize and crop volumes
- Generate 3D meshes from a segmentation volume

### 2. Affinity-map inference
Deep-learning affinity prediction, based on [PyTorch Connectomics](https://connectomics.readthedocs.io/en/latest/index.html):
- Pre-train / fine-tune a model
- Infer affinity maps block-wise over large volumes

### 3. Instance segmentation
Block-wise instance segmentation of affinity maps, then aggregation into global IDs. Supports global and large-volume chunked-parallel processing, 2D and 3D supervoxels, in several modes (set by `mode.type`):
- **neuron** — waterz agglomeration (dense; fills the volume)
- **mito / synapse** — binary watershed (sparse objects; respects background)
- **bouton** — binary watershed with neuron-membrane gating

### 4. Analysis (per-instance morphometrics)
Organelle-agnostic morphometrics for any sparse instance volume (mito, bouton, synapse):
- Discover each instance's bounding box from a high-mip volume
- Compute per-instance features (volume, surface area, sphericity, hull, max diameter, per-PC axes) → `morphometrics.parquet`
- Embed (PCA + UMAP) and run cross-volume relational analysis

### 5. Proofreading / ground-truth bootstrapping
Human-in-the-loop correction in [WebKnossos](https://webknossos.org/):
- **Membrane** — crop EM + affinity from precomputed, threshold the affinity to a binary membrane, and upload EM + membrane to WebKnossos in one command for correction in the volume tool
- **Skeletonize** — skeletonize an instance segmentation into a WebKnossos NML (via kimimaro) for skeleton-level correction

## Pipeline flow

```
EM (precomputed)
  └─ Affinity inference (PyTC)        →  fib_X_inference_<model>_v<N>/
       └─ Convert / downsample (toolkit)
       └─ Instance segmentation        →  per-block outputs
            └─ Merge (pools + apply)    →  fib_X_<model>_instances_v<N>/
                 └─ Downsample / mesh
                 └─ Analysis            →  morphometrics.parquet
                 └─ Proofreading        →  WebKnossos correction (membrane / skeleton)
```

Data is stored as Neuroglancer **precomputed** (CloudVolume); resolution metadata lives in `info` JSON files. Heavy computation runs on SLURM — configs carry an `hpc:` section.

## Installation

See [`docs/source/installation/`](docs/source/installation/) for full details. In brief:

```bash
conda create -y -n magneton python=3.9
conda activate magneton
# PyTorch with the right CUDA (H100/A100 → 12.4):
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

git clone https://github.com/kuan-lab/magneton.git
cd magneton
pip install --editable .

# Affinity-inference module:
cd pytorch_connectomics && pip install --editable . && cd ..
# Waterz (instance segmentation):
cd waterz && pip install --editable . && cd ..
```

See the installation docs for common build-error fixes (boost, Cython, numpy, mahotas, libstdc++).

## Usage

Launch the unified CLI and pick a module (1–5):

```bash
python -m magneton
```

Each module is also scriptable via its own entry point with a YAML config, e.g.:

```bash
python toolkit/main.py --config config.yaml                                   # toolkit
python pytorch_connectomics/main.py --config config.yaml                      # affinity inference
python instance_segmentation/main.py --config instance_segmentation/configs/config.yaml
python -m magneton.analysis.main --stage discover --config <analysis cfg>     # analysis
python -m magneton.proofreading.main --stage membrane --config <pf cfg>       # proofreading
```

Stage configs live under each module's `configs/`; the root `config.yaml` points to the active config per module.

## Repository layout

```
magneton/
├── toolkit/                 # data processing tools (split, merge, convert, downsample, crop, mesh, mask)
├── pytorch_connectomics/    # affinity prediction (PyTC training/inference)
├── instance_segmentation/   # segmentation (waterz / binary_watershed) + merge
├── analysis/                # per-instance morphometrics
├── proofreading/            # skeleton- and membrane-driven WebKnossos proofreading
├── docs/                    # Sphinx documentation (readthedocs)
└── config.yaml              # master config pointing to per-module sub-configs
```
