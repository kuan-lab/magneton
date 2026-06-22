# About 

**Magneton** is a connectomics segmentation and analysis pipeline developed by [Kuan Lab](https://www.kuanlab.org/) for neuron and organelle (mitochondria, bouton, synapse) processing of large-scale 3D EM data. It employs a chunk-based mode for processing large volumes and is available as a CLI tool, supporting both local jobs and HPC (SLURM) jobs.

> Based on Linux machines with NVIDIA GPUs

##### Magneton consists of 5 main modules:

1. Data pre- and post-processing toolkit
- Split a big tif volume into blocks / merge inference blocks back into a volume
- Convert a tif/h5 file to precomputed (Neuroglancer) format
- Downsample a precomputed volume (mip pyramid)
- Generate and apply a mask for affinity maps (tif or precomputed)
- Resize and crop volumes
- Generate 3D meshes from a segmentation volume

2. Deep learning based affinity maps inference
> Based on [pytorch connectomics](https://connectomics.readthedocs.io/en/latest/index.html). This is a deep learning framework for automatic and semi-automatic annotation of connectomics datasets, powered by [pytorch](https://pytorch.org/).
- Pre-train / fine-tune a DL model
- Inference affinity maps by blocks over large volumes

3. Instance segmentation for affinity maps
> Supports global segmentation and large-volume chunking parallel processing, 2D supervoxel and 3D supervoxel.
- Block-wise instance segmentation in one of several modes:
  - **neuron** — waterz agglomeration (dense; fills the volume)
  - **mito / synapse** — binary watershed (sparse objects; respects background)
  - **bouton** — binary watershed with neuron-membrane gating
- Aggregation of each blocked segmentation result into global IDs

4. Analysis (per-instance morphometrics)
> Organelle-agnostic: mitochondria, boutons, synapses, or any sparse instance volume.
- Discover each instance's bounding box from a high-mip volume
- Compute per-instance morphometric features (volume, surface area, sphericity, hull, PCA axes, …)
- Reduce to a single `morphometrics.parquet`, embed (PCA + UMAP), and run cross-volume relational analysis

5. Proofreading / ground-truth bootstrapping
> Human-in-the-loop correction in [WebKnossos](https://webknossos.org/).
- **Membrane** — crop EM + affinity from precomputed, threshold the affinity to a binary membrane, and upload both to WebKnossos for correction
- **Skeletonize** — skeletonize an instance segmentation into a WebKnossos NML for correction
