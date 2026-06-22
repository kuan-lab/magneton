# Analysis

This tutorial provides step-by-step guidance for the per-instance morphometrics pipeline.
> Organelle-agnostic: the same pipeline works for mitochondria, boutons, synapses, or any sparse instance-labeled volume.

The analysis module reads a high-mip instance-segmentation volume, finds each instance's bounding box, computes morphometric features per instance, and concatenates them into a single `morphometrics.parquet`. It can then embed the features (PCA + UMAP) and run cross-volume relational analysis.

#### Main Menu

Run ```python -m magneton``` to start the CLI.

Input ```4``` and ```Enter```, enter the **Analysis module**.

#### Options

This module can be used for configuration and for each stage of the morphometrics pipeline. The stages run in order: **Discover → Per-Instance Features → Reduce**, with HPC variants for the heavy steps.

##### Global Configuration
Input ```9``` and ```Enter```, view the current **analysis configuration** (resolved from the ```analysis.main``` config in the global configuration file).

Key fields:
- ```paths.input```: high-mip instance precomputed volume (```file://...```).
- ```paths.mip```: discovery mip (mito → ```2``` / 16 nm; bouton & synapse → ```1``` / 8 nm, since smaller objects need a finer mip to avoid downsample dropout).
- ```paths.output```: work dir for ```bboxes.parquet``` and ```morphometrics.parquet```.
- ```features.sa_method```: surface-area method — ```face_count```, ```marching_cubes``` (default), or paper-faithful ```sqrt_kernel```.

##### Discover Bboxes
Input ```1``` and ```Enter```, using ***local resources*** to read the high-mip volume and find each instance's bounding box (```find_objects```), writing a bbox manifest.

Input ```2``` and ```Enter```, using ***HPC resources*** to submit discovery as a big-memory SLURM job. Use the HPC variant for mip-1 / mip-0 discovery, where reading the full volume plus ```find_objects``` will not fit on a login node.

##### Per-Instance Features
Input ```3``` and ```Enter```, using ***local resources*** to compute per-instance features single-process (useful for debugging).

Input ```4``` and ```Enter```, using ***HPC resources*** to submit a SLURM array, one task per range of instances. Each instance is cropped from the volume and ~19 features are computed (volume, surface area, sphericity, convex hull, max diameter, and per-PC length / inertia / symmetry / cross-section area / cross-section perimeter for each of 3 principal axes).

##### Reduce / Concat
Input ```5``` and ```Enter```, to concatenate the per-task feature partials into a single ```morphometrics.parquet```.

##### All [HPC]
Input ```6``` and ```Enter```, to run the whole chain on the cluster end-to-end: Discover [HPC] → per-instance array → reduce (the reduce job is queued to run after the array completes).

##### Embed (PCA + UMAP)
Input ```7``` and ```Enter```, to z-score the features, run PCA + UMAP, and save the embedding and plots.

##### Relational (cross-volume)
Input ```8``` and ```Enter```, to match instances across volumes (e.g. mitochondria / synapses → boutons) and produce relational statistics and plots. This uses a relational config (a ```volumes:``` list) rather than a single-volume config.
