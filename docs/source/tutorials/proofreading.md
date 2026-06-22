# Proofreading

This tutorial provides step-by-step guidance for the proofreading / ground-truth bootstrapping module.
> Human-in-the-loop: correction happens in [WebKnossos](https://webknossos.org/), then the corrected result is fed back into the pipeline.

The idea is to let a human edit a **cheap representation** of a segmentation and regenerate the dense labels, giving a faster ground-truth / proofreading cycle. Two entries are provided:

- **Membrane** *(recommended)* — crop EM + affinity inference from precomputed, threshold the affinity to a binary membrane, and upload EM + membrane to WebKnossos in one command. The membrane is then corrected directly in the WebKnossos volume tool.
- **Skeletonize** — skeletonize an instance segmentation into a WebKnossos NML (with `kimimaro`) for skeleton-level correction.

> The skeleton → **nnInteractive expand** path (regenerating dense masks from corrected skeletons) is **not functional**: nnInteractive was trained on CT/MRI and light microscopy and floods across EM membranes. The skeletonize → WebKnossos-correct half is kept; for dense correction use the membrane entry.

#### Main Menu

Run ```python -m magneton``` to start the CLI.

Input ```5``` and ```Enter```, enter the **Proofreading module**.

#### Options

##### View Current Config
Input ```9``` and ```Enter```, view the current **proofreading configuration** (resolved from the ```proofreading.main``` config in the global configuration file). A single config file holds all stages as top-level sections (```membrane_stage```, ```skeletonize_stage```, ```expand_stage```).

##### Membrane → WebKnossos
Input ```4``` and ```Enter``` to run the membrane stage. It:

1. Crops the affinity **inference** and **EM** from precomputed at the configured ROI. The two are read at mips that resolve to the same resolution (inference at its 8 nm mip0; the 4 nm EM source at mip1 = 8 nm).
2. Converts the affinity to a binary membrane: ```membrane = reduce_over_channels(aff) < threshold``` (default ```min``` across channels, threshold ```140```). Membrane voxels become foreground, cytoplasm stays background.
3. Writes ```em.tif``` + ```membrane.tif``` to the output dir and uploads both to WebKnossos (EM as a color layer, membrane as the segmentation annotation layer) for correction.

Key config fields (under ```membrane_stage```):
- ```coords```: ROI as ```[z1, z2, y1, y2, x1, x2]``` in the common (post-mip, 8 nm) voxel space.
- ```inference.path``` / ```inference.mip``` / ```channel_reduce``` / ```threshold```.
- ```em.path``` / ```em.mip```.
- ```upload.enable``` (set ```false``` to only write the TIFs and upload later), ```upload.remote_folder``` (a WebKnossos folder path, id, or dashboard URL), ```upload.out_dir``` (local wkw scratch dir), ```upload.cleanup```.

> The crop + threshold run in the ```magneton``` env; the upload runs in a separate env that has webknossos-libs (set its name via ```upload.env``` in the config). A WebKnossos auth token is read from the ```WEBKNOSSOS_TOKEN``` environment variable. The whole step is light enough to run interactively — no HPC needed.

##### Skeletonize
Input ```1``` and ```Enter``` to skeletonize the instance segmentation (```paths.seg```) into ```skeletons.nml``` with ```kimimaro```. Upload it to WebKnossos for skeleton-level correction. The TEASAR parameters under ```skeletonize_stage.teasar``` are the main tuning surface.

##### Expand (nnInteractive) — not functional
Input ```2``` (local) / ```3``` (HPC) would expand a corrected NML into dense segments via nnInteractive. **This path does not work on EM** (see the note above) and is retained only for reference; use the membrane entry instead.
