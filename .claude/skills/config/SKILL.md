---
name: config
description: Generate a customized magneton pipeline config from a natural-language description — picks a reference template, interviews for missing fields, presents a draft for review, then writes the new config and patches the root `config.yaml` pointer. Invoked as `/config <english description of what to configure>`.
---

# /config

Generates a new stage config for the magneton pipeline. Parses the stage and any given fields from the English argument, picks a reference template, interviews for anything missing, and shows a draft for review before writing.

## Supported stages

This is the **authoritative static registry**. If a new pipeline function/stage/field is added to magneton that isn't listed here, this skill is out of date — ask the user whether to update it.

### Toolkit stages

| Stage keyword(s) | Template dir | Reference templates | Pointer in root config.yaml |
|---|---|---|---|
| `split` / `split volume` | `toolkit/configs/` | `config_split.yaml`, `configs_with_note/config_split.yaml` | `toolkit.split` |
| `merge volume` / `merge tif` | `toolkit/configs/` | `config_merge.yaml`, `config_merge_c.yaml`, `config_merge_f.yaml` | `toolkit.merge` |
| `prec` / `convert` / `to precomputed` / `h5 to precomputed` | `toolkit/configs/` | `config_prec.yaml`, `config_prec_c.yaml`, `config_prec_f.yaml` | `toolkit.prec` |
| `downsample` / `mip` / `igneous` | `toolkit/configs/` | `config_downsample.yaml`, `config_downsample_c.yaml`, `config_downsample_f.yaml` | `toolkit.downsample` |
| `gen_mask` / `generate mask` | `toolkit/configs/` | `config_gen_mask.yaml` | `toolkit.gen_mask` |
| `mask_prec` / `apply mask precomputed` | `toolkit/configs/` | `config_mask.yaml` | `toolkit.mask_prec` |
| `mask_tif` / `apply mask tif` | `toolkit/configs/` | `config_mask_tif.yaml` | `toolkit.mask_tif` |
| `resize_tif` / `resize` | `toolkit/configs/` | `config_resize_tif.yaml` | `toolkit.resize_tif` |
| `crop` / `crop volume` | `toolkit/configs/` | `config_crop.yaml`, `marmoset_project/bouton/configs/crop_{b,c,f}.yaml` | `toolkit.crop` |
| `mesh` / `meshing` / `mesh prec` | `toolkit/configs/` | `config_mesh.yaml`, `config_mesh_test1.yaml` | `toolkit.mesh` |

### PyTorch Connectomics (affinity prediction)

| Stage keyword(s) | Template dir | Reference templates | Pointer in root config.yaml |
|---|---|---|---|
| `pytc training` / `train affinity` / `training` / `synapse training` / `mito training` / `bouton training` | `pytorch_connectomics/configs/` | Neuron (3-channel affinity, 128³): `Isotropic-Neuron-Base.yaml` + `Isotropic-Neuron-Affinity-UNet.yaml` override. Binary 1-channel (mito/synapse/bouton/etc.): `Isotropic-Synapse-Base.yaml` (synapse, 160³), `Isotropic-Bouton-Base.yaml` (bouton, 192³), `Lucchi-Mitochondria.yaml` (mito reference, 112³). Single-file pattern: full yaml at `affinity_prediction.config_base`, `blank.yaml` at `affinity_prediction.config_file` as the no-op override (order matters — see Gotchas). | `affinity_prediction.config_base` / `affinity_prediction.config_file` |
| `pytc inference` / `run inference` / `affinity inference` / `pytc training hpc` / `pre-train-hpc` | `pytorch_connectomics/configs/` | Inference (short, batched): `hpc.yaml`, `hpc_b.yaml`, `hpc_c.yaml`, `hpc_f.yaml`, `hpc_f_bouton.yaml`. Training (long, single-job, one per model): `hpc_train_synapse.yaml`, `hpc_train_bouton.yaml` — keep separate yamls per concurrent training job so one job doesn't re-read a file the other is using. | `affinity_prediction.hpc` (and `affinity_prediction.checkpoint` for the model weights) |

### Instance segmentation

| Stage keyword(s) | Template dir | Reference templates | Pointer in root config.yaml |
|---|---|---|---|
| `segmentation` / `instance segmentation` / `waterz` / `mito seg` / `bouton seg` | `instance_segmentation/configs/` | `config.yaml`, `config_b.yaml`, `config_c.yaml`, `config_f.yaml`, `config_30tb.yaml`; bouton mode: `config_fib_b_bouton_v1.yaml` (full vol), `config_fib_b_bouton_roi_test.yaml` (single-block ROI) | `instance_segmentation.main` |

Note: one instance_segmentation config file holds **both** the segmentation stage and the merge stage (two top-level sections: `segmentation_stage` and `merge_stage`). They are not separate files.

### Analysis (per-instance morphometrics)

| Stage keyword(s) | Template dir | Reference templates | Pointer in root config.yaml |
|---|---|---|---|
| `analysis` / `morphometrics` / `mito features` / `bouton features` / `discover` | `analysis/configs/` | per-volume: `config_fib_b_mito.yaml`, `config_fib_b_mito_lucchi_v1.yaml`, `config_fib_b_bouton_v1.yaml`, `config_fib_b_synapse_v3.yaml`; relational (cross-volume, different shape): `config_fib_b_relational.yaml` | `analysis.main` |

Note: organelle-agnostic (mito, bouton, synapse, …) — the same per-instance bbox-driven pipeline. Relational configs have a different shape (a `volumes:` list), used for cross-volume matching.

### Proofreading (skeleton-driven GT correction)

| Stage keyword(s) | Template dir | Reference templates | Pointer in root config.yaml |
|---|---|---|---|
| `proofreading` / `skeletonize` / `expand` / `nninteractive` | `proofreading/configs/` | `config_fib_b_neuron_fennel.yaml` | `proofreading.main` |

Note: one file holds both the `skeletonize_stage` (runs in the `magneton` env) and `expand_stage` (runs in the `nninteractive` prefix env on GPU). The validated use is skeletonize→WebKnossos-correct; nnInteractive EM expansion is shelved (floods across membranes — see `project_skeleton_nninteractive_workflow` memory).

## Stage-specific fields to ask about

When generating a config, walk the fields below for the chosen stage. Fields already given in the user's English command should be filled in automatically; missing ones must be asked about.

### prec (convert to precomputed)
- `paths.input` — source tif/h5 path
- `paths.output` — target precomputed dir
- `input_format` — `tif` or `h5`
- `h5.datasets` — dataset key list (only if `h5`)
- `roi` — `[x, y, w, h]` or `null`
- `prec_info.num_channels`
- `prec_info.layer_type` — `image` or `segmentation`
- `prec_info.data_type` — `uint8`/`uint16`/`uint32`/`uint64`
- `prec_info.encoding` — `raw` / `jpeg` / `compressed_segmentation`
- `prec_info.resolution` — `[x, y, z]` in nanometers
- `prec_info.voxel_offset` — `[x, y, z]` voxels
- `prec_info.chunk_size` — `[x, y, z]` voxels
- `prec_info.compress` — bool
- `prec_info.lazy` — bool
- `hpc.*` — standard HPC block (see below)

### split
- `split.input`, `split.output`
- `split.mip` (if precomputed input)
- `split.chunk_size` — `[z, y, x]`
- `split.overlap` — `[z, y, x]`
- `hpc.*`

### merge (volume)
- `merge.input`, `merge.output`
- `merge.chunk_size`, `merge.overlap`
- `hpc.*`

### downsample
- `downsample.source_path`
- `downsample.queuepath` — igneous task state dir (usually `magneton/igneous_tasks`)
- `downsample.num_workers`
- `downsample.mip` — start mip
- `downsample.num_mips` — number of mip levels to generate
- `downsample.factor` — `[x, y, z]`
- `hpc.*`

### gen_mask
- `mask.input`, `mask.output`
- `mask.input_mip`
- `mask.preview_tif_flag`, `mask.preview_tif`
- `mask.min_region_size`, `mask.max_region_size`
- `mask.erode_size`, `mask.dilate_size`
- `hpc.*`

### mask_prec
- `mask.raw_path`, `mask.mask_path`, `mask.output_path`
- `mask.mip`
- `hpc.*`

### mask_tif
- `mask.raw_path`, `mask.mask_path`, `mask.output_path`
- `mask.mask_reverse` — bool
- `hpc.*`

### resize_tif
- `resize.input`, `resize.output`
- `resize.zoom_factor` — 3-axis tuple, <1 to shrink, >1 to grow
- `resize.zoom_order` — spline order (0 = nearest)
- `hpc.*`

### crop
- `crop.input`, `crop.output`
- `crop.coords` — `[x1, x2, y1, y2, z1, z2]`
- `crop.h5_key` (only if h5 input)
- `crop.resolution` — `[x, y, z]` nm
- `hpc.*`

### mesh
- `mesh.source_path` — path to precomputed segmentation volume (uint32 segment IDs)
- `mesh.queuepath` — igneous task state dir (usually `magneton/igneous_tasks`)
- `mesh.num_workers` — parallel workers (match to HPC `cpus`)
- `mesh.mip` — mip level to mesh from (0 = full resolution)
- `mesh.shape` — `[x, y, z]` spatial block size for meshing tasks (default `[448, 448, 448]`)
- `mesh.simplification` — bool, run mesh simplification after marching cubes
- `mesh.max_simplification_error` — max error in nm during simplification (higher = smaller files)
- `mesh.dust_threshold` — min voxel count for a segment to get a mesh (`null` = mesh everything)
- `hpc.*`

### pytc training (Isotropic-Neuron-Base.yaml style)
- `SYSTEM.NUM_GPUS`, `SYSTEM.NUM_CPUS`
- `DATASET.IMAGE_NAME` — path to a `.txt` file listing training image paths
- `DATASET.LABEL_NAME` — path to label list
- `DATASET.INPUT_PATH` — root dir for the lists
- `DATASET.OUTPUT_PATH` — checkpoint dir
- `DATASET.IS_ISOTROPIC` — bool (critical; see the FIB-SEM isotropic bug memory)
- `DATASET.PAD_SIZE` — `[z, y, x]`
- `MODEL.ARCHITECTURE`, `MODEL.BLOCK_TYPE`
- `MODEL.INPUT_SIZE`, `MODEL.OUTPUT_SIZE` — `[z, y, x]`
- `MODEL.IN_PLANES`, `MODEL.OUT_PLANES`
- `MODEL.LOSS_OPTION`, `MODEL.LOSS_KWARGS_*`, `MODEL.LOSS_WEIGHT`, `MODEL.WEIGHT_OPT`, `MODEL.OUTPUT_ACT`, `MODEL.TARGET_OPT`
- `MODEL.FILTERS`, `MODEL.NORM_MODE`
- `AUGMENTOR.*` — individual toggle blocks (ROTATE, FLIP, RESCALE, ELASTIC, GRAYSCALE, MISSINGPARTS, MISSINGSECTION, MISALIGNMENT, MOTIONBLUR, CUTBLUR, CUTNOISE)
- `SOLVER.LR_SCHEDULER_NAME`, `SOLVER.BASE_LR`, `SOLVER.ITERATION_*`, `SOLVER.SAMPLES_PER_BATCH`
- `MONITOR.ITERATION_NUM`
- `INFERENCE.*` — inference sub-block (INPUT_SIZE, OUTPUT_SIZE, IMAGE_NAME, OUTPUT_PATH, OUTPUT_NAME, PAD_SIZE, AUG_MODE, AUG_NUM, STRIDE, SAMPLES_PER_BATCH)

The override file (e.g. `Isotropic-Neuron-Affinity-UNet.yaml`) only contains a handful of keys that overlay the base — most often `MODEL.ARCHITECTURE`, `MODEL.BLOCK_TYPE`, `DATASET.OUTPUT_PATH`, `INFERENCE.OUTPUT_PATH`.

### pytc inference (hpc_*.yaml)
- `hpc.time`, `hpc.cpus`, `hpc.mem-per-gpu` (or `hpc.mem`)
- `hpc.gpus` — e.g. `h100:1`, `a100:1`, or bare `1` when pairing with `hpc.constraint`
- `hpc.constraint` — SLURM node-feature expression, e.g. `"h100|h200"`. Pipe = OR. Paired with `gpus: 1` to widen the eligible backfill pool beyond a single GPU type. Omit for no constraint.
- `hpc.partition`
- `hpc.env`, `hpc.conda`, `hpc.extra_modules`, `hpc.work_path`
- `hpc.mutil_jobs` — bool; batch inference over a split input volume
- `hpc.mutil_jobs_configs.configs_save_path`, `input_folder`, `batch_num` (only if `mutil_jobs: true`)
- `hpc.mutil_jobs_configs.chunks_per_task` — int, default 1. Groups N chunks into one SLURM array task so CUDA warmup (~19 s) + Python/model-load (~13 s) are paid once per N chunks instead of per chunk. Implemented in `run_hpc.py:_gen_chunk_configs` and the multi-chunk loop in `run.py`. Values of 5 work well on 512³ FIB-SEM (70 chunks → 14 tasks). Set to 1 to preserve legacy single-chunk-per-task behavior.

### instance_segmentation
- `paths.input` — affinity precomputed
- `paths.output` — global instance output
- `paths.output_local_base` — per-block output base path
- `mode.type` — `neuron` (waterz), `mito` (binary_watershed), or `bouton` (binary_watershed + neuron membrane gating)
- `mode.mito.*` (only if mito): `seed_threshold`, `foreground_threshold`, `min_segment_size`, `seed_min_size`, `remove_small_mode`, `erosion_iters`
- `mode.bouton.*` (only if bouton): same watershed knobs as mito **plus** membrane gating. **Required** `neuron_ref_path` — a neuron affinity precomputed; voxels where neuron affinity is low (membrane/ECS) get the bouton affinity zeroed *before* watershed, breaking cross-membrane merges. A bouton↔neuron resolution mismatch is auto-handled (reads neuron at its own mip, upsamples; e.g. 8nm neuron vs 4nm bouton). Other knobs: `neuron_aff_threshold` (below this = membrane; **lower = less masking**, fixes over-aggressive splits), `neuron_aff_reduce` (`mean`/`min`/`first`), `dilation_iters` (membrane-constrained rounding of final instances). Boutons are **much larger than mito**, so `min_segment_size`/`seed_min_size` must be far larger than mito values. fib_b converged reference (`config_fib_b_bouton_v1.yaml`): seed 0.745, foreground 0.353, erosion_iters 0, dilation_iters 2, neuron_aff_threshold 0.3, seed_min_size 3000, min_segment_size 15000.
- `mask.flag`, `mask.path`
- `checkpoint.segmentation_dir`, `checkpoint.merge_dir`
- `block.size` — `[z, y, x]` (isotropic vs anisotropic — see FIB-SEM isotropic bug memory)
- `block.overlap` — `[z, y, x]`
- `block.roi` — `null` for full volume, or `[z1, z2, y1, y2, x1, x2]` absolute voxel coords to segment a sub-region (output stays in the full coordinate frame, overlays on full EM). Auto-snapped to chunk (128) boundaries — required so the merge core-trimming doesn't mis-slice (non-chunk-aligned starts → negative local slice → `AlignmentError`). For a multi-block ROI, `block.size` must also be a multiple of the chunk size; for a single-block ROI set `block.size` = ROI extent.
- `segmentation_stage.parallel`, `workers`, `metadata_dir` (block-JSON dir — put it under `magneton/metadata/seg_metadata_<volume>`, NOT top-level; default if unset is `./metadata/local_metadata`), `mip`
- `segmentation_stage.thresholds` — list of waterz thresholds
- `segmentation_stage.aff_thresholds`
- `segmentation_stage.supervoxel` — `3d` (isotropic) or `2d` (serial section)
- `segmentation_stage.interior_threshold`, `min_distance` (3D supervoxel only)
- `segmentation_stage.method`, `merge_function` (2D supervoxel only)
- `segmentation_stage.hpc.*` + `blocks_per_job`, `workers_per_job`, `hpc_num`
- `merge_stage.metadata_dir` (same `magneton/metadata/seg_metadata_<volume>` dir as segmentation_stage), `workers`, `mip`
- `merge_stage.min_overlap_vox`, `min_frac_local`, `min_frac_global`, `max_voxel_size`
- `merge_stage.require_recip`, `allow_union_amb`, `dom_ratio`, `min_iou`
- `merge_stage.export_tif.enable`, `path`, `max_slices`
- `merge_stage.hpc.*` + `blocks_per_job`, `workers_per_job`

### analysis (per-instance morphometrics)
- `paths.input` — high-mip instance precomputed volume (`file://...`)
- `paths.mip` — discovery mip (mito → `2`/16nm; bouton/synapse → `1`/8nm, smaller objects need finer mip to avoid downsample dropout)
- `paths.output` — work dir (`bboxes.parquet`, `morphometrics.parquet`)
- `bbox_halo_mipN` — pad ± voxels at the discovery mip (compensates downsample rounding)
- `min_voxel_count_mipN`
- `features.sa_method` — `face_count` / `marching_cubes` (default) / `sqrt_kernel` (paper-faithful)
- `discover_stage.hpc.*` — big-mem single job (needed for mip-1/0 where read_full+find_objects won't fit on login; `mem` is TOTAL, e.g. `64G` at mip1, `240G` at mip0)
- `instance_stage.hpc.*` — SLURM array (per-instance feature math)
- **Relational config** (different shape): top-level `volumes:` list each with `pc` (precomputed) + `name`; `sample_mip`. Use `config_fib_b_relational.yaml` as the template.

### proofreading (skeleton-driven GT correction)
- `paths.seg` — instance seg to skeletonize (tif or `file://` precomputed)
- `paths.em` — EM tif (the nnInteractive image, for expand)
- `paths.output` — work dir (`skeletons.nml`, `expanded.tif`) — **put on shared storage, NOT home** (home quota is full); e.g. `/gpfs/radev/.marilyn/pi/kuan/shared/marmoset_project/nninteractive_output/<volume>`
- `skeletonize_stage`: `source` (`tif`/`precomputed`), `mip`, `res_nm`, `dust_threshold`, `parallel`, `parallel_chunk_size`, `fix_branching`, `fix_borders`; `teasar.*` (`scale`, `const`, `pdrf_exponent`, soma thresholds — the kimimaro tuning surface); `postprocess.{enable,tick_threshold,dust_threshold}`; `downsample_nodes`
- `expand_stage`: `nml` (the CORRECTED NML from WebKnossos; `null` → uses `<output>/skeletons.nml`), `prompt` (`scribble`/`points`), `point_subsample`, `max_neurons` (0=all), `model_dir` (`null`→resolve from HF cache), `hf_cache`; `hpc.*` is a **GPU** block (`partition: gpu`, `constraint: "a100|h100|h200"`, `gpus`, `mem_per_gpu`, `env` = the `nninteractive` prefix env path, `hf_cache`)

### Standard HPC block (all stages)
- `hpc.enable`
- `hpc.scheduler` — usually `slurm`
- `hpc.job_dir`
- `hpc.python_bin`
- `hpc.time` — **always quote** `"HH:MM:SS"` (YAML 1.1 sexagesimal bug)
- `hpc.cpus`, `hpc.mem`
- `hpc.partition` — `day` / `week` / `gpu` / `bigmem`
- `hpc.extra_modules`
- `hpc.conda` — `/gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh`
- `hpc.env` — `magneton` for pipeline work
- `hpc.work_path` — usually `.`

## Invocation flow

`/config <english description of what to configure>`

### Step 1 — parse the argument

From the English, extract:
- **Stage name** (match against keywords table above). If ambiguous or missing → ask.
- **Multi-stage intent** — if the English mentions two or more stages, or words like "both", "and segmentation", etc. Otherwise default to single-stage but still confirm.
- **Data paths, resolution, chunk sizes, thresholds, etc.** — anything explicitly mentioned that maps to a field in the stage's field list.
- **Reference config**, if explicitly named (e.g. "like config_f", "based on hpc_b").
- **New filename**, if explicitly given.

### Step 2 — confirm stage and multi-stage

If any of the following is ambiguous or missing, ask:
- Which stage? (show candidates)
- Single or multi-stage? (default: single; if multi, list which stages)

For multi-stage: walk each stage sequentially — finish one (through review + write) before starting the next. Do **not** batch them.

### Step 3 — pick the reference template

- If the user named a template explicitly, use it.
- Otherwise, present the **top 2-3 candidate templates** from the stage's reference list with a one-line summary of each (dataset it was made for, notable differences). Ask the user to pick.
- Read the chosen template with the Read tool.

### Step 4 — propose a new filename

- Following the `_<letter>` suffix pattern (`config_b`, `config_c`, `config_f`, `hpc_f_bouton`), suggest a filename based on the dataset name from the English or from the selected template.
- Show the suggestion and ask for confirmation or a different name.

### Step 5 — interview for missing fields

- Walk the stage's field list (above). For each field, check:
  - Was it explicitly provided in the English?
  - Is the template's current value a reasonable default for the new dataset?
- Group the still-unknown fields by section (paths / resolution / chunk size / HPC / stage-specific) and ask in one batch per section, not field-by-field.
- Use the template's value as a default suggestion where possible.

### Step 6 — present draft for review

- Show the full proposed YAML as a fenced code block.
- Highlight fields that differ from the chosen template (e.g., "changed from template: paths.input, block.size, hpc.time").
- Ask for approval. If the user asks for edits, apply them and re-present the full draft.

### Step 7 — write the new file + patch root config.yaml

On approval:
1. Write the new config file to the correct directory (same directory as the chosen template).
2. **Patch only the relevant pointer** in `/gpfs/radev/home/yf354/magneton/config.yaml`. Use Edit to change only the specific key from the pointer column of the stage table. Do not touch any other keys in `config.yaml`.
3. Report both paths to the user.

For multi-stage: repeat steps 3–7 for each additional stage.

## Gotchas

- **YAML 1.1 sexagesimal bug**: time fields must be quoted strings like `"10:00:00"`, otherwise YAML parses them as integer seconds (36000). See `pytorch_connectomics/main.py` `_fix_yaml_time()` which silently repairs this, but prevention is better.
- **Isotropic vs anisotropic**: the FIB-SEM isotropic bug memory notes that both mito and neuron templates default to anisotropic block sizes and must be switched to isotropic `[512, 512, 512]` for isotropic data. Verify with the user when asking about `block.size`.
- **Single instance_segmentation file, two stages**: both `segmentation_stage` and `merge_stage` live in one YAML file and are both pointed to by `instance_segmentation.main`. There's no separate merge config file.
- **Instance seg HPC block appears twice**: each of `segmentation_stage.hpc` and `merge_stage.hpc` has its own full HPC block with different resource profiles (waterz is single-threaded so `cpus: 2`; merge is multi-threaded so `cpus: 32`).
- **Conda path**: the canonical path for this HPC is `/gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh`. Some older templates still reference `/gpfs/radev/home/zz545/miniconda3/etc/profile.d/conda.sh` — replace when generating new configs.
- **GPU partition time limit vs backfill**: SLURM's backfill scheduler on the `gpu` partition rejects pytc inference jobs whose `hpc.time` is larger than the free-slot windows it can find on partially-allocated nodes. Observed 2026-04-15: a `time: "04:00:00"` submission projected a 2h 43m wait, while `"00:30:00"` immediately backfilled onto partial h100 nodes. When generating a pytc inference hpc yaml, set `hpc.time` to the *realistic* work budget (e.g. `"00:30:00"` for a 14-task 512³ FIB-SEM inference with `chunks_per_task: 5`) — not the 4h training default. The backfiller treats the limit as a hard upper bound it must reserve.
- **Single-file pytc training + `blank.yaml`**: pytc's CLI accepts a single `--config-file` (`connectomics/config/utils.py` — `config_base` is optional), but magneton's wrapper (`pytorch_connectomics/tools/run_hpc.py:162`) always passes both `--config-base` and `--config-file`. User prefers one self-contained yaml + `pytorch_connectomics/configs/blank.yaml` (contents: `null`) as the override placeholder, rather than splitting into a Base + UNet-override pair. **Order**: the full yaml goes at `affinity_prediction.config_base` (loaded first), `blank.yaml` goes at `affinity_prediction.config_file` (no-op override). Matches the existing neuron convention where Base is full and UNet is the override. Only use the two-file pattern if the user explicitly wants to reuse a Base across multiple experiments.
- **Synapse vs mito input size**: synapse detection needs ~1.28 μm³ context (default 160³ at 8nm) because the cleft is only ~20 nm and detection depends on vesicle/PSD context. Mito (Lucchi 112³, ~0.9 μm) is enough because mitochondria are large opaque blobs where local boundary appearance suffices. Scale `PAD_SIZE` and inference `STRIDE` with the input cube: 160³ → `PAD_SIZE: [40, 40, 40]`, `STRIDE: [80, 80, 80]` (50% overlap).
- **hpc yaml is used for training too, not just inference**: magneton's `pytorch_connectomics/main.py:204-215` loads `affinity_prediction.hpc` for the `pre-train-hpc`, `fine-tune-hpc`, and `inference-hpc` stages. Training needs a DIFFERENT hpc yaml than inference: `time` must fit the full training run (e.g. `"24:00:00"` for 150k iterations at batch 2 on h100), `mutil_jobs: false` (training is a single long job, not batched over input chunks), no `mutil_jobs_configs` block. Don't reuse an inference hpc yaml (e.g. `hpc_f.yaml` with `time: "00:30:00"` and `mutil_jobs: true`) for training — it'll either SIGTERM mid-training or try to split the training job into array tasks.

## Keeping this skill current

This skill uses **static** stage→fields tables. It does **not** live-scan the configs directory. When a new pipeline function/stage/field is added to magneton, this skill will not know about it until it is manually updated.

**Claude's responsibility**: whenever a session adds a new pipeline function, new stage, or new config field, proactively ask the user: "Should I update the `/config` skill since we added `<thing>`?" (This is also recorded in the auto-memory under `feedback_skill_update_prompts.md`.)
