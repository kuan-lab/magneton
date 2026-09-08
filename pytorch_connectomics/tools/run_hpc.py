# -*- coding: utf-8 -*-
import os
import math
import h5py
import yaml
import argparse
import numpy as np
import tifffile as tiff
from cloudvolume import CloudVolume

from magneton.pytorch_connectomics.utils.config import load_config, load_global_config_path
from magneton.pytorch_connectomics.inference_grid import (
    is_precomputed_url, volume_info_from_cv, block_count, apply_roi,
)
from magneton.pytorch_connectomics.init_prec_output import init_output_volume
import gc
import math as _math
import subprocess
from pathlib import Path


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _gen_chunk_configs(config_base, configs_save_path, input_folder, chunks_per_task=1):
    files = sorted(os.listdir(input_folder))
    num_batches = math.ceil(len(files) / chunks_per_task) if files else 0

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    for i in range(num_batches):
        batch = files[i * chunks_per_task : (i + 1) * chunks_per_task]
        image_names = list(batch)

        if chunks_per_task == 1:
            # Scalar values preserve the original single-chunk behavior and
            # the OUTPUT_NAME type (string) expected by YACS defaults.
            inference_fields = {
                "IMAGE_NAME": image_names[0],
                "OUTPUT_NAME": os.path.splitext(image_names[0])[0] + '.h5',
            }
        else:
            # Multi-chunk: only IMAGE_NAME is a list. YACS rejects list for
            # OUTPUT_NAME because its default is a string, so run.py derives
            # output names from image names at runtime.
            inference_fields = {
                "IMAGE_NAME": image_names,
            }

        new_params = {
            "INFERENCE": inference_fields,
            "DATASET": {
                "INPUT_PATH": f"{input_folder}",
            },
        }

        with open(config_base) as f:
            config_data = yaml.safe_load(f)
        update_dict(config_data, new_params)
        new_config_path = f"{configs_save_path}/temp_config_{i}.yaml"
        with open(new_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

    return num_batches




_AUTO_RESUME_TMPL = r'''CKPT="@@CKPT@@"
MAX_RESTARTS=@@MAXR@@
ITER_TOTAL=@@TOTAL@@
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
  if [ "${SLURM_RESTART_COUNT}" -gt "${MAX_RESTARTS}" ]; then
    echo "[auto-resume] restart #${SLURM_RESTART_COUNT} exceeds max ${MAX_RESTARTS}; not resuming again." >&2
    exit 1
  fi
  OUTDIR=$(dirname "$CKPT")
  # Sort by iteration NUMBER, not lexically: a plain ls sorts checkpoint_95000
  # after checkpoint_195000, which would silently rewind the run.
  CKPT_LIST=$(ls -1 "$OUTDIR"/checkpoint_*.pth.tar 2>/dev/null \
              | sed 's|.*/checkpoint_0*\([0-9][0-9]*\)\.pth\.tar$|\1 &|' \
              | sort -n -k1,1 | cut -d' ' -f2-)
  N=$(printf '%s\n' "$CKPT_LIST" | grep -c .)
  if [ "$N" -gt 0 ]; then
    NEWEST=$(printf '%s\n' "$CKPT_LIST" | tail -1)
    # Preemption can kill the job mid-torch.save. Checkpoints for one arm vary
    # by <1KB, so a >1% shortfall vs the previous one means a partial write.
    if [ "$N" -gt 1 ]; then
      PREV=$(printf '%s\n' "$CKPT_LIST" | tail -2 | head -1)
      SZ_NEW=$(stat -c%s "$NEWEST" 2>/dev/null || echo 0)
      SZ_PREV=$(stat -c%s "$PREV" 2>/dev/null || echo 0)
      if [ "$SZ_NEW" -lt $(( SZ_PREV * 99 / 100 )) ]; then
        echo "[auto-resume] $NEWEST looks truncated ($SZ_NEW vs $SZ_PREV); using $PREV" >&2
        NEWEST="$PREV"
      fi
    fi
    CKPT="$NEWEST"
  fi
  # 10# forces base 10: bash's test treats a leading zero as octal, so
  # checkpoint_05000 would otherwise compare as 2560.
  ITER=$(basename "$CKPT" | sed 's|checkpoint_\([0-9][0-9]*\)\.pth\.tar|\1|')
  ITER=$((10#$ITER))
  if [ "$ITER_TOTAL" -gt 0 ] && [ "$ITER" -ge "$ITER_TOTAL" ]; then
    echo "[auto-resume] checkpoint at $ITER >= ITERATION_TOTAL $ITER_TOTAL; already complete."
    exit 0
  fi
  echo "[auto-resume] restart #${SLURM_RESTART_COUNT}: resuming from $CKPT (iter $ITER)"
fi
'''

def _auto_resume_bash(checkpoint_path, iter_total, max_restarts):
    """Bash preamble choosing the checkpoint for a fine-tune-hpc job.

    A first run (SLURM_RESTART_COUNT unset) always honours the checkpoint
    pinned in config.yaml, so a deliberate "resume from an older one" choice
    is never silently overridden. Only an involuntary requeue -- preemption or
    node failure -- walks forward to the newest checkpoint on disk.
    """
    return (_AUTO_RESUME_TMPL
            .replace("@@CKPT@@", str(checkpoint_path))
            .replace("@@MAXR@@", str(max_restarts))
            .replace("@@TOTAL@@", str(iter_total))
            .rstrip("\n")
            .split("\n"))


def _slurm_script(cfg, stage_cfg, job_dir, array_len):
    hpc = stage_cfg["hpc"]
    python_bin = hpc.get("python_bin", "python")
    time = hpc.get("time", "04:00:00")
    mem = hpc.get("mem-per-gpu", "16G")
    cpus = hpc.get("cpus", "8")
    gpus = hpc.get("gpus", "a40:1")
    constraint = hpc.get("constraint", None)
    partition = hpc.get("partition", None)
    qos = hpc.get("qos", None)
    # Auto-resume on SLURM requeue (preemption / node failure). Opt-in per
    # hpc config. Only meaningful for fine-tune-hpc, which is the only stage
    # that both consumes a checkpoint and writes new ones.
    auto_resume = bool(hpc.get("auto_resume", False))
    max_restarts = int(hpc.get("max_restarts", 5))
    # account = hpc.get("account", None)
    extra_modules = hpc.get("extra_modules", [])

    conda = hpc.get("conda", None)
    env = hpc.get("env", None)
    work_path = hpc.get("work_path", None)

    script_path = os.path.join(job_dir, "submit_slurm.sh")
    log_dir = os.path.join(job_dir, "logs")
    _ensure_dir(log_dir)
    global_cfgs = load_global_config_path("magneton/config.yaml")
    # cfg_path = global_cfgs.get("instance_segmentation/gen_mask", "magneton/instance_segmentation/configs/config_gen_mask.yaml")
    cfg_file_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("config_file", "magneton/pytorch_connectomics/configs/config_file.yaml")
    )
    cfg_base_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("config_base", "magneton/pytorch_connectomics/configs/config_base.yaml")
    )
    checkpoint_path = (
        global_cfgs.get("affinity_prediction", {})
                .get("checkpoint", "magneton/pytorch_connectomics/configs/checkpoint.yaml")
    )

    # ITERATION_TOTAL is baked into the resume block so a requeue that lands
    # after training already finished exits cleanly instead of running a
    # degenerate zero-iteration job. Read before cfg_base_path can be
    # reassigned to a per-task temp config by the mutil_jobs branch.
    iter_total = 0
    if auto_resume and cfg.stage == "fine-tune-hpc":
        try:
            with open(cfg_base_path) as f:
                iter_total = int((yaml.safe_load(f) or {})
                                 .get("SOLVER", {}).get("ITERATION_TOTAL", 0))
        except Exception as e:
            print(f"[WARN] auto_resume: could not read ITERATION_TOTAL ({e}); "
                  f"completion guard disabled")

    # Direct-precomputed inference: detect by reading cfg-base's IMAGE_NAME.
    # If precomputed, replace temp-config generation with block-grid sizing
    # and pass --task-id / --chunks-per-task to run.py.
    precomputed_flag = False
    if cfg.stage == "inference-hpc":
        try:
            with open(cfg_base_path) as f:
                _base = yaml.safe_load(f) or {}
            _image_name = _base.get("INFERENCE", {}).get("IMAGE_NAME", None)
            _output_path = _base.get("INFERENCE", {}).get("OUTPUT_PATH", None)
            if is_precomputed_url(_image_name) and is_precomputed_url(_output_path):
                precomputed_flag = True
        except Exception:
            pass

    # Both legacy and precomputed paths are multi-job by nature, so
    # chunks_per_task/batch_num live under mutil_jobs_configs in either case.
    _mj_cfg = hpc.get("mutil_jobs_configs", {}) or {}
    chunks_per_task = int(_mj_cfg.get("chunks_per_task", hpc.get("chunks_per_task", 1)))
    if precomputed_flag:
        # Pre-create output volume now so workers find it ready.
        _geom = _base.get("INFERENCE", {}).get("GEOMETRY", {}) or {}
        _core = int(_geom.get("CORE_SIZE", 512))
        _chunk = int(_geom.get("OUTPUT_CHUNK_SIZE", 128))
        _mip = int(_geom.get("MIP", 0))
        _roi = _geom.get("ROI", None)
        vol_shape_zyx, vol_offset_zyx, _ = volume_info_from_cv(_image_name, mip=_mip)
        vol_shape_zyx, vol_offset_zyx = apply_roi(
            vol_shape_zyx, vol_offset_zyx, _roi, output_chunk_size=_chunk)
        # Mirror run.py: MTLSD trims the 10 LSD channels at write time.
        _out_planes = int(_base.get("INFERENCE", {}).get("OUTPUT_CHANNELS")
                          or _base.get("MODEL", {}).get("OUT_PLANES", 3))
        # Mirror run.py: optionally size the output to the ROI (offset = ROI
        # start). Must match the worker side or array tasks find a mismatched
        # volume. ZYX -> XYZ. See GEOMETRY.CROP_OUTPUT_TO_ROI.
        _crop_out = bool(_geom.get("CROP_OUTPUT_TO_ROI", False)) and _roi is not None
        _out_size_xyz = (vol_shape_zyx[2], vol_shape_zyx[1], vol_shape_zyx[0]) if _crop_out else None
        _out_offset_xyz = (vol_offset_zyx[2], vol_offset_zyx[1], vol_offset_zyx[0]) if _crop_out else None
        init_output_volume(_image_name, _output_path, mip=_mip,
                           num_channels=_out_planes,
                           dtype="uint8", chunk_size=_chunk,
                           output_size_xyz=_out_size_xyz, output_offset_xyz=_out_offset_xyz)
        n_blocks = block_count(vol_shape_zyx, core_size=_core)
        num_batches = _math.ceil(n_blocks / chunks_per_task)
        batch_num = int(_mj_cfg.get("batch_num", hpc.get("batch_num", num_batches)))
        print(f"[INFO] precomputed: {n_blocks} blocks "
              f"-> {num_batches} SLURM tasks (chunks_per_task={chunks_per_task})")
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pytc",
            f"#SBATCH --time={time}",
            f"#SBATCH --ntasks=1 --nodes=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem-per-gpu={mem}",
            f"#SBATCH --gpus={gpus}",
            f"#SBATCH --array=0-{num_batches-1}%{batch_num}",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]

    mutil_jobs_flag = hpc.get("mutil_jobs", False)
    if not precomputed_flag and mutil_jobs_flag and cfg.stage in ["inference-hpc",]:
        mutil_jobs_configs = hpc.get("mutil_jobs_configs", {})
        configs_save_path = mutil_jobs_configs.get("configs_save_path", '')
        input_folder = mutil_jobs_configs.get("input_folder", '')
        _ensure_dir(configs_save_path)
        batch_num = mutil_jobs_configs.get("batch_num", 1)
        chunks_per_task = mutil_jobs_configs.get("chunks_per_task", 1)
        num_batches = _gen_chunk_configs(
            cfg_base_path, configs_save_path, input_folder, chunks_per_task
        )
        print(
            f"[INFO] mutil_jobs: {len(os.listdir(input_folder))} chunks "
            f"-> {num_batches} SLURM tasks (chunks_per_task={chunks_per_task})"
        )
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pytc",
            f"#SBATCH --time={time}",
            f"#SBATCH --ntasks=1 --nodes=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem-per-gpu={mem}",
            f"#SBATCH --gpus={gpus}",
            f"#SBATCH --array=0-{num_batches-1}%{batch_num}",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]
        cfg_base_path = os.path.join(configs_save_path, "temp_config_${SLURM_ARRAY_TASK_ID}.yaml")

    elif not precomputed_flag:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pytc",
            f"#SBATCH --time={time}",
            f"#SBATCH --ntasks=1 --nodes=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem-per-gpu={mem}",
            f"#SBATCH --gpus={gpus}",
            f"#SBATCH --array=0-{array_len-1}",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]
    # A requeued job keeps its job ID, so %A resolves to the same log file;
    # the default open mode would truncate the preempted attempt's log away.
    if auto_resume and cfg.stage == "fine-tune-hpc":
        lines.append("#SBATCH --requeue")
        lines.append("#SBATCH --open-mode=append")
    if partition:   lines.append(f"#SBATCH --partition={partition}")
    if constraint: lines.append(f'#SBATCH --constraint="{constraint}"')
    if qos:         lines.append(f"#SBATCH --qos={qos}")

    # module load
    for m in extra_modules:
        lines.append(f"module load {m}")
        if m == "StdEnv":
            lines.append(f"export SLURM_EXPORT_ENV=ALL")
    if conda:       lines.append(f"source {conda}")
    if env:         lines.append(f"conda activate {env}")
    if work_path:   lines.append(f"cd {work_path}")

        
    if cfg.stage == "pre-train-hpc":
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path}"
        ]
    elif cfg.stage == "fine-tune-hpc":
        if auto_resume:
            lines += _auto_resume_bash(checkpoint_path, iter_total, max_restarts)
            _ckpt_arg = '"$CKPT"'
        else:
            _ckpt_arg = checkpoint_path
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path} --checkpoint {_ckpt_arg}"
        ]
    elif precomputed_flag:
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path} "
            f"--inference --checkpoint {checkpoint_path} "
            f"--task-id ${{SLURM_ARRAY_TASK_ID}} --chunks-per-task {chunks_per_task}"
        ]
    else:
        lines += [
            f"{python_bin} -u -m magneton.pytorch_connectomics.tools.run "
            f"--config-file {cfg_file_path} --config-base {cfg_base_path} --inference --checkpoint {checkpoint_path}"
        ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def submit_local_hpc(global_cfg, hpc_cfg, restart=False, dry_run=False):
    """
    Generate job lists and submission scripts (Slurm job arrays).
    Process a set of block indices locally on nodes using instance_segmentation.tools.gen_mask.
    """
    hpc = hpc_cfg.get("hpc", {})
    if not hpc.get("enable", False):
        print("[INFO] local_stage.hpc.enable=false, HPC submission is disabled.")
        return

    scheduler = hpc.get("scheduler", "slurm").lower()
    job_dir = hpc.get("job_dir", "./jobs/pytc")

    # Generate Script
    if scheduler == "slurm":
        script_path = _slurm_script(global_cfg, hpc_cfg, job_dir, 1)
        submit_cmd = ["sbatch", script_path]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    print(f"[INFO] Submit command: {' '.join(submit_cmd)}")
    if not dry_run:
        try:
            out = subprocess.check_output(submit_cmd, stderr=subprocess.STDOUT)
            out_msg = out.decode("utf-8", "ignore")
            print(f"[INFO] Submit output: {out_msg}")
        except Exception as e:
            print(f"[WARN] Submission failed:{e}")
            print(f"[HINT] You can manually execute the command:{' '.join(submit_cmd)}")

def run_hpc(global_cfg, hpc_cfg, restart=False, dry_run=False):
    submit_local_hpc(global_cfg, hpc_cfg, restart, dry_run)