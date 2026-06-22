"""
Stage — crop EM + affinity inference from precomputed, convert the affinity to a
binary membrane mask, and upload EM (image) + membrane (annotation) to WebKnossos.

This is the "membrane annotation" proofreading entry: instead of expanding a
skeleton, a human corrects the thresholded-affinity membrane directly in the WKS
volume tool. It compresses the old manual flow (crop EM tif -> crop affinity ->
threshold/invert -> two-step upload) into one command.

Membrane = where the affinity is LOW (a boundary between objects). With PyTC
affinities (high = same object, ~255 interior; low = ~0 across a membrane), the
membrane is `reduce_over_channels(aff) < threshold`. The affinity histogram is
strongly bimodal (peaks at 0 and 255), so any threshold in ~[10, 200] gives
nearly the same mask; only the ~[235, 254] range cuts into the interior peak.
See the fennel GT workflow note for the calibration.

Crop + threshold run in the `magneton` env (CloudVolume). The WKS upload is
shelled out to the `yf354` env (webknossos-libs), so the whole thing is one
invocation.  Run directly:
    python -m magneton.proofreading.stages.membrane --config <cfg>
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from magneton.proofreading.config import (
    load_config, get_stage_config, strip_file_prefix,
)

UPLOAD_SCRIPT = "/gpfs/radev/home/yf354/magneton/upload_webknossos.py"


def _read_roi(path, mip, coords):
    """Read a precomputed ROI.

    coords = [z1, z2, y1, y2, x1, x2] in the chosen mip's voxel space.
    Returns (C, Z, Y, X) in the volume's native dtype (C=1 for single-channel).
    """
    from cloudvolume import CloudVolume
    z1, z2, y1, y2, x1, x2 = coords
    vol = CloudVolume(path, mip=mip, fill_missing=True, progress=False)
    data = np.asarray(vol[x1:x2, y1:y2, z1:z2])     # (X, Y, Z, C)
    return np.transpose(data, (3, 2, 1, 0))         # -> (C, Z, Y, X)


def _to_membrane(aff, reduce, threshold, value):
    """aff: (C, Z, Y, X).  membrane = reduce_over_C(aff) < threshold -> value."""
    if reduce == "min":
        comb = aff.min(axis=0)        # interior only where ALL channels are high
    elif reduce == "max":
        comb = aff.max(axis=0)
    elif reduce == "mean":
        comb = aff.mean(axis=0)
    else:
        raise ValueError(f"channel_reduce must be min|max|mean, got '{reduce}'")
    mem = (comb < threshold).astype(np.uint8) * np.uint8(value)
    return mem


def membrane(cfg: dict):
    import tifffile

    paths = get_stage_config(cfg, "paths")
    m = get_stage_config(cfg, "membrane")
    out_dir = strip_file_prefix(paths["output"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    coords = m["coords"]                            # [z1,z2,y1,y2,x1,x2], common (post-mip) space
    inf = m["inference"]
    emc = m["em"]
    reduce = inf.get("channel_reduce", "min")
    thr = inf.get("threshold", 140)        # production value, swept Apr 2026 (min/140)

    t0 = time.time()
    aff = _read_roi(strip_file_prefix(inf["path"]), inf.get("mip", 0), coords)
    mem = _to_membrane(aff, reduce, thr, inf.get("membrane_value", 255))
    em = _read_roi(strip_file_prefix(emc["path"]), emc.get("mip", 0), coords)[0]   # (Z,Y,X)
    if em.shape != mem.shape:
        raise ValueError(
            f"EM {em.shape} and membrane {mem.shape} shapes differ — check that "
            "em.mip and inference.mip resolve to the same resolution for these coords")
    frac = float((mem > 0).mean()) * 100.0
    print(f"[membrane] aff{tuple(aff.shape)} reduce={reduce} thr={thr} "
          f"-> membrane {frac:.1f}% of voxels  ({time.time()-t0:.1f}s)", flush=True)

    em_tif = os.path.join(out_dir, "em.tif")
    mem_tif = os.path.join(out_dir, "membrane.tif")
    tifffile.imwrite(em_tif, em)                    # (Z, Y, X)
    tifffile.imwrite(mem_tif, mem)
    print(f"[membrane] wrote {em_tif} + {mem_tif}", flush=True)

    up = m.get("upload") or {}
    if up.get("enable", False):
        _upload(em_tif, mem_tif, m, up, out_dir)
    else:
        print("[membrane] upload disabled; upload manually with /wk", flush=True)
    return em_tif, mem_tif


def _folder_id(rf):
    """Resolve a WKS folder spec to what upload_webknossos expects.

    Accepts a dashboard URL (.../datasets/<name>-<24hex>), a bare 24-hex id, or a
    plain folder path/name. Extracts the trailing 24-hex id when present (URL or
    "name-<id>"); otherwise passes the value through as a folder path.
    """
    if not rf:
        return None
    seg = str(rf).rstrip("/").split("/")[-1]      # last URL/path segment
    m = re.search(r"([0-9a-f]{24})$", seg)
    return m.group(1) if m else str(rf)


def _upload(em_tif, mem_tif, m, up, out_dir):
    token = up.get("token") or os.environ.get("WEBKNOSSOS_TOKEN")
    vx, vy, vz = m.get("voxel_size", [8, 8, 8])
    name = up.get("name") or "membrane_" + "_".join(str(c) for c in m["coords"])
    # local wkw scratch dir the upload script builds before pushing to WKS.
    # `or` (not .get's default) so an explicit `out_dir: null` still falls back.
    scratch = up.get("out_dir") or os.path.join(out_dir, "wk_upload_out")
    cmd = [
        "conda", "run", "-n", up.get("env", "yf354"), "python", UPLOAD_SCRIPT,
        "--img-files", em_tif,
        "--seg-files", mem_tif,
        "--voxel-sizes", f"({vx},{vy},{vz})",
        "--names", name,
        "--out-dir", scratch,
    ]
    folder = _folder_id(up.get("remote_folder"))
    if folder:
        cmd += ["--remote-folder", folder]
    if not token:
        print("[membrane] WEBKNOSSOS_TOKEN unset; TIFs written but NOT uploaded.\n"
              "  Set WEBKNOSSOS_TOKEN (or upload.token) and rerun, or run manually:\n  "
              + " ".join(cmd) + ' --token "$WEBKNOSSOS_TOKEN"', flush=True)
        return
    print(f"[membrane] uploading '{name}' via {up.get('env','yf354')} env ...", flush=True)
    subprocess.run(cmd + ["--token", token], check=True)

    if up.get("cleanup", False):
        # upload_webknossos builds the local wkw at <scratch>/<name>; drop it now
        # that the push succeeded (subprocess.run(check=True) raised otherwise).
        import shutil
        wkw_dir = os.path.join(scratch, name)
        if os.path.isdir(wkw_dir):
            shutil.rmtree(wkw_dir)
            print(f"[membrane] cleaned up wkw scratch {wkw_dir}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="proofreading — membrane crop+threshold+upload")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    membrane(load_config(args.config))


if __name__ == "__main__":
    main()
