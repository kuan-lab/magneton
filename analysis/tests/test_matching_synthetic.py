"""
Synthetic end-to-end verification of the cross-volume matchers.

Builds three tiny co-registered precomputed volumes in a temp dir:
  - bouton: one cube (label 7) at [20:44]^3
  - mito:   a sphere (label 3) inside the cube + an orphan sphere (label 5) outside
  - synapse: a blob (label 9) overlapping the cube's +x face + an isolated blob (label 11)

Then runs the REAL matchers (read_full / find_objects / CloudVolume slicing) and
asserts the expected assignments. Run:  python -m magneton.analysis.tests.test_matching_synthetic
"""
import shutil
import tempfile

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume

from magneton.analysis.lib.matching import match_mito_to_bouton, match_synapse_to_bouton


def _write_vol(path, arr_xyz):
    cv = CloudVolume.from_numpy(
        arr_xyz[:, :, :, None],          # (x,y,z,c)
        vol_path=f"file://{path}",
        resolution=(4, 4, 4),
        chunk_size=(64, 64, 64),
        layer_type="segmentation",
        progress=False,
    )
    return cv


def _sphere(arr, center, radius, label):
    cx, cy, cz = center
    xx, yy, zz = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
    arr[m] = label


def main():
    tmp = tempfile.mkdtemp(prefix="match_synth_")
    try:
        S = 64
        bouton = np.zeros((S, S, S), dtype=np.uint32)
        bouton[20:44, 20:44, 20:44] = 7

        mito = np.zeros((S, S, S), dtype=np.uint32)
        _sphere(mito, (32, 32, 32), 6, 3)     # inside the bouton cube
        _sphere(mito, (54, 54, 54), 3, 5)     # orphan, outside the cube

        synapse = np.zeros((S, S, S), dtype=np.uint32)
        synapse[40:48, 30:38, 30:38] = 9      # overlaps bouton on x in [40,44) → 4*8*8 = 256
        synapse[2:6, 2:6, 2:6] = 11           # isolated, no bouton

        bpath = f"{tmp}/bouton"; mpath = f"{tmp}/mito"; spath = f"{tmp}/synapse"
        _write_vol(bpath, bouton); _write_vol(mpath, mito); _write_vol(spath, synapse)

        # --- mito → bouton (mip 0; the synthetic volume only has mip0) ---
        df_mb = match_mito_to_bouton(f"file://{mpath}", f"file://{bpath}", mip=0)
        mb = df_mb.set_index("mito_seg_id")["parent_bouton"].to_dict()
        assert mb[3] == 7, f"mito 3 should be in bouton 7, got {mb[3]}"
        assert mb[5] == 0, f"mito 5 is orphan, got {mb[5]}"

        # --- synapse → bouton (direct overlap) ---
        syn_bboxes = pd.DataFrame([
            dict(seg_id=9,  bbox_x0=40, bbox_x1=48, bbox_y0=30, bbox_y1=38, bbox_z0=30, bbox_z1=38),
            dict(seg_id=11, bbox_x0=2,  bbox_x1=6,  bbox_y0=2,  bbox_y1=6,  bbox_z0=2,  bbox_z1=6),
        ])
        df_sb = match_synapse_to_bouton(f"file://{spath}", f"file://{bpath}", syn_bboxes).set_index("synapse_seg_id")
        assert df_sb.loc[9, "best_bouton"] == 7, df_sb.loc[9].to_dict()
        assert df_sb.loc[9, "overlap_voxels"] == 256, df_sb.loc[9].to_dict()
        assert df_sb.loc[9, "n_boutons_touched"] == 1, df_sb.loc[9].to_dict()
        assert df_sb.loc[11, "best_bouton"] == 0, df_sb.loc[11].to_dict()
        assert df_sb.loc[11, "overlap_voxels"] == 0, df_sb.loc[11].to_dict()

        print("\n[test] ALL ASSERTIONS PASSED")
        print("[test] mito→bouton:", mb)
        print("[test] synapse→bouton:\n", df_sb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
