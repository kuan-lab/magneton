import numpy as np
import tifffile

def export_tif_from_volume(out_vol, save_path: str, max_slices: int = 200):
    """
    Export preview TIFF from CloudVolume
    - By default, max_slices Z slices are exported before exporting
    """
    print(f"[INFO] Exporting TIFF preview → {save_path}")
    size_xyz = out_vol.volume_size
    z_dim = size_xyz[2]
    n = min(int(max_slices), int(z_dim))

    vol = out_vol[:, :, :n]   # (x,y,z,1)
    seg = vol[:, :, :, 0]
    seg = np.transpose(seg, (2, 1, 0))  # (z,y,x)

    tifffile.imwrite(save_path, seg.astype(np.uint32), dtype=np.uint32)
    print(f"[DONE] TIFF saved: {save_path}, shape={seg.shape}")
