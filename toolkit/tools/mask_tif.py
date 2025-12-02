import numpy as np
import tifffile as tiff
import argparse
from magneton.toolkit.utils.config import load_config


def _mask_tif(
    img_path,
    mask_path,
    output_tif_path,
    mask_reverse=False
):
    """
    Apply a binary mask to a large multi-channel 3D BigTIFF and save result as BigTIFF.
    Process is done slice-by-slice to avoid loading full dataset in memory.

    Args:
        img_path (str): Path to the 3D TIFF image (C, Z, Y, X).
        mask_path (str): Path to the 3D TIFF mask (Z, Y, X).
        output_tif_path (str): Path for output BigTIFF.
        chunk_size (tuple): Currently unused, reserved for future block processing.
    """

    img_tif = tiff.imread(img_path)
    C, Z, Y, X = img_tif.shape
    print(f"Image shape: C={C}, Z={Z}, Y={Y}, X={X}")

    masktif = tiff.imread(mask_path)
    if len(masktif.shape) == 3:
        if mask_reverse:
            masktif = np.transpose(masktif, (2, 1, 0))
        print(f"Mask shape: Z={masktif.shape[0]}, Y={masktif.shape[1]}, X={masktif.shape[2]}")
        
        img_tif[0] = img_tif[0] * masktif
        img_tif[1] = img_tif[1] * masktif
        img_tif[2] = img_tif[2] * masktif
    elif len(masktif.shape) == 4:
        if mask_reverse:
            masktif = np.transpose(masktif, (3, 2, 1, 0))
        print(f"Mask shape: Z={masktif.shape[1]}, Y={masktif.shape[2]}, X={masktif.shape[3]}, C={masktif.shape[0]},")
        
        img_tif[0] = img_tif[0] * masktif[0]
        img_tif[1] = img_tif[1] * masktif[1]
        img_tif[2] = img_tif[2] * masktif[2]
    
    tiff.imwrite(output_tif_path, img_tif)
    print(f"Done. Saved masked image to {output_tif_path}")

def main():
    parser = argparse.ArgumentParser(description="Mask Tif data.")
    parser.add_argument("--config", default="config_mask_tif.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = cfg["mask"]["raw_path"]
    mask_path = cfg["mask"]["mask_path"]
    output_path = cfg["mask"]["output_path"]
    mask_reverse = cfg["mask"]["mask_reverse"]
    _mask_tif(raw_path, mask_path, output_path, mask_reverse)

def mask_tif(cfg):
    raw_path = cfg["mask"]["raw_path"]
    mask_path = cfg["mask"]["mask_path"]
    output_path = cfg["mask"]["output_path"]
    mask_reverse = cfg["mask"]["mask_reverse"]
    _mask_tif(raw_path, mask_path, output_path, mask_reverse)

if __name__ == "__main__":
    main()
