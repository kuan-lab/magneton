import numpy as np
from cloudvolume import CloudVolume
from tqdm import tqdm
import argparse
from magneton.toolkit.utils.config import load_config


def apply_mask_to_precomputed(
    raw_path, 
    mask_path, 
    output_path, 
    mip=0, 
    fill_missing=True, 
    bounded=True, 
    progress=True
):
    raw_vol = CloudVolume(raw_path, mip=mip, fill_missing=fill_missing, bounded=bounded, progress=progress)
    mask_vol = CloudVolume(mask_path, mip=mip, fill_missing=fill_missing, bounded=bounded, progress=progress)

    # Create output volume
    info = CloudVolume.create_new_info(
        num_channels=raw_vol.num_channels,
        layer_type=raw_vol.layer_type,
        data_type=raw_vol.dtype,
        encoding=raw_vol.encoding,
        resolution=raw_vol.resolution,
        voxel_offset=raw_vol.voxel_offset,
        chunk_size=raw_vol.chunk_size,
        volume_size=raw_vol.volume_size,
    )
    out_vol = CloudVolume(output_path, info=info, mip=mip, progress=progress, compress=False)
    out_vol.commit_info()
    out_vol.commit_provenance()

    X, Y, Z = raw_vol.volume_size
    cx, cy, cz = raw_vol.chunk_size

    # triple for loop over chunks
    for z in tqdm(range(0, Z, cz), desc="Z-slices"):
        try:
            for y in range(0, Y, cy):
                for x in range(0, X, cx):
                    x1, y1, z1 = min(x+cx, X), min(y+cy, Y), min(z+cz, Z)

                    raw_block = raw_vol[x:x1, y:y1, z:z1]
                    mask_block = mask_vol[x:x1, y:y1, z:z1]

                    if raw_block is None or mask_block is None:
                        continue

                    # Apply mask
                    # mask_bool = mask_block.astype(bool)
                    masked_block = raw_block * mask_block

                    out_vol[x:x1, y:y1, z:z1] = masked_block
        except KeyboardInterrupt:
            break

    print(f"Done! Masked dataset saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Mask Neuroglancer Precomputed data.")
    parser.add_argument("--config", default="config_mask.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = cfg["mask"]["raw_path"]
    mask_path = cfg["mask"]["mask_path"]
    output_path = cfg["mask"]["output_path"]
    mip = cfg["mask"]["mip"]
    apply_mask_to_precomputed(
        raw_path=raw_path,
        mask_path=mask_path,
        output_path=output_path,
        mip=mip
    )

def mask_prec(cfg):
    raw_path = cfg["mask"]["raw_path"]
    mask_path = cfg["mask"]["mask_path"]
    output_path = cfg["mask"]["output_path"]
    mip = cfg["mask"]["mip"]
    apply_mask_to_precomputed(
        raw_path=raw_path,
        mask_path=mask_path,
        output_path=output_path,
        mip=mip
    )

if __name__=='__main__':
    main()
