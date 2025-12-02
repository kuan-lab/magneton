import numpy as np
import tifffile
import argparse
from cloudvolume import CloudVolume
from scipy.ndimage import binary_erosion, binary_dilation
from magneton.toolkit.utils.config import load_config


def _gen_aff_mask(input, output, input_mip, min_region_size, max_region_size, erode_size, dilate_size, preview_tif_flag, preview_tif):

    mip = input_mip
    print(f"Enter precomputed volume:  {input}")
    cv_in = CloudVolume(input, mip=mip, progress=True, bounded=False, fill_missing=True)
    
    info = cv_in.info

    cv_out_info = CloudVolume.create_new_info(
        num_channels=info['num_channels'],
        layer_type="segmentation",
        data_type=info['data_type'],
        encoding="raw",
        resolution=info['scales'][mip]['resolution'],
        voxel_offset=info['scales'][mip]['voxel_offset'],
        chunk_size=info['scales'][mip]['chunk_sizes'][0],
        volume_size=info['scales'][mip]['size'],
    )
    
    cv_out = CloudVolume(output, info=cv_out_info, compress=False, progress=True)
    cv_out.commit_info()
    cv_out.commit_provenance()

    cv_preview = cv_in[:]
    labeled = cv_preview.copy()
    num = np.unique(labeled[labeled != 0]).size

    if num > 0:
        sizes = np.bincount(labeled.ravel())
        valid_labels = np.where(
            (sizes > min_region_size) &
            (sizes < max_region_size)
        )[0]
        print(valid_labels)
        print(f"Number of valid tags after filtering:{len(valid_labels)}")
        cv_preview = np.isin(labeled, valid_labels)

    else:
        cv_preview[:] = False  # Complete empty, safety processing
    cv_preview = cv_preview[:].squeeze()

    struct_e = np.ones((erode_size, erode_size, erode_size), dtype=bool)
    struct_d = np.ones((dilate_size, dilate_size, dilate_size), dtype=bool)
    eroded = binary_erosion(cv_preview, structure=struct_e)
    smoothed = binary_dilation(eroded, structure=struct_d)
    
    preview_data = smoothed.astype(info['data_type'])

    cv_out[:] = preview_data
    if preview_tif_flag:
        tifffile.imwrite(preview_tif, preview_data.astype(np.uint8), bigtiff=True)
    print(f"Done! Output:{output}, {preview_tif}")

def main():
    parser = argparse.ArgumentParser(description="Generate a Mask for Affinity Map.")
    parser.add_argument("--config", default="config_gen_mask.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mask_flag = cfg["mask"]["mask_flag"]
    input = cfg["mask"]["input"]
    output = cfg["mask"]["output"]
    input_mip = cfg["mask"]["input_mip"]
    preview_tif_flag = cfg["mask"]["preview_tif_flag"]
    preview_tif = cfg["mask"]["preview_tif"]
    min_region_size = cfg["mask"]["min_region_size"]
    max_region_size = cfg["mask"]["max_region_size"]
    erode_size = cfg["mask"]["erode_size"]
    dilate_size = cfg["mask"]["dilate_size"]

    if mask_flag:
        _gen_aff_mask(input, output, input_mip, min_region_size, max_region_size, erode_size, dilate_size, preview_tif_flag, preview_tif )
    else:
        print('Generate flag is false.')

def gen_aff_mask(cfg):
    mask_flag = cfg["mask"]["mask_flag"]
    input = cfg["mask"]["input"]
    output = cfg["mask"]["output"]
    input_mip = cfg["mask"]["input_mip"]
    preview_tif_flag = cfg["mask"]["preview_tif_flag"]
    preview_tif = cfg["mask"]["preview_tif"]
    min_region_size = cfg["mask"]["min_region_size"]
    max_region_size = cfg["mask"]["max_region_size"]
    erode_size = cfg["mask"]["erode_size"]
    dilate_size = cfg["mask"]["dilate_size"]

    if mask_flag:
        gen_aff_mask(input, output, input_mip, min_region_size, max_region_size, erode_size, dilate_size, preview_tif_flag, preview_tif )
    else:
        print('Generate flag is false.')

if __name__ == "__main__":
    main()
