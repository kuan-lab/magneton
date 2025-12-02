import numpy as np
import tifffile
from scipy.ndimage import zoom
import h5py
import os
import argparse
from magneton.toolkit.utils.config import load_config
from fractions import Fraction


def image_resize_3c(file_path, output_path, zoom_factor, zoom_order=0):
    print(f"Processing: {file_path}")
    image_ac = tifffile.imread(file_path)
    print(image_ac.shape, image_ac.dtype)
    upsampled = zoom(image_ac, zoom=zoom_factor, order=zoom_order)  # Interpolation
    print(upsampled.shape)
    tifffile.imwrite(output_path, upsampled)
    print(f"Saved to: {output_path}")

def image_resize_4c(file_path, output_path, zoom_factor, zoom_order):
    print(f"Processing: {file_path}")
    image_ac = tifffile.imread(file_path)
    upsampled_ac = []
    print(image_ac.shape, image_ac.dtype)
    for image in image_ac:
        print(image.shape, image.dtype)
        upsampled = zoom(image, zoom=zoom_factor, order=zoom_order)  # Interpolation
        print(upsampled.shape)
        upsampled_ac.append(upsampled)
    upsampled_ac = np.array(upsampled_ac, dtype=image_ac.dtype)
    tifffile.imwrite(output_path, upsampled_ac)
    print(f"Saved to: {output_path}")

def _image_resize(file_path, output_path, zoom_factor, zoom_order=0):
    print(f"Processing: {file_path}")
    image_ac = tifffile.imread(file_path)
    if len(image_ac.shape) == 3:
        image_resize_3c(file_path, output_path, zoom_factor, zoom_order)
    elif len(image_ac.shape) == 4:
        image_resize_4c(file_path, output_path, zoom_factor, zoom_order)

def main():
    parser = argparse.ArgumentParser(description="Resize Tif data.")
    parser.add_argument("--config", default="config_resize_tif.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    input = cfg["resize"]["input"]
    output = cfg["resize"]["output"]
    zoom_factor = cfg["resize"]["zoom_factor"]
    zoom_factor = [float(Fraction(val)) for val in zoom_factor]
    zoom_order = cfg["resize"]["zoom_order"]
    # print(zoom_factor, zoom_order)
    _image_resize(input, output, zoom_factor, zoom_order)

def resize_tif(cfg):
    
    input = cfg["resize"]["input"]
    output = cfg["resize"]["output"]
    zoom_factor = cfg["resize"]["zoom_factor"]
    zoom_factor = [float(Fraction(val)) for val in zoom_factor]
    zoom_order = cfg["resize"]["zoom_order"]
    # print(zoom_factor, zoom_order)
    _image_resize(input, output, zoom_factor, zoom_order)
    

if __name__=="__main__":
    main()
    