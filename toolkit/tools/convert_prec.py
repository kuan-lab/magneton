# -*- coding: utf-8 -*-
import os
import h5py
import argparse
import numpy as np
import tifffile as tiff
from cloudvolume import CloudVolume
from magneton.toolkit.utils.config import load_config
import gc


class PrecConverter:
    def __init__(self, input_path, out_path, input_format, datasets, prec_info):
        self.input_path = input_path
        self.out_path = out_path
        self.input_format = input_format
        self.datasets = datasets
        self.prec_info = prec_info

    def convert(self):
        print(f"[INFO] Starting full (non-lazy) conversion...")
        print(f"[INFO] Reading input file: {self.input_path}")

        # --- Load input data --- #
        if self.input_format == "tif":
            full_data = tiff.imread(self.input_path)
        elif self.input_format == "h5":
            with h5py.File(self.input_path, "r") as f:
                for dataset in self.datasets:
                    f = f[dataset]
                full_data = f[:]
        else:
            raise RuntimeError("Unsupported input format. Use 'tif' or 'h5'.")

        print(f"[INFO] Loaded data shape: {full_data.shape}, dtype={full_data.dtype}")

        # --- Determine dimensionality --- #
        if len(full_data.shape) == 3:
            # (Z, Y, X)
            num_channels = 1
            volume_size = [full_data.shape[2], full_data.shape[1], full_data.shape[0]]
        elif len(full_data.shape) == 4:
            # (C, Z, Y, X)
            num_channels = full_data.shape[0]
            volume_size = [full_data.shape[3], full_data.shape[2], full_data.shape[1]]
        else:
            raise RuntimeError(f"Unsupported data dimensions: {len(full_data.shape)}")

        print(f"[INFO] Volume size (X,Y,Z): {volume_size}, Channels: {num_channels}")

        # --- Create CloudVolume --- #
        info = CloudVolume.create_new_info(
            num_channels=self.prec_info["num_channels"],
            layer_type=self.prec_info["layer_type"],
            data_type=self.prec_info["data_type"],
            encoding=self.prec_info["encoding"],
            resolution=self.prec_info["resolution"],
            voxel_offset=self.prec_info["voxel_offset"],
            chunk_size=self.prec_info["chunk_size"],
            volume_size=volume_size,
        )

        vol = CloudVolume(self.out_path, info=info, compress=self.prec_info["compress"])
        vol.commit_info()
        vol.commit_provenance()

        print(f"[INFO] Uploading to CloudVolume at: {self.out_path}")

        # --- Upload data --- #
        if num_channels == 1:
            full_data = np.transpose(full_data, (2, 1, 0))  # (X,Y,Z)
            assert vol.shape[:3] == full_data.shape
            vol[:, :, :] = full_data
        else:
            full_data = np.transpose(full_data, (3, 2, 1, 0))  # (X,Y,Z,C)
            assert vol.shape == full_data.shape
            vol[:, :, :, :] = full_data

        print("[INFO] Upload completed successfully.")
        del full_data
        gc.collect()

    def convert_lazy(self):
        print(f"[INFO] Starting lazy conversion for large dataset...")
        print(f"[INFO] Opening input file: {self.input_path}")

        # -------------------------
        # 1. Detect TIFF or HDF5
        # -------------------------
        if self.input_format == "tif":
            tiff_file = tiff.TiffFile(self.input_path)

            # ---- Detect OME-TIFF vs normal TIFF ---- #
            series = tiff_file.series[0]

            if len(series.shape) == 4:
                # OME-TIFF (C, Z, Y, X)
                full_shape = series.shape
                num_channels = full_shape[0]
                dtype = series.pages[0].asarray().dtype
                is_ome = True
                print(f"[INFO] Detected OME-TIFF with shape {full_shape} (C,Z,Y,X)")
            elif len(series.shape) == 3:
                # Normal TIFF (Z, Y, X)
                # print(series.shape)
                # num_pages = len(tiff_file.pages)
                # print(num_pages)
                sample_page = tiff_file.pages[0].asarray()
                # full_shape = (num_pages, *sample_page.shape)  # (Z,Y,X)
                full_shape = series.shape
                num_channels = 1
                dtype = sample_page.dtype
                is_ome = False
                print(f"[INFO] Detected standard TIFF with shape {full_shape} (Z,Y,X)")
            else:
                raise RuntimeError(f"Unsupported TIFF dimension: {series.shape}")

        elif self.input_format == "h5":
            f = h5py.File(self.input_path, "r")
            for dataset in self.datasets:
                f = f[dataset]
            full_shape = f.shape
            dtype = f.dtype
            num_channels = full_shape[0] if len(full_shape) == 4 else 1
            is_ome = False
            print(f"[INFO] Detected HDF5 dataset with shape {full_shape}")
        else:
            raise RuntimeError("Unsupported input format for lazy conversion.")

        print(f"[INFO] Data type: {dtype}")
        print(f"[INFO] Number of channels: {num_channels}")

        # -------------------------
        # 2. Volume size definition
        # -------------------------
        if len(full_shape) == 3:
            # (Z,Y,X)
            volume_size = [full_shape[2], full_shape[1], full_shape[0]]
        elif len(full_shape) == 4:
            # (C,Z,Y,X)
            volume_size = [full_shape[3], full_shape[2], full_shape[1]]
        else:
            raise RuntimeError(f"Unexpected data shape: {full_shape}")

        print(f"[INFO] Volume size (X,Y,Z): {volume_size}")

        # -------------------------
        # 3. Initialize CloudVolume
        # -------------------------
        info = CloudVolume.create_new_info(
            num_channels=self.prec_info["num_channels"],
            layer_type=self.prec_info["layer_type"],
            data_type=self.prec_info["data_type"],
            encoding=self.prec_info["encoding"],
            resolution=self.prec_info["resolution"],
            voxel_offset=self.prec_info["voxel_offset"],
            chunk_size=self.prec_info["chunk_size"],
            volume_size=volume_size,
        )

        vol = CloudVolume(self.out_path, info=info, compress=self.prec_info["compress"])
        vol.commit_info()
        vol.commit_provenance()

        chunk_size = self.prec_info["chunk_size"]
        cx, cy, cz = chunk_size
        x_size, y_size, z_size = volume_size
        print(f"[INFO] Uploading in chunks of {chunk_size} voxels...")

        # -------------------------
        # 4. Chunk-wise upload
        # -------------------------
        for z0 in range(0, z_size, cz):
            z1 = min(z0 + cz, z_size)
            print(f"[INFO] Processing Z-range [{z0}:{z1}]")

            # --- Load Z block --- #
            if self.input_format == "tif":
                if is_ome:
                    # (C,Z,Y,X)
                    z_block = series.asarray()[..., z0:z1, :, :]  # (C,ΔZ,Y,X)
                else:
                    # (Z,Y,X)
                    if len(tiff_file.pages) > 1:
                        # Multi-page TIFF, one Z-layer per page
                        z_block = np.stack(
                            [tiff_file.pages[z].asarray() for z in range(z0, z1)],
                            axis=0
                        )
                    else:
                        # Single-page 3D TIFF, read from series
                        full_stack = tiff_file.series[0].asarray()  # shape: (Z, Y, X)
                        z_block = full_stack[z0:z1, :, :]
            else:
                # HDF5
                z_block = f[z0:z1, :, :] if num_channels == 1 else f[:, z0:z1, :, :]

            # --- XY chunk upload --- #
            for y0 in range(0, y_size, cy):
                y1 = min(y0 + cy, y_size)
                for x0 in range(0, x_size, cx):
                    x1 = min(x0 + cx, x_size)

                    if num_channels == 1:
                        subvol = z_block[:, y0:y1, x0:x1]  # (Z,Y,X)
                        subvol = np.transpose(subvol, (2, 1, 0))  # (X,Y,Z)
                        vol[x0:x1, y0:y1, z0:z1] = subvol
                    else:
                        subvol = z_block[:, :, y0:y1, x0:x1]  # (C,Z,Y,X)
                        subvol = np.transpose(subvol, (3, 2, 1, 0))  # (X,Y,Z,C)
                        vol[x0:x1, y0:y1, z0:z1, :] = subvol

            del z_block
            import gc; gc.collect()
            print(f"[INFO] Completed Z-range [{z0}:{z1}]")

        print("[INFO] Lazy conversion finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Convert 3D/4D TIFF or HDF5 to Neuroglancer Precomputed format.")
    parser.add_argument("--config", default="configs/config_prec.yaml", type=str, help="Path to configuration YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    converter = PrecConverter(
        input_path=cfg["paths"]["input"],
        out_path=cfg["paths"]["output"],
        input_format=cfg["input_format"],
        datasets=cfg["h5"]["datasets"],
        prec_info=cfg["prec_info"],
    )

    if not cfg["prec_info"]["lazy"]:
        converter.convert()
    else:
        converter.convert_lazy()

def convert_prec(cfg):
    # print(cfg)
    converter = PrecConverter(
        input_path=cfg["paths"]["input"],
        out_path=cfg["paths"]["output"],
        input_format=cfg["input_format"],
        datasets=cfg["h5"]["datasets"],
        prec_info=cfg["prec_info"],
    )

    if not cfg["prec_info"]["lazy"]:
        converter.convert()
    else:
        converter.convert_lazy()

if __name__ == "__main__":
    main()
