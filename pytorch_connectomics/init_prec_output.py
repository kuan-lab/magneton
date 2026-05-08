"""Pre-create the output precomputed volume for direct-precomputed inference.

Copies geometry (resolution, voxel_offset, volume_size) from the input
precomputed and sets dtype=uint8, num_channels=3, configurable chunk_size.
Idempotent: if the output already has an info file, the existing volume is
returned unchanged.
"""
from cloudvolume import CloudVolume


def init_output_volume(
    input_url: str,
    output_url: str,
    *,
    mip: int = 0,
    num_channels: int = 3,
    dtype: str = "uint8",
    chunk_size: int = 128,
    encoding: str = "raw",
    compress: bool = False,
) -> CloudVolume:
    try:
        out_vol = CloudVolume(output_url, mip=mip, progress=False)
        print(f"[INFO] Output volume already exists at {output_url} — reusing")
        return out_vol
    except Exception:
        pass

    in_vol = CloudVolume(input_url, mip=mip, progress=False)
    info = CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type="image",
        data_type=dtype,
        encoding=encoding,
        resolution=in_vol.resolution,
        voxel_offset=list(in_vol.voxel_offset),
        volume_size=list(in_vol.info["scales"][mip]["size"]),
        chunk_size=[chunk_size, chunk_size, chunk_size],
    )
    out_vol = CloudVolume(output_url, info=info, compress=compress, progress=False)
    out_vol.commit_info()
    out_vol.commit_provenance()
    print(f"[INFO] Created output volume at {output_url} "
          f"(size={list(in_vol.info['scales'][mip]['size'])}, "
          f"chunk={chunk_size}, dtype={dtype}, channels={num_channels})")
    return out_vol
