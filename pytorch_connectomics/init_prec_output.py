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
    output_size_xyz=None,
    output_offset_xyz=None,
) -> CloudVolume:
    # `mip` selects the scale to read from the *input* pyramid. The output is
    # freshly created as a single-scale precomputed (scales[0] = input's chosen
    # mip), so we always open the output at mip=0.
    #
    # By default the output spans the FULL input extent (offset 0), so an ROI
    # run writes sparse data into the full coordinate frame — keeping absolute
    # coords aligned with the source for Neuroglancer and the origin-(0,0,0)
    # instance-seg pipeline. Pass output_size_xyz/output_offset_xyz (both in the
    # chosen mip's voxel units, XYZ) to instead size the output to an ROI; the
    # offset MUST be the ROI start so the inference write path (which writes
    # cores at absolute coords) lands inside the volume's bounds unchanged.
    try:
        out_vol = CloudVolume(output_url, mip=0, progress=False)
        print(f"[INFO] Output volume already exists at {output_url} — reusing")
        return out_vol
    except Exception:
        pass

    in_vol = CloudVolume(input_url, mip=mip, progress=False)
    vol_size = list(output_size_xyz) if output_size_xyz is not None \
        else list(in_vol.info["scales"][mip]["size"])
    vox_offset = list(output_offset_xyz) if output_offset_xyz is not None \
        else list(in_vol.voxel_offset)
    info = CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type="image",
        data_type=dtype,
        encoding=encoding,
        resolution=in_vol.resolution,
        voxel_offset=vox_offset,
        volume_size=vol_size,
        chunk_size=[chunk_size, chunk_size, chunk_size],
    )
    out_vol = CloudVolume(output_url, info=info, compress=compress, progress=False)
    out_vol.commit_info()
    out_vol.commit_provenance()
    print(f"[INFO] Created output volume at {output_url} "
          f"(size={vol_size}, offset={vox_offset}, "
          f"chunk={chunk_size}, dtype={dtype}, channels={num_channels})")
    return out_vol
