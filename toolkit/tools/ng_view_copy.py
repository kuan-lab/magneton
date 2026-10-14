#!/usr/bin/env python
"""Make a Neuroglancer-friendly viewing copy of a precomputed volume.

Pipeline outputs use isotropic 128^3 raw chunks, which is what the block-wise
stages want but is slow to browse on anisotropic ssTEM: one xy slice of a 30 nm
stack pulls 128 sections per chunk (16.8 MB/chunk for uint64 seg). This tool
re-chunks to thin-z chunks (default 256x256x16, like BossDB's 512x512x16),
re-encodes (compressed_segmentation for seg -- lossless; jpeg for images --
lossy, VIEWING ONLY), restricts to an ROI, skips all-zero chunks, and builds a
deeper pyramid: `xy_mips` levels of (2,2,1) then `iso_mips` levels of (2,2,2).

compress=False everywhere: gzip-wrapped (.gz) chunk files are not decoded when
served by nginx / file:// (see lab notebook 2026-03-25, 04-16, 05-08).

bounds = [z1, z2, y1, y2, x1, x2] at mip 0. They are snapped OUTWARD to
chunk * 2^(xy_mips+iso_mips) in xy and chunk_z * 2^iso_mips in z, otherwise
igneous raises AlignmentError (and downsample_prec exits 0 regardless).
"""
import argparse
import math

from cloudvolume import Bbox, CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


def snap(lo, hi, m):
    return (lo // m) * m, int(math.ceil(hi / m)) * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bounds", type=int, nargs=6, required=True, metavar=("Z1", "Z2", "Y1", "Y2", "X1", "X2"))
    ap.add_argument("--chunk", type=int, nargs=3, default=[256, 256, 16], metavar=("X", "Y", "Z"))
    ap.add_argument("--encoding", required=True, help="compressed_segmentation | jpeg | raw")
    ap.add_argument("--xy-mips", type=int, default=3)
    ap.add_argument("--iso-mips", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--memory-target", type=float, default=3.5e9)
    args = ap.parse_args()

    z1, z2, y1, y2, x1, x2 = args.bounds
    cx, cy, cz = args.chunk
    mxy = 2 ** (args.xy_mips + args.iso_mips)
    x1, x2 = snap(x1, x2, cx * mxy); y1, y2 = snap(y1, y2, cy * mxy)
    z1, z2 = snap(z1, z2, cz * 2 ** args.iso_mips)
    bounds = Bbox([x1, y1, z1], [x2, y2, z2])
    print(f"[INFO] snapped bounds (xyz): {bounds}")
    src, dst = "file://" + args.src.rstrip("/"), "file://" + args.dst.rstrip("/")
    common = dict(encoding=args.encoding, compress=False, delete_black_uploads=True,
                  fill_missing=True, memory_target=int(args.memory_target))

    def run(tasks, what):
        tasks = list(tasks)
        print(f"[INFO] {what}: {len(tasks)} tasks", flush=True)
        tq = LocalTaskQueue(parallel=args.workers)
        tq.insert(tasks)
        tq.execute()

    run(tc.create_transfer_tasks(src, dst, chunk_size=args.chunk, mip=0, bounds=bounds,
                                 skip_downsamples=True, **common), "transfer mip0")
    if args.xy_mips:
        run(tc.create_downsampling_tasks(dst, mip=0, num_mips=args.xy_mips, factor=(2, 2, 1),
                                         bounds=bounds, bounds_mip=0, chunk_size=args.chunk,
                                         **common), f"downsample {args.xy_mips} x (2,2,1)")
    if args.iso_mips:
        run(tc.create_downsampling_tasks(dst, mip=args.xy_mips, num_mips=args.iso_mips, factor=(2, 2, 2),
                                         bounds=bounds, bounds_mip=0, chunk_size=args.chunk,
                                         **common), f"downsample {args.iso_mips} x (2,2,2)")
    info = CloudVolume(dst, progress=False).info
    print("[DONE]", [(s["key"], s["chunk_sizes"][0], s["encoding"]) for s in info["scales"]])


if __name__ == "__main__":
    main()
