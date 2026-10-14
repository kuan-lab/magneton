#!/usr/bin/env python
"""Axon / dendrite classification from the skeleton morphometrics' radius.

Caliber is the robust axon/dendrite signal: split errors strip spines off many
dendrites (so spine density under-reads), but the shaft stays much thicker than
any axon. The radius used per neurite, in order of preference:
  r_median_nm   median cross-section radius, if >= --min-profile-um of valid profile
  r_vpl_nm      sqrt(V / (pi L)) volume per path length, if path >= --min-path-um
                (tracks r_median within ~2-6%, correlation 0.90-0.96 on fib_b/c/f)
  otherwise     'short' (too little of it inside the volume to call)
dendrite if that radius >= --cut-nm (220: sparsest region shared by fib_b/c/f).

glia: thick glial processes pass the caliber test, and a volume-clipped one can
look like an extremely spiny dendrite. They are sheet-like, so their cross-
sections are irregular (circularity 4 pi A / P^2 < --glia-circ) AND they carry far
more surface than their caliber implies (surface_excess = r_class / (1.55 x
2V/SA radius) > --glia-excess; 1.55 is the typical cross-section / 2V/SA ratio,
so a smooth tube sits near 1). User-verified 8/8 on fib_c + fib_f (2026-09-29),
including small sections. Needs cross-sections, so volume-per-length-only
neurites (no circularity) cannot be flagged.

Rewrites morphometrics/neurites.csv in place with r_class_nm / r_source / cls
(the 2V/SA class from neurite_filter.py is kept as cls_2vsa), and with --tag
re-writes the volume's Neuroglancer segment-property tags.
"""
import argparse, os
import numpy as np
import pandas as pd

from magneton.analysis.neurite_filter import tag_segment_properties

FILT = "/gpfs/marilyn/pi/kuan/shared/marmoset_project/analysis_outputs/neurite_filter"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--cut-nm", type=float, default=220.0)
    ap.add_argument("--min-profile-um", type=float, default=2.0)
    ap.add_argument("--min-path-um", type=float, default=1.0)
    ap.add_argument("--glia-circ", type=float, default=0.45)
    ap.add_argument("--glia-excess", type=float, default=1.5)
    ap.add_argument("--tag", action="store_true")
    args = ap.parse_args()

    path = os.path.join(FILT, f"{args.seg}_mip{args.mip}", "morphometrics", "neurites.csv")
    d = pd.read_csv(path)
    if "cls_2vsa" not in d:
        d["cls_2vsa"] = d.cls
    d["r_vpl_nm"] = np.sqrt(d.volume_um3 / (np.pi * d.path_length_um)) * 1000.0
    prof = d.profile_length_um.fillna(0) >= args.min_profile_um
    vpl = ~prof & (d.path_length_um.fillna(0) >= args.min_path_um)
    d["r_source"] = np.select([prof, vpl], ["cross_section", "volume_per_length"], "none")
    d["r_class_nm"] = np.select([prof, vpl], [d.r_median_nm, d.r_vpl_nm], np.nan)
    d["cls"] = np.where(d.r_source == "none", "short",
                        np.where(d.r_class_nm >= args.cut_nm, "dendrite", "axon"))
    d["surface_excess"] = d.r_class_nm / (1.55 * d.r_2v_sa_nm)
    glia = ((d.cls != "short") & (d.circularity < args.glia_circ)
            & (d.surface_excess > args.glia_excess))
    d.loc[glia, "cls"] = "glia"
    d.to_csv(path, index=False)

    print(pd.crosstab(d.cls_2vsa, d.cls, margins=True))
    print(d.groupby(["r_source", "cls"]).size().unstack(fill_value=0))
    if args.tag:
        t = d.assign(neurite=True, radius_nm=d.r_class_nm)
        tag_segment_properties(args.seg, t)


if __name__ == "__main__":
    main()
