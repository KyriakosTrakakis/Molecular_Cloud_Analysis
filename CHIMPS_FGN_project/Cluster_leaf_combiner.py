#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:17:14 2026

@author: kyriakos.trakakis
"""

import numpy as np
from astropy.io import fits

# ------------------------------------------------------------------
# File names
# ------------------------------------------------------------------
major_file = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_037_047_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters.fits"
minor_file = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_037_047_13CO_23arcsec_aligned_SNR_min_del_2_leaves.fits"

out_file = (
    "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_037_047_13CO_23arcsec_aligned_SNR_min_del_2_"
    "parallel_clusters_leaves.fits"
)

# ------------------------------------------------------------------
# Read FITS files
# ------------------------------------------------------------------
with fits.open(major_file, memmap=True) as hdul_major:
    major_data = hdul_major[0].data
    header = hdul_major[0].header

with fits.open(minor_file, memmap=True) as hdul_minor:
    minor_data = hdul_minor[0].data

# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if major_data.shape != minor_data.shape:
    raise ValueError("Major and minor FITS files have different shapes!")

# ------------------------------------------------------------------
# Combine according to your rules
# ------------------------------------------------------------------
# Start with everything = -1
combined = np.full(major_data.shape, -1, dtype=major_data.dtype)

# Where major has valid values, use major
mask_major = major_data > -1
combined[mask_major] = major_data[mask_major]

# Where major is -1 and minor has valid values, use minor
mask_minor_only = (major_data == -1) & (minor_data > -1)
combined[mask_minor_only] = minor_data[mask_minor_only]

# ------------------------------------------------------------------
# Write output FITS
# ------------------------------------------------------------------
hdu = fits.PrimaryHDU(data=combined, header=header)
hdu.writeto(out_file, overwrite=True)

print(f"Combined FITS written to:\n{out_file}")
