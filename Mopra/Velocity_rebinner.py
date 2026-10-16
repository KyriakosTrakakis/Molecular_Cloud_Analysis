#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:44:49 2025

@author: kyriakos.trakakis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

input_file  = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/12CO/G310-12CO.fits"          # your input file
output_file = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/12CO/G310-12CO_05res.fits" # output file
rebin_factor = 10

with fits.open(input_file, memmap=True) as hdul:
    data = hdul[0].data  # shape expected (l, b, v) = (1213, 263, 11071)
    header = hdul[0].header.copy()

# --- Sanity check ---
print("Original shape:", data.shape)

l, b, v = data.shape  # 11139, 263, 1213

# Trim velocity axis to multiple of 5
trim = l % rebin_factor
if trim > 0:
    data = data[:-trim,:, :]

# Reshape velocity axis into groups of 5
groups = data.shape[0] // rebin_factor  # 11135 // 5 = 2227
data_reshaped = data.reshape(groups, rebin_factor, b, v)  # shape: (2227, 5, 263, 1213)

# Average over the last axis (groups of 5)
rebinned = data_reshaped.mean(axis=1)

print("Rebinned shape:", rebinned.shape)

# --- Update header ---
# CDELT3 (velocity step) should be scaled by rebin_factor
if "CDELT3" in header:
    header["CDELT3"] *= rebin_factor
if "CRPIX3" in header:
    header["CRPIX3"] = (header["CRPIX3"] - 0.5) / rebin_factor + 0.5

# Save new FITS file
hdu = fits.PrimaryHDU(data=rebinned.astype(np.float32), header=header)
hdu.writeto(output_file, overwrite=True)

print(f"Saved rebinned cube to {output_file}")
