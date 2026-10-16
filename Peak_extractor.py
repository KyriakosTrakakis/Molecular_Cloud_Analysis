#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:49:13 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
import os

os.chdir('/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO')
#os.chdir('/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO')


# ------------------------------------------------------------------
# File names
# ------------------------------------------------------------------
mask_file = "FGN_027_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters.fits"
#data_file = "FGN_017_027_13CO_23arcsec_aligned_SNR.fits" # Original code used SNR here
data_file = "FGN_027_12_13_ratio_23arcsec_aligned.fits"
#data_file = "G320-13CO_smallercube_05res.fits"
#mask_file = "G320-13CO_smallercube_05res_min_del_2_parallel_clusters.fits"

# ------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------
with fits.open(mask_file, memmap=True) as hdul:
    mask_data = hdul[0].data.astype(np.int32)

with fits.open(data_file, memmap=True) as hdul:
    data = hdul[0].data.astype(np.float64)

# ------------------------------------------------------------------
# Basic sanity checks
# ------------------------------------------------------------------
if mask_data.shape != data.shape:
    raise ValueError("Mask cube and data cube have different shapes!")

# ------------------------------------------------------------------
# Get structure IDs (ignore background)
# ------------------------------------------------------------------
structure_ids = np.unique(mask_data)
structure_ids = structure_ids[structure_ids > 0]  # ignore background (0)

print(f"Found {len(structure_ids)} structures")

# ------------------------------------------------------------------
# Compute peak value for each structure
# ------------------------------------------------------------------
peaks = {}

for sid in structure_ids:
    structure_mask = (mask_data == sid)

    # Extract data values within the structure
    values = data[structure_mask]

    # Guard against empty or all-NaN cases
    if values.size == 0 or np.all(~np.isfinite(values)):
        peak = np.nan
    else:
        peak = np.nanmax(values)

    peaks[sid] = peak

# ------------------------------------------------------------------
# Example output
# ------------------------------------------------------------------
for sid, peak in list(peaks.items())[:10]:
    print(f"Structure {sid:5d} : peak = {peak:.3f}")
    
    
# ------------------------------------------------------------------
# Make a histogram
# ------------------------------------------------------------------
peak_values = np.array(list(peaks.values()), dtype=float)

# Remove NaNs / non-finite values
peak_values = peak_values[np.isfinite(peak_values)]

print(f"Number of valid peaks: {peak_values.size}")

# ------------------------------------------------------------------
# Plot histogram
# ------------------------------------------------------------------
plt.figure()
plt.hist(peak_values, bins=50)
plt.xlabel("Peak value (SNR)")
plt.xlabel('Peak Ratio within cloud')
plt.ylabel("Number of structures")
plt.title("Histogram of structure peak values")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Save to csv
# ------------------------------------------------------------------
#out_csv = "FGN_027_parallel_peaks.csv"
out_csv = "Mopra_320_parallel_peaks.csv"

#with open(out_csv, "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow(["structure_id", "peak_value"])

#    for sid, peak in peaks.items():
#        writer.writerow([sid, peak])

#print(f"Saved peaks to {out_csv}")