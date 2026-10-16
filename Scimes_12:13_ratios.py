#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:22:53 2026

@author: kyriakos.trakakis
"""

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
from astropy.table import Table
import pandas as pd

path_mask  = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/'
path_12    = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/'
path_13    = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/'
region     = "037_047"
dataset    = "FGN"

if dataset == "CHIMPS":
    dataset_12 = "COHRS"
else:
    dataset_12 = dataset

# ------------------------------------------------------------------
# File names
# ------------------------------------------------------------------
mask_file       = path_mask + f"{dataset}_{region}_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters.fits"
data_file_12    = path_12 + f"{dataset_12}_{region}_12CO_23arcsec_aligned.fits"
data_file_13    = path_13 + f"{dataset}_{region}_13CO_23arcsec_aligned.fits"
table_file      = path_mask + f"{dataset}_13CO_{region}_vrms_3_X_factor_masses_SNR0.csv"

# ------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------
with fits.open(mask_file, memmap=True) as hdul:
    mask_data = hdul[0].data.astype(np.int32)

with fits.open(data_file_12, memmap=True) as hdul:
    data_12 = hdul[0].data.astype(np.float64)
    
with fits.open(data_file_13, memmap=True) as hdul:
    data_13 = hdul[0].data.astype(np.float64)
    
df = pd.read_csv(table_file)
df=df[~np.isnan(df["_idx"])]

# ------------------------------------------------------------------
# Basic sanity checks
# ------------------------------------------------------------------
if mask_data.shape != data_12.shape:
    raise ValueError("Mask cube and 12CO data cube have different shapes!")
    
if mask_data.shape != data_13.shape:
    raise ValueError("Mask cube and 13CO data cube have different shapes!")

# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------
cluster_numbers = df['_idx']
sum_13 = []
sum_12 = []
ratio = []

for num in cluster_numbers:
    mask = (mask_data == num)
    if num == 0:
        # Replace cluster 0 voxels with 1, keep background (-1) untouched
        new_data = np.where(mask, 1, 0)
    else:
        new_data = np.where(mask, mask_data, 0)
        
    data_13_masked = data_13[mask]
    data_13_good = data_13_masked[~np.isnan(data_13_masked)]
    data_13_good = data_13_good[data_13_good >= 0]
    flux_13 = np.sum(data_13_good)
    
    
    
    data_12_masked = data_12[mask]
    data_12_good = data_12_masked[~np.isnan(data_12_masked)]
    data_12_good = data_12_good[data_12_good >= 0]
    flux_12 = np.sum(data_12_good)
    
    ratio_cloud = flux_12 / flux_13
    
    sum_13.append(flux_13)
    sum_12.append(flux_12)
    ratio.append(ratio_cloud)
    
    print(F"\n Cloud {num} has 13CO flux = {flux_13}, 12CO flux = {flux_12}, ratio = {ratio_cloud} ")

df['Sum_13'] = sum_13
df['Sum_12'] = sum_12
df['Sum_13'] = df['Sum_13']/2
df['Sum_12'] = df['Sum_12']/2
df['Ratio']  = ratio

df = df[["_idx", "Sum_13", "Sum_12", "Ratio"]]

# ------------------------------------------------------------------
# Plot histogram
# ------------------------------------------------------------------
plt.figure()
plt.hist(df['Ratio'], bins=50)
plt.xlabel("Peak value (SNR)")
plt.ylabel("Number of structures")
plt.title("Histogram of structure peak values")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Save to csv
# ------------------------------------------------------------------
out_csv =  path_mask + f"{dataset}_1-0_{region}_X-factor_ratios.csv"

df.to_csv(out_csv, index = False)

#    for sid, peak in peaks.items():
#        writer.writerow([sid, peak])

#print(f"Saved peaks to {out_csv}")