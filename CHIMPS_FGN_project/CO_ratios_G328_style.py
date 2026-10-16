#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:30:08 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import math
from scipy.stats import anderson
import scipy.stats as stats
from scipy.stats import norm

path_13 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
path_12 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/"
region  = "037_047"

if region == "017_027":
    sigma_13 = 0.51
    sigma_12 = 0.91
elif region == "027_037":
    sigma_13 = 0.54
    sigma_12 = 1.24
else:
    sigma_13 = 0.26
    sigma_12 = 0.66    

file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"
SNR_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned_SNR.fits"
SNR_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned_SNR.fits"

with fits.open(file_13) as hdul:
    data_13        = hdul[0].data
    data_13_header = hdul[0].header

with fits.open(file_12) as hdul:
    data_12        = hdul[0].data
    data_12_header = hdul[0].header
    
with fits.open(SNR_13) as hdul:
    snr_13 = hdul[0].data
    
with fits.open(SNR_12) as hdul:
    snr_12 = hdul[0].data
    
    
bad = (snr_12 > 1) & (data_12 < 0)
print(np.sum(bad))

if np.any(bad):
    idx_12 = np.argwhere(bad)[0]
    print("First bad voxel (12CO):", tuple(idx_12))
    print("data_12 =", data_12[tuple(idx_12)])
    print("snr_12  =", snr_12[tuple(idx_12)])
    
bad = (snr_13 > 1) & (data_13 < 0)
data_13[bad] = -data_13[bad]
bad = (snr_13 > 1) & (data_13 < 0)
print(np.sum(bad))

if np.any(bad):
    idx_13 = np.argwhere(bad)[0]
    print("First bad voxel (13CO):", tuple(idx_13))
    print("data_13 =", data_13[tuple(idx_13)])
    print("snr_13  =", snr_13[tuple(idx_13)])

    
## SNR = 1 limit ##
#-----------------------------------#
mask_1 = (snr_12 > 1) & (snr_13 > 1)

data_13_1 = np.where(mask_1, data_13, 0)
data_12_1 = np.where(mask_1, data_12, 0)

print(np.shape(data_12_1))
print(np.shape(data_13_1))

mask_nan_1 = (~np.isnan(data_13_1)) & (~np.isnan(data_12_1)) & (data_13_1!=0) & (data_12_1!=0)

data_13_1 = data_13_1[mask_nan_1]
data_12_1 = data_12_1[mask_nan_1]
#snr = snr[mask_nan]

print(np.shape(data_12_1))
print(np.shape(data_13_1))

ratio_1 = data_12_1 / data_13_1

min_ratio_1 = np.log10(ratio_1.min())
max_ratio = np.log10(20)
#max_ratio = np.log10(ratio.max())

n_bins_1 = math.ceil(2 * len(ratio_1) ** 0.4)

log_bin_edges_1 = np.linspace(min_ratio_1, max_ratio, n_bins_1 + 1)

linear_bin_edges_1 = 10**log_bin_edges_1

counts_1, _ = np.histogram(ratio_1, bins=linear_bin_edges_1)

linear_bin_widths_1 = linear_bin_edges_1[1:] - linear_bin_edges_1[:-1]

bin_centers_log_1 = 0.5 * (log_bin_edges_1[:-1] + log_bin_edges_1[1:])
bin_centers_ratio_1 = 10 ** bin_centers_log_1

## SNR = 3 limit ##
#-----------------------------------#

mask_3 = (snr_12 > 3) & (snr_13 > 3)

data_13_3 = np.where(mask_3, data_13, 0)
data_12_3 = np.where(mask_3, data_12, 0)

print(np.shape(data_12_3))
print(np.shape(data_13_3))

mask_nan_3 = (~np.isnan(data_13_3)) & (~np.isnan(data_12_3)) & (data_13_3!=0) & (data_12_3!=0)

data_13_3 = data_13_3[mask_nan_3]
data_12_3 = data_12_3[mask_nan_3]
#snr = snr[mask_nan]

print(np.shape(data_12_3))
print(np.shape(data_13_3))

ratio_3 = data_12_3 / data_13_3

min_ratio_3 = np.log10(ratio_3.min())
max_ratio = np.log10(20)
#max_ratio = np.log10(ratio.max())

n_bins_3 = math.ceil(2 * len(ratio_3) ** 0.4)

log_bin_edges_3 = np.linspace(min_ratio_3, max_ratio, n_bins_3 + 1)

linear_bin_edges_3 = 10**log_bin_edges_3

counts_3, _ = np.histogram(ratio_3, bins=linear_bin_edges_3)

linear_bin_widths_3 = linear_bin_edges_3[1:] - linear_bin_edges_3[:-1]

bin_centers_log_3 = 0.5 * (log_bin_edges_3[:-1] + log_bin_edges_3[1:])
bin_centers_ratio_3 = 10 ** bin_centers_log_3

plt.figure(figsize = (8,6))
plt.scatter(bin_centers_ratio_1, counts_1/counts_1.max(), s = 0.1, label = "S/N > 1")
plt.scatter(bin_centers_ratio_3, counts_3/counts_3.max(), s = 0.1, color = 'k', label = "S/N > 3")
plt.xlabel('Ratio')
plt.ylabel('Normalised number')
plt.title(rf"$^{{12}}\mathrm{{CO}} \,/\, ^{{13}}\mathrm{{CO}}\;\; {region.replace('_', '-')}$")
plt.legend()

peak_position_1 = counts_1.argmax()
peak_1 = bin_centers_ratio_1[peak_position_1]

peak_position_3 = counts_3.argmax()
peak_3 = bin_centers_ratio_3[peak_position_3]

print(f"\n Ratio peaks at  {peak_1} for S/N > 1 and {peak_3} for S/N > 3")
print(f"\n {region}")

bins = 500  # adjust resolution, this is ususally 500 for 17-27 and 37-47 but 2500 for 27-37. Not sure why

H, xedges, yedges = np.histogram2d(
    data_13_1,
    ratio_1,
    bins=bins
)

H_log = np.log10(H + 1)

# Mask empty bins
H_masked = np.ma.masked_where(H == 0, H_log)

# Copy colormap and set masked (bad) values to white
cmap = plt.cm.gray.copy()
cmap.set_bad(color='white')

plt.figure(figsize = (8,6))
plt.imshow(
    H_masked.T,
    origin='lower',
    aspect='auto',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap=cmap
)

plt.xlim([0,20])
plt.ylim([0,20])  # 10 if using 3, 20 if using 1
plt.ylabel("$^{12}CO \;/\;^{13}CO$")
plt.xlabel("$^{13}CO$")

X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

levels = np.linspace(H_log[H > 0].min(), H_log.max(), 10)  # Number of contours

plt.contour(
    X,
    Y,
    H_log.T,
    levels=levels,
    colors='black',
    linewidths=0.5
)

sdev = (sigma_12 / data_12_1 + sigma_13 / data_13_1) * ratio_1
sdev_3 = (sigma_12 / data_12_3 + sigma_13 / data_13_3) * ratio_3

# --- weighted linear fit ---
mask_fit = np.isfinite(sdev) & (sdev > 0)

m, b = np.polyfit(
    data_13_1[mask_fit],
    ratio_1[mask_fit],
    1,
    w=1/sdev[mask_fit]
)

# --- plot the fit line ---
x_fit = np.linspace(0, 20, 200)
y_fit = m * x_fit + b

plt.plot(
    x_fit,
    y_fit,
    color='red',
    linewidth=2,
    label=rf"Fit: $y = {m:.2f}x + {b:.2f}$"
)




## Working now

log_ratio_3 = np.log10(ratio_3[(ratio_3 > 0) & np.isfinite(ratio_3)])

mu, sigma = norm.fit(log_ratio_3)

s = sigma * np.log(10)
mean = 10**mu * np.exp(0.5 * s**2)

std = mean * np.sqrt(np.exp(s**2) - 1)

low  = 10**(mu - sigma)
high = 10**(mu + sigma)

mask_ratio_3_peak = ((ratio_3 > low) & (ratio_3 < high))

#mask_ratio_peak = mask_ratio_3_peak & np.isfinite(data_13_1) & np.isfinite(ratio_1) & (sdev > 0)  # This will fail, because
# ratio_3 has fewer points than ratio_1 and data_13_1. I also checked, sdev > 0 everywhere, ratio_1 is finite everywhere, so is data_13_1

n, c = np.polyfit(data_13_3[mask_ratio_3_peak], ratio_3[mask_ratio_3_peak], 1, w = 1 / sdev_3[mask_ratio_3_peak])
y_fit_2 = n*x_fit +c


plt.plot(
    x_fit,
    y_fit_2,
    color='blue',
    linewidth=2,
    label=rf"Fit: $y = {n:.2f}x + {c:.2f}$"
)
plt.legend()

print(f" \n 1σ stats for {region}: μ = {np.mean(ratio_1):.3f}, median = {np.median(ratio_1):.3f}, max = {np.max(ratio_1):.3f}, σ = {np.std(ratio_1):.3f}, peak = {peak_1:.3f}")
print(f" \n 3σ stats for {region}: μ = {np.mean(ratio_3):.3f}, median = {np.median(ratio_3):.3f}, max = {np.max(ratio_3):.3f}, σ = {np.std(ratio_3):.3f}, peak = {peak_3:.3f}")
