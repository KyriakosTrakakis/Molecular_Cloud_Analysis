#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:32:06 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

dataset = "FGN_out"
ratio_all = np.array([3.26])

if dataset == "FGN_CHIMPS":
    path_13 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
    path_12 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/"
    regions = ["017_027", "027_037", "037_047"]
   # regions = ["037_047"]
    
    for region in regions:
        data_file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
        data_file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"
        
        with fits.open(data_file_13) as hdul:
            data_13        = hdul[0].data
            data_13_header = hdul[0].header
        print(f"Region {region} 13CO loaded")

        with fits.open(data_file_12) as hdul:
            data_12        = hdul[0].data
            data_12_header = hdul[0].header 
        print(f"Region {region} 12CO loaded")
        
        SNR_file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned_SNR.fits"
        SNR_file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned_SNR.fits"
        
        with fits.open(SNR_file_13) as hdul:
            SNR_13        = hdul[0].data
        print(f"Region {region} 13CO SNR loaded")

        with fits.open(SNR_file_12) as hdul:
            SNR_12        = hdul[0].data
        print(f"Region {region} 12CO SNR loaded")
            
        mask = (SNR_13 > 2) & (SNR_12 > 2)
    
        masked_13 = np.where(mask, data_13, np.nan)
        masked_12 = np.where(mask, data_12, np.nan)
        print(f"Region {region} masked")

        collapsed_13 = np.nansum(masked_13, axis=0)
        collapsed_12 = np.nansum(masked_12, axis=0)
        print(f"Region {region} summed")

        ratio = np.divide(
                collapsed_12,
                collapsed_13,
                out=np.full_like(collapsed_13, np.nan),
                where=collapsed_13 > 0
            )
        print(f"Region {region} divided")
        
        r = ratio[np.isfinite(ratio)]

    # optional clipping of extreme outliers for nicer display
        r = r[(r > 0) & (r < np.percentile(r, 99.5))]

        ratio_all = np.concatenate((ratio_all, r), axis=0)


elif dataset == "Mopra":
    path_13 = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO/"
    path_12 = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/12CO/"
    regions = ["250-260", "260-270", "270-280", "280-290", "290-300", "300-310", "310-320", "320-330", "330-340", "340-350", "350-355"]
    noise_13 = [0.55,0.58, 0.59, 0.58, 0.59, 0.61, 0.52, 0.54, 0.65, 0.57, 0.59]
    noise_12 = [1.2, 1.39, 1.33, 1.24, 1.32, 1.31, 1.16, 1.2, 1.21, 1.19, 1.39]
    
    for i in range(len(regions)):
        region = regions[i]
        
        data_file_13 = path_13 + f"G{region}-13CO_smallercube_05res.fits"
        data_file_12 = path_12 + f"G{region}-12CO_smallercube_05res.fits"
        
        with fits.open(data_file_13) as hdul:
            data_13        = hdul[0].data
            data_13_header = hdul[0].header
        print(f"Region {region} 13CO loaded")

        with fits.open(data_file_12) as hdul:
            data_12        = hdul[0].data
            data_12_header = hdul[0].header 
        print(f"Region {region} 12CO loaded")

        sigma_13 = noise_13[i]
        sigma_12 = noise_12[i]
        
        mask = (data_13 > 2 * sigma_13) & (data_12 > 2 * sigma_12)

        masked_13 = np.where(mask, data_13, np.nan)
        masked_12 = np.where(mask, data_12, np.nan)
        print(f"Region {region} masked")

        collapsed_13 = np.nansum(masked_13, axis=0)
        collapsed_12 = np.nansum(masked_12, axis=0)
        print(f"Region {region} summed")

        ratio = np.divide(
                collapsed_12,
                collapsed_13,
                out=np.full_like(collapsed_13, np.nan),
                where=collapsed_13 > 0
            )
        print(f"Region {region} divided")
        
        r = ratio[np.isfinite(ratio)]

    # optional clipping of extreme outliers for nicer display
        r = r[(r > 0) & (r < np.percentile(r, 99.5))]

        ratio_all = np.concatenate((ratio_all, r), axis=0)
        
        
        
elif dataset == "FGN_in":
    path_13 = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/"
    path_12 = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/12CO/"
    regions = ["010_020", "020_030", "030_040", "040_050"]
    
    for region in regions:
        data_file_13 = path_13 + f"FGN_13CO_{region}_23arcsec_trimmed.fits"
        data_file_12 = path_12 + f"FGN_12CO_{region}_23arcsec_trimmed.fits"
        
        with fits.open(data_file_13) as hdul:
            data_13        = hdul[0].data
            data_13_header = hdul[0].header
        print(f"Region {region} 13CO loaded")

        with fits.open(data_file_12) as hdul:
            data_12        = hdul[0].data
            data_12_header = hdul[0].header 
        print(f"Region {region} 12CO loaded")
        
        SNR_file_13 = path_13 + f"FGN_13CO_{region}_23arcsec_SNR_spline.fits"
        SNR_file_12 = path_12 + f"FGN_12CO_{region}_23arcsec_SNR_spline.fits"
        
        with fits.open(SNR_file_13) as hdul:
            SNR_13        = hdul[0].data
        print(f"Region {region} 13CO SNR loaded")

        with fits.open(SNR_file_12) as hdul:
            SNR_12        = hdul[0].data
        print(f"Region {region} 12CO SNR loaded")
            
        mask = (SNR_13 > 2) & (SNR_12 > 2)
    
        masked_13 = np.where(mask, data_13, np.nan)
        masked_12 = np.where(mask, data_12, np.nan)
        print(f"Region {region} masked")

        collapsed_13 = np.nansum(masked_13, axis=0)
        collapsed_12 = np.nansum(masked_12, axis=0)
        print(f"Region {region} summed")

        ratio = np.divide(
                collapsed_12,
                collapsed_13,
                out=np.full_like(collapsed_13, np.nan),
                where=collapsed_13 > 0
            )
        print(f"Region {region} divided")
        
        r = ratio[np.isfinite(ratio)]

    # optional clipping of extreme outliers for nicer display
        r = r[(r > 0) & (r < np.percentile(r, 99.5))]

        ratio_all = np.concatenate((ratio_all, r), axis=0)



hist_counts_r, bin_edges_r = np.histogram(ratio_all, bins=60)

hist_norm_unit = hist_counts_r / np.max(hist_counts_r)

hist_norm_area = hist_counts_r / np.nansum(hist_counts_r)

bin_centers_r = 0.5 * (bin_edges_r[:-1] + bin_edges_r[1:])



    # -------------------------------------------------
    # Fit left side only
    # -------------------------------------------------

hist_vals, bin_edges = np.histogram(ratio_all, bins=60)

mode_bin = np.argmax(hist_vals)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

r_mode = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])

r_left = ratio_all[ratio_all <= r_mode]

sigma = np.sqrt(np.mean((r_left - r_mode)**2))

x_fit = np.linspace(ratio_all.min(), ratio_all.max(), 500)
    
bin_width = bin_edges[1] - bin_edges[0]

# scale to histogram peak
amplitude = hist_vals[mode_bin]

y_fit = amplitude * np.exp(-(x_fit - r_mode)**2 / (2 * sigma**2))


plt.figure(figsize=(9,6))
plt.bar(
    bin_centers_r,
    hist_norm_unit,
    width=np.diff(bin_edges),
    align='center',
    color='royalblue',
    edgecolor='white',
    alpha=0.85
    )

plt.plot(
    x_fit,
    y_fit/ amplitude,
    color="black",
    lw=3,
    label=rf"Gaussian fit: $\mu={r_mode:.2f}$, $\sigma={sigma:.2f}$"
    )


plt.axvline(np.nanmedian(ratio_all), color="red", ls="--", lw=2,
            label=f"Median = {np.nanmedian(ratio_all):.2f}")

plt.axvline(np.nanmean(ratio_all), color="orange", ls="--", lw=2,
            label=f"Mean = {np.nanmean(ratio_all):.2f}")

plt.xlabel("12 / 13 Ratio")
plt.ylabel("Number of Pixels normalised to mode")
plt.title(f"All {dataset} data ratio Distribution for S/N > 2")
plt.legend()
plt.xlim([0, ratio_all.max()+0.25])
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()    


# Note: for FGN this gives 2.96 and not 3.26. Turns out 3.26 is just for 37-47 and the true value is 2.96.
# I an not changing everything again for a 10% change, take this to your grave.