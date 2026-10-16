#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:42:29 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import hsv_to_rgb
from scipy.stats import norm
from statistics import mode

# =====================================================
# USER SETTINGS
# =====================================================
#dataset = "Mopra"  
dataset = "FGN" 

#v_range = [-60, -50] # Mopra has negative velocities
v_range = [80,90] # FGN has positive velocities  # Original code, uncomment when done
v_noise = (0, 40)   # This one doesn't need change

region = "027_037"       # FGN: 017_027 etc
#region = "330-340"      # Mopra example

# choose any two:
isotopologues = ["12", "13"]   # Thicker one first!


# =====================================================
# PATHS
# =====================================================
FGN_paths = {
    "12": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/",
    "13": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/",
    "18": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/C18O/",
}

Mopra_paths = {
    "12": "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/",
    "13": "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/",
    "18": "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/",
}



# =====================================================
# HELPERS
# =====================================================

def iso_label(iso):
    if iso == "18":
        return r"C$^{18}$O"
    return rf"$^{{{iso}}}$CO"


def get_paths(dataset):
    if dataset.upper() == "FGN":
        return FGN_paths
    return Mopra_paths


def cube_filename(dataset, region, iso):

    if dataset.upper() == "FGN":

        if iso == "18":
            return f"FGN_{region}_C18O_23arcsec_aligned.fits"
        return f"FGN_{region}_{iso}CO_23arcsec_aligned.fits"

    else:

        if iso == "18":
            return f"G{region}-C18O_smallercube_05res.fits"
        return f"G{region}-{iso}CO_smallercube_05res.fits"


def snr_filename(region, iso):
    if iso == "18":
        return f"FGN_{region}_C18O_23arcsec_aligned_SNR.fits"
    else:
        return f"FGN_{region}_{iso}CO_23arcsec_aligned_SNR.fits"
    
    
def calculate_cube_noise(fits_file, v_range):
    """
    Calculate the standard deviation of pixel values within the specified range.

    Parameters:
        fits_file (str): Path to the FITS file.
        l_range (tuple): Range of l-coordinates as (l_min, l_max).
        b_range (tuple): Range of b-coordinates as (b_min, b_max).

    Returns:
        float: Standard deviation of the selected pixel values.
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Assuming the data is in the primary HDU
        data = hdul[0].data

    # Check if the data is 3D
    if data.ndim != 3:
        raise ValueError("The FITS file does not contain 3D data.")

    # Extract the specified range
    v_min, v_max = v_range
    selected_data = data[v_min:v_max+1,:,:]

    # Flatten the selected data and calculate the standard deviation
    std_dev = np.nanstd(selected_data)

    return std_dev
    

# =====================================================
# SETUP
# =====================================================

paths = get_paths(dataset)

iso1, iso2 = isotopologues

file_ref = paths[iso1] + cube_filename(dataset, region, iso1)

with fits.open(file_ref) as hdul:
    header = hdul[0].header
    cube_ref = hdul[0].data
    
nchan = cube_ref.shape[0]

crval3 = header["CRVAL3"]
crpix3 = header["CRPIX3"]
cdelt3 = header["CDELT3"]
    
channels = np.arange(nchan)
vel = crval3 + (channels + 1 - crpix3) * cdelt3

if abs(cdelt3) > 100:
    vel = vel / 1000.0
    
#v_range = [np.nanmin(vel), np.nanmax(vel)]   # Added code, delete when done

#print(f"Using full velocity range: {v_range[0]:.1f} to {v_range[1]:.1f} km/s")  ## Added code, delete when done
    
data = {}
file_cube = {}
mask_snr = {}

for iso in ["12", "13", "18"]:

    file_cube[iso] = paths[iso] + cube_filename(dataset, region, iso)
    data[iso] = fits.getdata(file_cube[iso])
    
mask_v = (vel >= v_range[0]) & (vel < v_range[1])

data1 = data[iso1][mask_v,:,:]
data2 = data[iso2][mask_v,:,:]

if dataset == "FGN":
    
    snr1 = fits.getdata(paths[iso1] + snr_filename(region, iso1))[mask_v,:,:]
    snr2 = fits.getdata(paths[iso2] + snr_filename(region, iso2))[mask_v,:,:]
else:
    sigma_1 = calculate_cube_noise(file_cube[iso1], v_noise)
    sigma_2 = calculate_cube_noise(file_cube[iso2], v_noise)
    
    snr1 = data1 / sigma_1
    snr2 = data2 / sigma_2
    

def plot_ratio_histograms_thresh(threshold):
    

    shared_mask = (snr1 > threshold) & (snr2 > threshold)

    masked1 = np.where(shared_mask, data1, np.nan)
    masked2 = np.where(shared_mask, data2, np.nan)

    collapsed1 = np.nansum(masked1, axis=0)
    collapsed2 = np.nansum(masked2, axis=0)

    ratio = np.divide(
        collapsed1,
        collapsed2,
        out=np.full_like(collapsed1, np.nan),
        where=collapsed2 > 0
    )

# ratio map already created earlier:
# ratio = collapsed[iso1] / collapsed[iso2]

# flatten and remove bad values
    r = ratio[np.isfinite(ratio)]

# optional clipping of extreme outliers for nicer display
    r = r[(r > 0) & (r < np.percentile(r, 99.5))]

# -------------------------------------------------
# Original histogram
# -------------------------------------------------

    hist_counts_r, bin_edges_r = np.histogram(r, bins=60)

    hist_norm_unit = hist_counts_r / np.max(hist_counts_r)

    hist_norm_area = hist_counts_r / np.nansum(hist_counts_r)

    bin_centers_r = 0.5 * (bin_edges_r[:-1] + bin_edges_r[1:])

    plt.figure(figsize=(9,6))


# -------------------------------------------------
# Fit left side only
# -------------------------------------------------

    hist_vals, bin_edges = np.histogram(r, bins=60)

    mode_bin = np.argmax(hist_vals)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    r_mode = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])

    r_left = r[r <= r_mode]

    sigma = np.sqrt(np.mean((r_left - r_mode)**2))

    x_fit = np.linspace(r.min(), r.max(), 500)
    
    modes.append(r_mode)


# -------------------------------------------------
# Find the excess
# -------------------------------------------------

    bin_width = bin_edges[1] - bin_edges[0]

# scale to histogram peak
    amplitude = hist_vals[mode_bin]

    y_fit = amplitude * np.exp(
    -(x_fit - r_mode)**2 / (2 * sigma**2)
)

## Adding this to subtract and find the excess
    gaussian_counts = amplitude * np.exp(
    -(bin_centers - r_mode)**2 / (2 * sigma**2)
)

    excess = hist_vals - gaussian_counts


# -------------------------------------------------
# Plot 1: Normalised to peak
# -------------------------------------------------

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


    plt.axvline(np.nanmedian(r), color="red", ls="--", lw=2,
            label=f"Median = {np.nanmedian(r):.2f}")

    plt.axvline(np.nanmean(r), color="orange", ls="--", lw=2,
            label=f"Mean = {np.nanmean(r):.2f}")

    plt.xlabel(f"{iso_label(iso1)} / {iso_label(iso2)} Ratio")
    plt.ylabel("Number of Pixels normalised to mode")
    plt.title(f"{region} Ratio Distribution ({v_range[0]}–{v_range[1]} km/s) for S/N >{threshold}")
    plt.legend()
    plt.xlim([0, r.max()+0.25])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    
    means.append(np.nanmean(r))
    medians.append(np.nanmedian(r))
    
# -------------------------------------------------
# Plot 2: Excess normalised to peak
# -------------------------------------------------



    plt.figure(figsize=(9,5))

    plt.plot(
    bin_centers,
    excess / amplitude,
    drawstyle='steps-mid',
    lw=2
)

    plt.axhline(0, color='black', ls='--')

    plt.xlabel("Ratio")
    plt.ylabel("Observed - Gaussian")

    plt.title(f"{region} Non-Gaussian Excess ({v_range[0]}–{v_range[1]} km/s) for S/N >{threshold} (peak normalised)")
    plt.axvline(np.nanmedian(r), color="red", ls="--", lw=2,
            label=f"Median = {np.nanmedian(r):.2f}")

    plt.axvline(np.nanmean(r), color="orange", ls="--", lw=2,
            label=f"Mean = {np.nanmean(r):.2f}")
    plt.legend()
    plt.xlim([0, r.max()+0.25])

    plt.grid(alpha=0.3)

    plt.show()



# -------------------------------------------------
# Plot 3: Normalised to area under curve
# -------------------------------------------------


    plt.figure(figsize=(9,6))
    plt.bar(
    bin_centers_r,
    hist_norm_area,
    width=np.diff(bin_edges),
    align='center',
    color='royalblue',
    edgecolor='white',
    alpha=0.85
)

    plt.plot(
    x_fit,
    y_fit/ np.nansum(hist_counts_r),
    color="black",
    lw=3,
    label=rf"Gaussian fit: $\mu={r_mode:.2f}$, $\sigma={sigma:.2f}$"
)


    plt.axvline(np.nanmedian(r), color="red", ls="--", lw=2,
            label=f"Median = {np.nanmedian(r):.2f}")

    plt.axvline(np.nanmean(r), color="orange", ls="--", lw=2,
            label=f"Mean = {np.nanmean(r):.2f}")

    plt.xlabel(f"{iso_label(iso1)} / {iso_label(iso2)} Ratio")
    plt.ylabel("Normalised Pixel Density")
    plt.title(f"{region} Ratio Distribution ({v_range[0]}–{v_range[1]} km/s) for S/N >{threshold}")
    plt.legend()
    plt.xlim([0, r.max()+0.25])
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Plot 4: Excess normalised to area
# -------------------------------------------------



    plt.figure(figsize=(9,5))

    plt.plot(
    bin_centers,
    excess / np.nansum(hist_counts_r),
    drawstyle='steps-mid',
    lw=2
)

    plt.axhline(0, color='black', ls='--')

    plt.xlabel("Ratio")
    plt.ylabel("Observed - Gaussian")

    plt.title(f"{region} Non-Gaussian Excess ({v_range[0]}–{v_range[1]} km/s) for S/N >{threshold} (area normalised)")
    plt.axvline(np.nanmedian(r), color="red", ls="--", lw=2,
            label=f"Median = {np.nanmedian(r):.2f}")

    plt.axvline(np.nanmean(r), color="orange", ls="--", lw=2,
            label=f"Mean = {np.nanmean(r):.2f}")
    plt.legend()
    plt.xlim([0, r.max()+0.25])

    plt.grid(alpha=0.3)

    plt.show()

    


thresholds = np.linspace(0.5,10,20)  # Original code, uncomment when done
#thresholds = 2     # Added code, delete when done
means = []
medians = []
modes = []
for thresh in thresholds:   # Original code, uncomment when done
    plot_ratio_histograms_thresh(thresh)   # Original code, uncomment when done
#plot_ratio_histograms_thresh(2)  # Added code, delete when done

    
means = np.array(means)
medians = np.array(medians)
modes = np.array(modes)

mins = np.array([np.min(means), np.min(medians), np.min(modes)])
maxs = np.array([np.max(means), np.max(medians), np.max(modes)])
ymin = np.min(mins) - 0.5
ymax = np.max(maxs) + 0.5

region_str =region.replace("_","-")


plt.scatter(thresholds, means, label = 'Mean ratio', color = 'red')
plt.scatter(thresholds, medians, label = "Median ratio", color = 'blue')
plt.scatter(thresholds, modes, label = "Mode ratio", color = 'black')
plt.ylim([0,ymax])
plt.xlim([0,11])
plt.title(f"l={region_str} ({v_range[0]}–{v_range[1]} km/s) Ratio trend for different levels of S/N")
plt.xlabel('S/N clipping level')
plt.ylabel('Ratio value')
plt.legend()
plt.show()