#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evans Plot Universal Version
Supports:

1) FGN dataset
2) Mopra dataset

Choose dataset, region, isotopologues at top.

Author: Kyriakos Trakakis
Created on Thu May  7 10:03:17 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import hsv_to_rgb

# =====================================================
# USER SETTINGS
# =====================================================

dataset = "FGN"          # "FGN" or "Mopra"

region = "027_037"       # FGN: 017_027 etc
#region = "330-340"      # Mopra example
v_range_Mopra = -137, 15

isotopologues = ["12", "13"]   # thicker tracer first

save_plots = False
noise_mode = "white"     # "white" or "grey"

# Ratio scale
rmin = 1
rmax = 15

# SNR scale
snrmax = 10

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


def get_region_parameters(file_dict, v_noise):

    # ---------------------------
    # velocity limits unchanged
    # ---------------------------

    if dataset.upper() == "FGN":

        if region == "017_027":
            vlow, vhigh = 0, 130

        elif region == "027_037":
            vlow, vhigh = 10, 120

        else:
            vlow, vhigh = -10, 100

    else:
        vlow, vhigh = v_range_Mopra

    # ---------------------------
    # sigma measured from cubes
    # ---------------------------

    sigma = {}

    for iso in ["12", "13", "18"]:
        sigma[iso] = calculate_cube_noise(file_dict[iso], v_noise)
        
    print("Measured RMS:")
    for iso in sigma:
        print(iso, sigma[iso])

    return sigma, vlow, vhigh

v_noise = (0, 40)


# =====================================================
# SETUP
# =====================================================

paths = get_paths(dataset)

iso1, iso2 = isotopologues

# output folder
outdir = f"{get_paths(dataset)['13']}Evans_plots/"
os.makedirs(outdir, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

data = {}
file_cube = {}

for iso in ["12", "13", "18"]:

    file_cube[iso] = paths[iso] + cube_filename(dataset, region, iso)
    data[iso] = fits.getdata(file_cube[iso])
    




sigma, vlow, vhigh = get_region_parameters(file_cube, v_noise)

# use first isotope as reference WCS/header
file_ref = paths[iso1] + cube_filename(dataset, region, iso1)

with fits.open(file_ref) as hdul:
    header = hdul[0].header
    cube_ref = hdul[0].data

# =====================================================
# VELOCITY AXIS
# =====================================================

nchan = cube_ref.shape[0]

crval3 = header["CRVAL3"]
crpix3 = header["CRPIX3"]
cdelt3 = header["CDELT3"]

channels = np.arange(nchan)
vel = crval3 + (channels + 1 - crpix3) * cdelt3

if abs(cdelt3) > 100:
    vel = vel / 1000.0

# =====================================================
# MAIN LOOP
# =====================================================
vlow, vhigh = 70,110  # Only use this for JCMT CFE maps
bins = np.arange(vlow, vhigh + 10, 40)
dust_file = "/Users/kyriakos.trakakis/Downloads/Other_Stuff/cfe_l30_scutum.fits"
#dust_file = f"/Users/kyriakos.trakakis/Downloads/ATLASGAL_{region}.fits"
dust = fits.getdata(dust_file)
dust = dust/np.nanmax(dust)
dust_header = fits.getheader(dust_file)
dust_wcs = WCS(dust_header).celestial

for i in range(len(bins) - 1):

    v0 = bins[i]
    v1 = bins[i + 1]

    mask = (vel >= v0) & (vel < v1)

    cube1 = data[iso1][mask, :, :]
    cube2 = data[iso2][mask, :, :]


    # -----------------------------------------
    # Integrated intensities
    # -----------------------------------------

    I1 = np.nansum(cube1, axis=0)
    I2 = np.nansum(cube2, axis=0)

    levels = ([0.015, 0.036, 0.0525, 0.075, 0.15, 0.225, 0.55])
    # -----------------------------------------
    # Ratio
    # -----------------------------------------

    ratio = np.full_like(I1, np.nan, dtype=float)

    good = I2 > 0
    ratio[good] = I1[good] / I2[good]

    # -----------------------------------------
    # SNR estimate
    # -----------------------------------------

    sigma1_eff = sigma[iso1] * np.sqrt(nchan)
    sigma2_eff = sigma[iso2] * np.sqrt(nchan)

    a = sigma1_eff / I1
    b = sigma2_eff / I2

    snr = 1.0 / (a**2 + b**2)

    # -----------------------------------------
    # HSV conversion
    # -----------------------------------------

    ratio_clip = np.clip(ratio, rmin, rmax)

   # H = (ratio_clip - rmin) / (rmax - rmin)
    scaled = (ratio_clip - rmin)/(rmax-rmin)
    H = 0.8 * scaled

    if noise_mode.lower() == "grey":

        S = np.clip(snr, 0, snrmax) / snrmax
        V = 0.5 + 0.5 * S

    else:

        S = np.clip(snr, 1, snrmax)
        S = (S - 1) / (snrmax - 1)
        V = np.ones_like(H)

    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)

    RGB[np.isnan(ratio)] = 1

    # -----------------------------------------
    # Plot
    # -----------------------------------------

    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection=WCS(header).celestial)
    ax.contour(dust, levels=levels, colors='black',linewidths = 0.1,transform=ax.get_transform(dust_wcs))

    ax.imshow(RGB, origin="lower")

    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")

    ax.set_title(
        f"{iso_label(iso1)} / {iso_label(iso2)}   "
        f"{v0} to {v1} km/s"
    )

    plt.tight_layout()

    if save_plots:

        reg = region.replace("_", "-")

      #  fname = (
       #     f"{outdir}/"
       #     f"Evans_plot_{dataset}_{iso1}v{iso2}_"
       #     f"l{reg}_v{v0}-{v1}kms.png"
       # )   ## No dust maps
       
        fname = (
            f"/Users/kyriakos.trakakis/Downloads/Ratios_Test/Evans_plots_on_dust/Scutum_l30_{iso1}v{iso2}"
            f"v{v0}-{v1}kms.png"
            )  ## JCMT dust maps
       
      #  fname = (
      #      f"/Users/kyriakos.trakakis/Downloads/Ratios_Test/Evans_plots_on_dust/ATLASGAL_{iso1}v{iso2}_"
      #      f"l{region}_v{v0}-{v1}kms.png"
      #      )  ## JCMT dust maps

      #  plt.savefig(fname, dpi=600, bbox_inches="tight")

    plt.show()

# =====================================================
# LEGEND
# =====================================================

nx, ny = 300, 300

ratio_vals = np.linspace(rmin, rmax, nx)

if noise_mode.lower() == "grey":
    snr_vals = np.linspace(0, snrmax, ny)
else:
    snr_vals = np.linspace(1, snrmax, ny)

RR, SS = np.meshgrid(ratio_vals, snr_vals)

scaled_leg = (RR - rmin) / (rmax - rmin)
H_leg = 0.8 * scaled_leg

if noise_mode.lower() == "grey":

    S_leg = SS / snrmax
    V_leg = 0.5 + 0.5 * S_leg
    ymin = 0

else:

    S_leg = (SS - 1) / (snrmax - 1)
    V_leg = np.ones_like(H_leg)
    ymin = 1

HSV_leg = np.dstack((H_leg, S_leg, V_leg))
RGB_leg = hsv_to_rgb(HSV_leg)

plt.figure(figsize=(8, 6))

plt.imshow(
    RGB_leg,
    origin="lower",
    aspect="auto",
    extent=[rmin, rmax, ymin, snrmax]
)

plt.xlabel(f"{iso_label(iso1)} / {iso_label(iso2)} Ratio")
plt.ylabel("S/N")
plt.title("Evans Plot Colour Legend")

plt.tight_layout()

#if save_plots:
  #  plt.savefig(f"{outdir}/Evans_plot_colour_legend.png", dpi=300, bbox_inches="tight")
  

plt.show()