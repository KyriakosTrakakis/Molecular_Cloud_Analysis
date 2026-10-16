#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:13:40 2026

@author: kyriakos.trakakis
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Evans Plot Script
Choose any two isotopologues at the top:

["12","13"]
["13","18"]
["12","18"]

Author: Kyriakos Trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import hsv_to_rgb

# =====================================================
# USER SETTINGS
# =====================================================

region = "027_037"

# choose any two:
isotopologues = ["12", "13"]   # Thicker one first!

rmin = 1
rmax = 15
snrmax = 10

# =====================================================
# PATHS
# =====================================================

paths = {
    "12": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/",
    "13": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/",
    "18": "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/C18O/",
}

# =====================================================
# REGION NOISE VALUES
# =====================================================

if region == "017_027":
    sigma = {"12": 0.91, "13": 0.51, "18": 0.52}
    vlow, vhigh = 0, 130

elif region == "027_037":
    sigma = {"12": 1.24, "13": 0.54, "18": 0.59}
    vlow, vhigh = 10, 120

else:
    sigma = {"12": 0.66, "13": 0.26, "18": 0.31}
    vlow, vhigh = -10, 100

# =====================================================
# HELPERS
# =====================================================

def iso_label(iso):
    if iso == "18":
        return r"C$^{18}$O"
    else:
        return rf"$^{{{iso}}}$CO"


def cube_filename(region, iso):
    if iso == "18":
        return f"FGN_{region}_C18O_23arcsec_aligned.fits"
    else:
        return f"FGN_{region}_{iso}CO_23arcsec_aligned.fits"


def snr_filename(region, iso):
    if iso == "18":
        return f"FGN_{region}_C18O_23arcsec_aligned_SNR.fits"
    else:
        return f"FGN_{region}_{iso}CO_23arcsec_aligned_SNR.fits"


# =====================================================
# LOAD DATA
# =====================================================

data = {}
snr_cube = {}

for iso in ["12", "13", "18"]:

    file_cube = paths[iso] + cube_filename(region, iso)
    file_snr  = paths[iso] + snr_filename(region, iso)

    data[iso] = fits.getdata(file_cube)
    snr_cube[iso] = fits.getdata(file_snr)

# Use first selected tracer for header/WCS
iso_ref = isotopologues[0]
file_ref = paths[iso_ref] + cube_filename(region, iso_ref)

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
# SELECT ACTIVE TRACERS
# =====================================================

iso1, iso2 = isotopologues

# =====================================================
# LOOP THROUGH VELOCITY BLOCKS
# =====================================================

bins = np.arange(vlow, vhigh + 10, 10)

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

    H = (ratio_clip - rmin) / (rmax - rmin)

    # Original white noise
    S = np.clip(snr, 1, snrmax)
    S = (S-1)/(snrmax-1)

    V = np.ones_like(H)

    # grey at SNR=0, bright at high SNR
    #S = np.clip(snr, 0, snrmax) / snrmax
    #V = 0.5 + 0.5 * S

    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)

    RGB[np.isnan(ratio)] = 1

    # -----------------------------------------
    # Plot
    # -----------------------------------------

    wcs2d = WCS(header).celestial

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection=wcs2d)

    ax.imshow(RGB, origin="lower")

    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")

    ax.set_title(
        f"{iso_label(iso1)} / {iso_label(iso2)}   "
        f"{v0} to {v1} km/s"
    )
    region_mod = region.replace("_","-")
    plt.tight_layout()
    filename = (
    f"Evans_plots/"
    f"Evans_plot_{iso1}v{iso2}_l{region_mod}_v{v0}-{v1}kms.png"
)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 2D LEGEND
# =====================================================

nx, ny = 300, 300

ratio_vals = np.linspace(rmin, rmax, nx)
snr_vals   = np.linspace(1, snrmax, ny)  # Can also do 0,snrmax

RR, SS = np.meshgrid(ratio_vals, snr_vals)

H_leg = (RR - rmin) / (rmax - rmin)
#S_leg = SS / snrmax  # For gray
S_leg = (SS - 1) / 9.0
#V_leg = 0.5 + 0.5 * S_leg # For gray
V_leg = np.ones_like(H_leg)

HSV_leg = np.dstack((H_leg, S_leg, V_leg))
RGB_leg = hsv_to_rgb(HSV_leg)

plt.figure(figsize=(8, 6))

plt.imshow(
    RGB_leg,
    origin="lower",
    aspect="auto",
    extent=[rmin, rmax, 1, snrmax]  # Can also do 0, snrmax
)

plt.xlabel(f"{iso_label(iso1)} / {iso_label(iso2)} Ratio")
plt.ylabel("S/N")
plt.title("Evans Plot Colour Legend")

plt.tight_layout()
#plt.savefig("Evans_plots/Legend.png", dpi=300, bbox_inches="tight")
plt.show()