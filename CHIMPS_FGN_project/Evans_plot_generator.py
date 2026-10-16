#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:08:58 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import hsv_to_rgb

# ---------------------------------------
# FILES
# ---------------------------------------
path_13 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
path_12 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/"
#path_18 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/C18O/"
region  = "037_047"
isotopologues = ["12","13"]

file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"
#file_18 = path_18 + f"FGN_{region}_C18O_23arcsec_aligned.fits"
#SNR_18 = path_18 + f"FGN_{region}_C18O_23arcsec_aligned_SNR.fits"
SNR_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned_SNR.fits"
SNR_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned_SNR.fits"
# ---------------------------------------
# LOAD
# ---------------------------------------
data12 = fits.getdata(file_12)
data13 = fits.getdata(file_13)
#data18 = fits.getdata(file_18)
#snr18  = fits.getdata(SNR_18)
snr12  = fits.getdata(SNR_12)
snr13  = fits.getdata(SNR_13)

with fits.open(file_13) as hdul:
    data_13 = hdul[0].data
    data_13_header = hdul[0].header

nv, ny, nx = data12.shape

# ---------------------------------------
# SETTINGS
# ---------------------------------------
rmin = 1
rmax = 15
snrmax = 10

crval3 = data_13_header['CRVAL3']
crpix3 = data_13_header['CRPIX3']
cdelt3 = data_13_header['CDELT3']

nchan = data_13.shape[0]

channels = np.arange(nchan)

vel = crval3 + (channels + 1 - crpix3) * cdelt3

if region == "017_027":
    vlow, vhigh = 0, 130
elif region == "027_037":
    vlow, vhigh = 10, 120
else:
    vlow, vhigh = -10, 100


if abs(cdelt3) > 100:
    vel = vel / 1000.0
    
if region == "017_027":
    sigma = {
    "12": 0.91,
    "13": 0.51,
    "18": 0.52
}
elif region == "027_037":
    #sigma_13 = 0.54
    #sigma_12 = 1.24
    #sigma_18 = 0.59
    sigma = {
    "12": 1.24,
    "13": 0.54,
    "18": 0.59
}
else:
    sigma_13 = 0.26
    sigma_12 = 0.66
    sigma_18 = 0.31    
    
# ---------------------------------------
# LOOP THROUGH VELOCITY BLOCKS
# ---------------------------------------
vmin = np.floor(vel.min()/10)*10
vmax = np.ceil(vel.max()/10)*10

bins = np.arange(vlow, vhigh + 10, 10)

for i in range(len(bins)-1):

    v0 = bins[i]
    v1 = bins[i+1]

    mask = (vel >= v0) & (vel < v1)

    cube12 = data12[mask,:,:]
    cube13 = data13[mask,:,:]
    s12    = snr12[mask,:,:]
    s13    = snr13[mask,:,:]

    # -----------------------------------
    # collapse intensities
    # -----------------------------------
    I12 = np.nansum(cube12, axis=0)
    I13 = np.nansum(cube13, axis=0)

    ratio = np.full_like(I12, np.nan, dtype=float)
    good = I13 > 0
    ratio[good] = I12[good] / I13[good]

    # -----------------------------------
    # confidence map
    # -----------------------------------
    #snr = np.nanmax(np.minimum(s12, s13), axis=0)
   
    sigma12_eff = sigma_12 * np.sqrt(nchan)
    sigma13_eff = sigma_13 * np.sqrt(nchan)
    
    a = sigma12_eff / I12
    b = sigma13_eff / I13

   
    snr = 1 / (a ** 2 + b ** 2)
    
    ## Give this a shot at some point, makes the weaker tracer dominate confidence but not a SNR measure by itself
  #  snr_12 = I12 / sigma12_eff
  #  snr_13 = I13 / sigma13_eff
  #  snr = np.minimum(snr_12, snr_13)

    # -----------------------------------
    # HSV conversion
    # -----------------------------------
    
    ratio_clip = np.clip(ratio, rmin, rmax)
    
    H = (ratio_clip - rmin) / (rmax - rmin)

    S = np.clip(snr, 1, snrmax)
    S = (S-1)/(snrmax-1)

    V = np.ones_like(H)

    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    RGB[np.isnan(ratio)] = 1

    # -----------------------------------
    # plot
    # -----------------------------------
    wcs2d = WCS(data_13_header).celestial

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection=wcs2d)

    ax.imshow(RGB, origin='lower')

    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")
    ax.set_title(f"Velocity range {v0}-{v1} km/s Evans Plot")

    plt.tight_layout()
    plt.show()
    
# ==========================================
# 2D LEGEND : Ratio vs S/N
# ==========================================

nx, ny = 300, 300

ratio_vals = np.linspace(rmin, rmax, nx)
snr_vals   = np.linspace(1, 10, ny)

RR, SS = np.meshgrid(ratio_vals, snr_vals)

# same transforms as main plot
H_leg = (RR - rmin) / (rmax - rmin)
S_leg = (SS - 1) / 9.0
V_leg = np.ones_like(H_leg)

HSV_leg = np.dstack((H_leg, S_leg, V_leg))
RGB_leg = hsv_to_rgb(HSV_leg)

plt.figure(figsize=(8,6))
plt.imshow(
    RGB_leg,
    origin='lower',
    aspect='auto',
    extent=[rmin, rmax, 1, 10]
)

plt.xlabel(r"$^{12}$CO / $^{13}$CO Ratio")
plt.ylabel("S/N")
plt.title("Evans Plot Colour Legend")

plt.tight_layout()
plt.show()
