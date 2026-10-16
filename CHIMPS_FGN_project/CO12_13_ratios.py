#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:01:22 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd

## Trim the cubes first to the range where we have emission! 12 and 13, load the trimmed cubes here
## Do SNR too but let's start with trimming
## Maybe also do SNR while we have the cubes downloaded

#path_13 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
#path_12 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/"
region  = "046"
#file_13 = path_13 +f"FGN_{region}_13CO_23arcsec_aligned_trimmed.fits"
#file_12 = path_12 +f"FGN_{region}_12CO_23arcsec_aligned_trimmed.fits"
file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned_nomagic.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned_nomagic.fits"
SNR_filename = path_13 + f"FGN_{region}_13CO_23arcsec_aligned_SNR_nomagic.fits"


## Mopra ##
#---------------------
path_13 = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra"
path_12 = path_13


with fits.open(file_13) as hdul:
    data_13        = hdul[0].data
    data_13_header = hdul[0].header

with fits.open(file_12) as hdul:
    data_12        = hdul[0].data
    data_12_header = hdul[0].header
    
def sum_extractor(cube_13, cube_12, Mask, snr_cube):
    with fits.open(cube_13) as hdul:
        data_13        = hdul[0].data
        data_13_header = hdul[0].header

    with fits.open(cube_12) as hdul:
        data_12        = hdul[0].data
        data_12_header = hdul[0].header

    region = file_13[57:60]
    print(Mask)

    if Mask == True:
        with fits.open(snr_cube) as hdul:
            snr = hdul[0].data

        mask    = snr > 3
        data_13 = np.where(mask, data_13, 0)
        data_12 = np.where(mask, data_12, 0)


    total_ratio = np.sum(data_12) / np.sum(data_13)

    CRVAL1 = data_13_header['CRVAL1']
    CRPIX1 = data_13_header['CRPIX1']
    CDELT1 = data_13_header['CDELT1']

    CRVAL2 = data_13_header['CRVAL2']
    CRPIX2 = data_13_header['CRPIX2']
    CDELT2 = data_13_header['CDELT2']

    CRVAL3 = data_13_header['CRVAL3']
    CRPIX3 = data_13_header['CRPIX3']
    CDELT3 = data_13_header['CDELT3']

    low_l  = float(region[1:]) - 1
    high_l = low_l + 2

    low_b  = -0.5
    high_b = 0.5

    low_v  = CRVAL3 + (1 - CRPIX3) * CDELT3
    high_v = CRVAL3 + (data_13_header['NAXIS3'] - CRPIX3) * CDELT3

    print(f'Longitude limits = {low_l:.2f} - {high_l:.2f}')
    print(f'Latitude limits = {low_b:.2f} - {high_b:.2f}')
    print(f'Velocity limits = {low_v:.2f} - {high_v:.2f}')

    nv, nb, nl = data_13.shape

    # Create bin edges (indices)
    v_bins = np.linspace(0, nv, 11, dtype=int)
    b_bins = np.linspace(0, nb, 11, dtype=int)
    l_bins = np.linspace(0, nl, 11, dtype=int)

    # Output arrays
    sum_v_13 = np.zeros(10)
    sum_b_13 = np.zeros(10)
    sum_l_13 = np.zeros(10)

    sum_v_12 = np.zeros(10)
    sum_b_12 = np.zeros(10)
    sum_l_12 = np.zeros(10)

    edges_l = np.zeros(10)
    edges_b = np.zeros(10)
    edges_v = np.zeros(10)

    # ---- Velocity bins 13 ----
    for i in range(10):
        v_start, v_end = v_bins[i], v_bins[i+1]
        subcube = data_13[v_start:v_end, :, :]
        sum_v_13[i] = np.nansum(subcube)
        v_cen = (v_start + v_end) / 2 + 1
        edges_v[i] = v_cen

    # ---- Latitude bins 13 ----
    for i in range(10):
        b_start, b_end = b_bins[i], b_bins[i+1]
        subcube = data_13[:, b_start:b_end, :]
        sum_b_13[i] = np.nansum(subcube)
        b_cen = (b_start + b_end) / 2 + 1
        edges_b[i] = b_cen

    # ---- Longitude bins 13 ----
    for i in range(10):
        l_start, l_end = l_bins[i], l_bins[i+1]
        subcube = data_13[:, :, l_start:l_end]
        sum_l_13[i] = np.nansum(subcube)
        l_cen = (l_start + l_end) / 2 + 1
        edges_l[i] = l_cen

    # ---- Velocity bins 12 ----
    for i in range(10):
        v_start, v_end = v_bins[i], v_bins[i+1]
        subcube = data_12[v_start:v_end, :, :]
        sum_v_12[i] = np.nansum(subcube)

    # ---- Latitude bins 12 ----
    for i in range(10):
        b_start, b_end = b_bins[i], b_bins[i+1]
        subcube = data_12[:, b_start:b_end, :]
        sum_b_12[i] = np.nansum(subcube)

    # ---- Longitude bins 12 ----
    for i in range(10):
        l_start, l_end = l_bins[i], l_bins[i+1]
        subcube = data_12[:, :, l_start:l_end]
        sum_l_12[i] = np.nansum(subcube)

    l_values = CRVAL1 + (edges_l - CRPIX1) * CDELT1
    b_values = CRVAL2 + (edges_b - CRPIX2) * CDELT2
    v_values = CRVAL3 + (edges_v - CRPIX3) * CDELT3

    ratio_l = np.full_like(sum_l_12, 0)
    ratio_b = np.full_like(sum_b_12, 0)
    ratio_v = np.full_like(sum_v_12, 0)

    # ---- Longitude ----
    ratio_l[sum_l_13 != 0] = sum_l_12[sum_l_13 != 0] / sum_l_13[sum_l_13 != 0]
    plt.figure()
    plt.plot(l_values, ratio_l)
    plt.xlabel('l (degrees)')
    plt.ylabel('ratio')
    plt.title('Ratio vs Longitude')
    plt.show()

    # ---- Latitude ----
    ratio_b[sum_b_13 != 0] = sum_b_12[sum_b_13 != 0] / sum_b_13[sum_b_13 != 0]
    plt.figure()
    plt.plot(b_values, ratio_b)
    plt.xlabel('b (degrees)')
    plt.ylabel('ratio')
    plt.title('Ratio vs Latitude')
    plt.show()

    # ---- Velocity ----
    ratio_v[sum_v_13 != 0] = sum_v_12[sum_v_13 != 0] / sum_v_13[sum_v_13 != 0]
    plt.figure()
    plt.plot(v_values, ratio_v)
    plt.xlabel('v (km/s)')
    plt.ylabel('ratio')
    plt.title('Ratio vs Velocity')
    plt.show()

    df_l = pd.DataFrame({
    'l_values': l_values,
    'sum_l_12': sum_l_12,
    'sum_l_13': sum_l_13,
    'ratio_l': ratio_l
    })

    df_l = df_l.iloc[::-1].reset_index(drop=True)

    df_b = pd.DataFrame({
    'b_values': b_values,
    'sum_b_12': sum_b_12,
    'sum_b_13': sum_b_13,
    'ratio_b': ratio_b
    })

    df_v = pd.DataFrame({
    'v_values': v_values,
    'sum_v_12': sum_v_12,
    'sum_v_13': sum_v_13,
    'ratio_v': ratio_v
    })

    l_name_1 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_longitude_ratio_trimmed.csv"
    b_name_1 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_latitude_ratio_trimmed.csv"
    v_name_1 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_velocity_ratio_trimmed.csv"

    l_name_2 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_longitude_ratio_masked.csv"
    b_name_2 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_latitude_ratio_masked.csv"
    v_name_2 = path_13 + "Ratio testing FGN/" + f"FGN_12_13_{region}_velocity_ratio_masked.csv"

    if "trimmed" in file_13:
        df_l.to_csv(l_name_1, index= False)
        df_b.to_csv(b_name_1, index= False)
        df_v.to_csv(v_name_1, index= False)
    elif "nomagic" in file_13:
        df_l.to_csv(l_name_2, index= False)
        df_b.to_csv(b_name_2, index= False)
        df_v.to_csv(v_name_2, index= False)
    else:
        print('Wrong name')

    return total_ratio, df_l, df_b, df_v

Mask = "nomagic" in file_13
test_ratio_1, test_l_1, test_b_1, test_v_1 = sum_extractor(file_13, file_12, Mask, SNR_filename)
