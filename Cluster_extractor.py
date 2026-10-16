#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:20:06 2025

@author: kyriakos.trakakis
"""

import os
import numpy as np
import matplotlib
from matplotlib.pylab import *
from astropy import wcs
from astropy.io import fits
from astropy.wcs import WCS

# This code takes a cluster map from scimes and a cluster number and extracts a fits file
# of only the area of the cluster (cube of lmin:lmax, bmin:bmax, vmin:vmax).
# All other things that might be in the area and are not part of the cluster are set to 0. 
# Also divides by the cluster number to make it binary

fits_file='/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO/G340-350-13CO_smallercube_05res_min_del_2_parallel_clusters.fits'
number = 1705

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

    # Mask: keep only voxels equal to the target number
mask = (data == number)
matching_indices = np.argwhere(mask)

if matching_indices.size == 0:
    print(f"No voxels with value {number} found.")

vel_indices = matching_indices[:, 0]
lat_indices = matching_indices[:, 1]
lon_indices = matching_indices[:, 2]

# Pixel bounds
pix_bounds = {
    "lon": (np.min(lon_indices), np.max(lon_indices)),
    "lat": (np.min(lat_indices), np.max(lat_indices)),
    "vel": (np.min(vel_indices), np.max(vel_indices)),
}

# World coordinate conversion using proper CRVALn terms
crpix1, cdelt1, crval1 = header["CRPIX1"], header["CDELT1"], header["CRVAL1"]
crpix2, cdelt2, crval2 = header["CRPIX2"], header["CDELT2"], header["CRVAL2"]
crpix3, cdelt3, crval3 = header["CRPIX3"], header["CDELT3"], header["CRVAL3"]

world_bounds = {
    "lon": (
        (pix_bounds["lon"][0] + 1 - crpix1) * cdelt1 + crval1,
        (pix_bounds["lon"][1] + 1 - crpix1) * cdelt1 + crval1,
    ),
    "lat": (
        (pix_bounds["lat"][0] + 1 - crpix2) * cdelt2 + crval2,
        (pix_bounds["lat"][1] + 1 - crpix2) * cdelt2 + crval2,
    ),
    "vel": (
        (pix_bounds["vel"][0] + 1 - crpix3) * cdelt3 + crval3,
        (pix_bounds["vel"][1] + 1 - crpix3) * cdelt3 + crval3,
    ),
}
print(f"\nVoxel value: {number}")
print("Pixel Bounds:")
print(f"  Lon: {pix_bounds['lon'][0]} to {pix_bounds['lon'][1]}")
print(f"  Lat: {pix_bounds['lat'][0]} to {pix_bounds['lat'][1]}")
print(f"  Vel: {pix_bounds['vel'][0]} to {pix_bounds['vel'][1]}")

print("World Coordinate Bounds:")
print(f"  Lon: {world_bounds['lon'][0]:.5f} to {world_bounds['lon'][1]:.5f}")
print(f"  Lat: {world_bounds['lat'][0]:.5f} to {world_bounds['lat'][1]:.5f}")
print(f"  Vel: {world_bounds['vel'][0]:.5f} to {world_bounds['vel'][1]:.5f}")

vmin, vmax = np.min(vel_indices), np.max(vel_indices)
latmin, latmax = np.min(lat_indices), np.max(lat_indices)
lonmin, lonmax = np.min(lon_indices), np.max(lon_indices)

trimmed_data = np.where(mask, data, -1)
trimmed_data = trimmed_data[vmin:vmax+1, latmin:latmax+1, lonmin:lonmax+1]


# Divide by number
scaled_data = trimmed_data / np.max([number,1])

# Adjust header WCS keywords to reflect trimmed cube:
# CRPIX shifts because we cut pixels from the beginning
header["CRPIX1"] -= lonmin
header["CRPIX2"] -= latmin
header["CRPIX3"] -= vmin

# Prepare output filename
name_wo_ext = os.path.splitext(fits_file)[0][:-1]
# 'FGN_027_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters'

# You can now build an output filename, e.g.,
output_file = name_wo_ext + f"_{number}.fits"

hdu = fits.PrimaryHDU(data=scaled_data, header=header)
hdu.writeto(output_file, overwrite=True)

print(f"Trimmed and scaled FITS saved to: {output_file}")