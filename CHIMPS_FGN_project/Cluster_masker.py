#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:13:46 2025

@author: kyriakos.trakakis
"""

from astropy.io import fits
import numpy as np
import subprocess
import os
from os.path import join
from astropy.table import Table
from matplotlib.pylab import *
import pandas as pd


starlink_path=  '/usr/local/star-2023A/bin/convert/fits2ndf'

def run_command(command, cwd=None):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)

# --- File and label list ---
input_path      = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
base_prefix     = "FGN_027_13CO_23arcsec_aligned_SNR_spherical_source_inserted_min_del_2_parallel_cluster"
input_filename  = base_prefix + 's.fits'
mult_filename   = "FGN_027_13CO_23arcsec_aligned.sdf"
results         = []


# --- Read cluster numbers from fits file if ther are many ---
#with fits.open("FGN_027_037_23arcsec_aligned_SNR_min_del_2_parallel_clusters_vrms_3_table.fits") as hdul_1:
#    data_1          =  Table(hdul_1[1].data)
#    cluster_numbers = data_1.columns[0]
cluster_numbers = {0}

# --- Load original data ---
with fits.open(os.path.join(input_path, input_filename)) as hdul:
    data = hdul[0].data
    header = hdul[0].header
    
source_command = f"source {starlink_path}"


# --- Loop over each number and create individual FITS files ---
for num in cluster_numbers:
    mask = (data == num)
    if num == 0:
        # Replace cluster 0 voxels with 1, keep background (-1) untouched
        new_data = np.where(mask, 1, 0)
    else:
        new_data = np.where(mask, data, 0)
    
    output_filename = f"{base_prefix}_{num}.fits"
    output_filename = os.path.join(input_path, output_filename)
    
    hdu = fits.PrimaryHDU(new_data, header=header)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Written: {output_filename}")
    
    ndffile         = f"{base_prefix}_{num}.sdf"
    run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {output_filename} {ndffile}")
    os.system('rm -rf ' + output_filename)
    
    ndffile_2       = f"{base_prefix}_{num}_masked.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/mult IN1={mult_filename} IN2={ndffile} OUT={ndffile_2}")
    os.system('rm -rf '+ndffile)
    
    if num == 0 :
        num = 1
    ndffile_3       = f"{base_prefix}_{num}_data.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/cdiv {ndffile_2} SCALAR={num} OUT={ndffile_3}")
    os.system('rm -rf '+ndffile_2)
    
    #run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {ndffile_3} OUT={'*'}")
    
    ndffile_4           = f"{base_prefix}_{num}_integ.sdf"
    fits_file_int       = f"{base_prefix}_{num}_integ.fit"
    run_command(f"/usr/local/star-2023A/bin/kappa/collapse estimator=integ {ndffile_3} AXIS={'VRAD'} OUT={ndffile_4}")
    run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {ndffile_4} OUT={'*'}")
    os.system('rm -rf '+ndffile_3)
    os.system('rm -rf '+ndffile_4)


    
    clust_data, headerr = fits.getdata(fits_file_int, 0, header=True)
    clust_good          = clust_data[~isnan(clust_data)]
    clust_good          = clust_good[clust_good>=0]
    nonzero_pixels      = np.count_nonzero(clust_good)
    total_flux          = np.sum(clust_good)
    if header['CUNIT3'] == 'm/s':
        total_flux = total_flux / 1000
        
  #  print(num, np.sum(clust_good), np.count_nonzero(clust_good))
    results.append((num, total_flux, nonzero_pixels))

print(results)    
    

for num, flux, count in results:
    print(f"{num:<10} {flux:15.3f} {count:20}")
   # print(f"{flux:15.3f} {count:20}")


results_df = pd.DataFrame(results)
results_df.to_csv("FGN_027_inserted_fluxes_pixels.csv", index=False)
