#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:58:22 2025

@author: kyriakos.trakakis
"""
import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import subprocess
import matplotlib.pyplot as plt
import os
from matplotlib.pylab import *


catalog = Table.read('FGN_037_13CO_23arcsec_aligned_SNR_min_del_2_cluster_catalogue.fits', format='fits')
df   = catalog.to_pandas()
cluster_numbers = df['idx']       

mult_file       = "FGN_037_13CO_23arcsec_aligned_SNR"
mult_filename   = mult_file + '.fits'
mult_sdf        = mult_file + '.sdf'

def run_command(command, cwd=None):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)        
        
starlink_path=  '/usr/local/star-2023A/bin/convert/fits2ndf'
source_command = f"source {starlink_path}"

run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {mult_filename} {mult_sdf}")

with fits.open('FGN_037_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters.fits') as hdul:
    data = hdul[0].data
    header = hdul[0].header

max_signal = []

for num in cluster_numbers:
    mask = (data == num)
    new_data = np.where(mask, data, 0)


    base_prefix     = "FGN_037_13CO_23arcsec_aligned_SNR_min_del_2_parallel_cluster"
    output_filename = f"{base_prefix}_{num}.fits"
    
    hdu = fits.PrimaryHDU(new_data, header=header)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Written: {output_filename}")
    
    ndffile         = f"{base_prefix}_{num}.sdf"
    run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {output_filename} {ndffile}")
    os.system('rm -rf ' + output_filename)
    
    ndffile_2       = f"{base_prefix}_{num}_masked.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/mult IN1={mult_sdf} IN2={ndffile} OUT={ndffile_2}")
    os.system('rm -rf '+ndffile)
    
    ndffile_3       = f"{base_prefix}_{num}_data.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/cdiv {ndffile_2} SCALAR={num} OUT={ndffile_3}")
    os.system('rm -rf '+ndffile_2)
    
    run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {ndffile_3} OUT={'*'}")
    
    fits_file = ndffile_3.replace(".sdf", ".fit")

# Compute the maximum signal in the FITS data
    with fits.open(fits_file) as hdul:
        data_cube = hdul[0].data
        max_val = np.nanmax(data_cube)   # ignores NaNs safely
        max_signal.append(max_val)
        
    print(f"Cluster {num}: Max signal = {max_val}")
    
    os.system('rm -rf '+ndffile_3)
    os.system('rm -rf '+fits_file)
    

os.system('rm -rf ' +mult_sdf) 