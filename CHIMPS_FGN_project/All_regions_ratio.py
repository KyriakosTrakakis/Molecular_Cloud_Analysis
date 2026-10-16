#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:29:30 2026

@author: kyriakos.trakakis
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


path_13 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
path_12 = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/12CO/"

region  = "017_027"
file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"

with fits.open(file_13) as hdul:
    data_13_R1        = hdul[0].data
    data_13_R1_header = hdul[0].header

with fits.open(file_12) as hdul:
    data_12_R1        = hdul[0].data
    data_12_R1_header = hdul[0].header
    
data_13_R1 = data_13_R1.flatten()
data_12_R1 = data_12_R1.flatten()
    
data_12 = data_12_R1.copy()
data_13 = data_13_R1.copy()

print(f"\n Region {region} loaded")

del data_12_R1
del data_13_R1
    
region  = "027_037"
file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"

with fits.open(file_13) as hdul:
    data_13_R2       = hdul[0].data
    data_13_R2_header = hdul[0].header

with fits.open(file_12) as hdul:
    data_12_R2        = hdul[0].data
    data_12_R2_header = hdul[0].header
    
data_13_R2 = data_13_R2.flatten()
data_12_R2 = data_12_R2.flatten()
    
data_12 = np.concatenate((data_12, data_12_R2), axis=0)
data_13 = np.concatenate((data_13, data_13_R2), axis=0)

print(f"\n Region {region} loaded")

del data_12_R2
del data_13_R2
    
region  = "037_047"
file_13 = path_13 + f"FGN_{region}_13CO_23arcsec_aligned.fits"
file_12 = path_12 + f"FGN_{region}_12CO_23arcsec_aligned.fits"

with fits.open(file_13) as hdul:
    data_13_R3        = hdul[0].data
    data_13_R3_header = hdul[0].header

with fits.open(file_12) as hdul:
    data_12_R3        = hdul[0].data
    data_12_R3_header = hdul[0].header
    
data_13_R3 = data_13_R3.flatten()
data_12_R3 = data_12_R3.flatten()
    
data_12 = np.concatenate((data_12, data_12_R3), axis=0)
data_13 = np.concatenate((data_13, data_13_R3), axis=0)

print(f"\n Region {region} loaded")


del data_12_R3
del data_13_R3


mask = ((~np.isnan(data_12)) & (~np.isnan(data_13)) & (data_13 !=0))

data_12 = data_12[mask]
data_13 = data_13[mask]
ratio = data_12 / data_13

mask_ratio = (ratio > 0 )

data_12 = data_12[mask_ratio]
data_13 = data_13[mask_ratio]
ratio   = ratio[mask_ratio]

print("\n Ratio calculated")

thresholds = np.linspace(0,15,31)

contained_12 = np.zeros(len(thresholds))
contained_13 = np.zeros(len(thresholds))

for i in range(len(thresholds)):
    contained_12[i] = np.sum(data_12[ratio < thresholds[i]]) / np.sum(data_12)
    contained_13[i] = np.sum(data_13[ratio < thresholds[i]]) / np.sum(data_13)
    
    if i % 5 == 0 :
        print(i)
    
plt.figure()
plt.plot(thresholds, contained_12, label = '12CO')
plt.plot(thresholds, contained_13, label ='13CO')
plt.xlabel('Ratio level')
plt.ylabel('Flux contained')
plt.grid("On")
plt.legend()
plt.show()
