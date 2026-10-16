#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:50:37 2025

@author: kyriakos.trakakis
"""

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit


file_path = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/"
file_name = "G340-350-13CO_giantcube_05res.fits"
file = file_path + file_name

# Read cube
hdr = fits.getheader(file)
data = fits.getdata(file)


# Build WCS
w = WCS(hdr)

# 1. Compute spectrum: mean over spatial axes
spectrum = np.nanmean(data, axis=(1, 2))

v0    = hdr["CRVAL3"]   # velocity at reference pixel
dv    = hdr["CDELT3"]   # velocity channel width
v_pix = hdr["CRPIX3"]   # reference pixel (1-indexed)

# 2. Velocity axis from WCS
n_vel = data.shape[0]
chan = np.arange(n_vel)
vel_axis = v0 + (np.arange(n_vel) + 1 - v_pix) * dv

# 3. Plot
plt.figure(figsize=(8,4))
plt.plot(vel_axis, spectrum)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Mean Intensity")
plt.title("Mean Spectrum")
plt.grid(True)
plt.show()

mask = (vel_axis >= -500000) & (vel_axis <= -150000)

v_fit = vel_axis[mask]
y_fit = spectrum[mask]

def sine_func(v, A, freq, phase, offset):
    return A * np.sin(2 * np.pi * freq * v + phase) + offset

A0      = 0.08
freq0   = 1 / 75000      # very small because velocity is large in magnitude
phase0  = 0
offset0 = np.nanmean(y_fit)

p0 = [A0, freq0, phase0, offset0]

popt, pcov = curve_fit(sine_func, v_fit, y_fit, p0=p0)
A, freq, phase, offset = popt

print("Best fit parameters:")
print(f"Amplitude: {A}")
print(f"Frequency: {freq}")
print(f"Phase: {phase}")
print(f"Offset: {offset}")

fit_curve = sine_func(v_fit, *popt)

plt.figure(figsize=(10,6))
plt.plot(v_fit, y_fit, 'k.', label="Data")
plt.plot(v_fit, fit_curve, 'r-', linewidth=2, label="Sine fit")
plt.xlabel("Velocity")
plt.ylabel("Intensity")
plt.legend()
plt.show()

baseline = sine_func(vel_axis, *popt)
new_spectrum = spectrum - baseline

plt.figure(figsize=(8,4))
plt.plot(vel_axis, new_spectrum)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Mean Intensity")
plt.title("Mean Spectrum")
plt.grid(True)
plt.show()

data = data.astype('float32')
baseline_reshaped = baseline.reshape(-1, 1, 1).astype('float32')

corrected_cube = data - baseline_reshaped

re_file_name = file_path + file_name.replace(".fits", "_rebaselined.fits")

hdu = fits.PrimaryHDU(corrected_cube, header = hdr)  # reuse original header
hdu.writeto(re_file_name, overwrite=True)


#var, h_var = fits.getdata('/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/The_Mopra_Southern_Galactic_Plane_Carbon_Monoxide_(CO)_Survey-BkOfmK-X-/data/Supplementary/G250-260-1+1-12CO_beamRSS.fit', 0, header=True)

rms = np.sqrt(abs(var))
