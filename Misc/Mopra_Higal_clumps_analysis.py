#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:58:46 2025

@author: kyriakos.trakakis
"""

import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import os
from math import sin, cos, tan, radians, sqrt, pi
from os.path import join
from matplotlib.pylab import *
from scipy.optimize import curve_fit



cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/Mopra_HiGAL_beam_fillers_above_threshold.csv") 
cat_csv = cat_csv[cat_csv['log10(mopra_lum/mopra_mass)'] > -999] 
df = cat_csv

df = df[df['mopra_diam_pc'] > df['mopra_dist_kpc'] / 10]


logL        = df['log10(mopra_lum)'].values
logM        = df['log10(mopra_mass)'].values
M           = 10 ** logM
L           = 10 ** logL
LM          = L / M
logLM       = df['log10(mopra_lum/mopra_mass)'].values
Rgc_kpc     = df['mopra_rgc_kpc'].values
d_kpc       = df['mopra_dist_kpc'].values
vlsr_kms    = df['mopra_vlsr_final'].values
gamma       = df['gas2dust_ratio'].values
diameter_pc = df['mopra_diam_pc'].values

## Fitting power law

# Define the model: power law
def power_law(R, c, a):
    return c * R**a

# Fit the model, example M ~ c * R ^ a
popt, pcov = curve_fit(power_law, Rgc_kpc, logM)
c_fit, a_fit = popt

# Print results
print(f"Best-fit parameters: c = {c_fit:.4f}, a = {a_fit:.4f}")

# Plot the data and the fit
R_fit = np.logspace(np.log10(min(Rgc_kpc)), np.log10(max(Rgc_kpc)), 100)
M_fit = power_law(R_fit, c_fit, a_fit)

plt.figure(figsize=(6, 4))
plt.scatter(Rgc_kpc, logM, color='blue')
plt.plot(R_fit, M_fit, label=fr'Fit: $M = {c_fit:.5f} R^{{{a_fit:.2f}}}$', color='red')
#plt.xscale('log')
plt.yscale('log')
plt.title('Beam-filling clumps')
plt.ylabel('logM ($M_\odot$)')
plt.xlabel('$R_{gc}$ (pc)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
#plt.show()
#plt.savefig('Larson_mass_v_sigma_big_code.png')





## Colour - coding

# Example, logM = a * d(pc) + c
# Define weights (e.g., directly or 1/W if you want inverse-distance weight)
W = Rgc_kpc
X = d_kpc
X_var = "$D_{hc}$"
X_unit = "kpc"
Y = diameter_pc / 2
Y_var = "radius"
Y_unit = "pc"
weights = W  # or 1/W, or 1/W**2 depending on purpose

# Weighted polyfit
fit = np.polyfit(X, Y, 1, w=weights)
a_fit, c_fit = fit

print(f"Weighted fit: {Y_var} = {a_fit:.4f} * {X_var} + {c_fit:.4f}")

# Plot
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, c=W, cmap='inferno_r', s=20)#, edgecolor='k')
#plt.plot(X, a_fit * X + c_fit, color='red', label=fr'Fit: ${Y_var} = \ {a_fit:.2f}*{X_var} + {c_fit:.2f}$')
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(rf'{X_var} ({X_unit})')
plt.ylabel(rf'{Y_var} ({Y_unit})')
plt.title('Beam-filling clumps')
plt.legend()
plt.grid(True)
plt.colorbar(label='$R_{gc}$')
plt.tight_layout()
#plt.show()
#plt.savefig('FUGIN_13CO_Mass_radius_python.png')


mask_zero = df['mopra_evol_flag'] == 0
mask_nonzero = df['mopra_evol_flag'] != 0

plt.figure(figsize=(6, 4))

# Plot points with mopra_evol_flag == 0
plt.scatter(
    X[mask_zero], Y[mask_zero],
    c='blue', s=20, label='unbound'
)

# Plot points with mopra_evol_flag != 0
plt.scatter(
    X[mask_nonzero], Y[mask_nonzero],
    c='red', s=20, label='bound'
)

# Optional: fit line (unchanged)
plt.xlabel(rf'{X_var} ({X_unit})')
plt.ylabel(rf'{Y_var} ({Y_unit})')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()


rgc = df["mopra_rgc_kpc"]
lm_log = df["log10(mopra_lum/mopra_mass)"]

# Scatter plot of raw data
plt.figure(figsize=(8,6))
plt.scatter(rgc, 10**lm_log, alpha=0.4, label="Raw data", s=10)

# Bin edges from 8 to 14 with step 0.25
bins = np.arange(8, 14, 0.25)

# Compute bin labels (cut into intervals)
rgc_bins = pd.cut(rgc, bins)

# Compute bin means in log space
bin_means = lm_log.groupby(rgc_bins, observed=False).mean()
bin_centers = np.array([interval.mid for interval in bin_means.index])

# --- Remove NaNs (empty bins) ---
mask = ~bin_means.isna()
bin_centers = bin_centers[mask]
bin_means = bin_means[mask]

# Overlay average trend
plt.scatter(bin_centers, 10**bin_means, color="red", s=40, label="Bin averages")
plt.plot(bin_centers, 10**bin_means, color="red", linestyle="--", alpha=0.7)

# --- Linear fit in log space ---
coeffs = np.polyfit(bin_centers, bin_means, 1)  # slope & intercept in log10 space
fit_line = np.polyval(coeffs, bin_centers)

# Plot the fit (back in linear space)
plt.plot(bin_centers, 10**fit_line, color="blue", linewidth=2,
         label=f"Fit: log10(L/M) = {coeffs[0]:.3f}*Rgc + {coeffs[1]:.3f}")

# Labels, log scale, legend
plt.xlabel("Rgc [kpc]")
plt.ylabel("L/M")
plt.yscale("log")
plt.title("Scatter with binned averages and linear fit (filled beam)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
