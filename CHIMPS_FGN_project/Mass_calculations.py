#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 11:58:54 2025

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, join
from scipy.stats import linregress


table_Trakakis          = Table.read("CHIMPS_13CO_mass_calculations_Trakakis_sample.fits", format='fits')
table_Rigby             = Table.read("CHIMPS_13CO_mass_calculations_Rigby_sample.fits", format='fits')


df_me = table_Trakakis.to_pandas()
df_Andy = table_Rigby.to_pandas()

df_me = df_me.drop(['NH2_ellipse', 'angular_size_sr', 'angular_size_sr_new', 'area_cm2_pixel_method', 'area_cm2_new', 'NH2_pixel'], axis = 1)
df_me = df_me.drop(['NH2_pixel_new', 'M_pixel_g', 'M_pixel_g_new', 'M_pixel_Sol', 'M_pixel_Sol_new', 'M_COLTE_Sol_CO'], axis = 1)
df_me = df_me.drop(['massX_ours', 'M_COLTE_Sol', 'Mass_Test'], axis = 1)


df_Andy = df_Andy.drop(['M_pixel_g', 'angular_size_sr', 'area_cm2', 'NH2_pixel', 'M_pixel_Sol', 'COLTE_N13_total', 'mass_ratio'], axis = 1)
df_Andy = df_Andy.drop(['NH2_COLTE', 'M_COLTE_g', 'M_COLTE_Sol', 'R_test', 'R_ratio', 'WCO', 'M_pixel_equation'], axis = 1)
df_Andy = df_Andy.drop(['massX', 'M_COLTE_equation', 'l_cen_2', 'b_cen_2', 'v_cen_2'], axis = 1)

## Constants and definitions
eta_mb = 0.72
dv     = 0.5 * u.km / u.s

mu_Andy = 2.8
mu_me   = 2.33

Xco    = 6E21 * u.cm**-2 / (u.K * u.km / u.s)  # From Rigby et al. 2025
R_Andy = 5.8*10**5
R_me   = 9.3*10**5

dist_Andy = df_Andy['Dist'].values * u.kpc
dist_Andy = dist_Andy.to('cm')
dist_me   = df_me['distance_cm'].values * u.cm

pixsize_old = (7.612 * u.arcsec).to('rad').value
pixsize_new = (6 * u.arcsec).to('rad').value

Andy_N_COLTE = df_Andy['N13CO'].values * (u.cm ** -2)
Our_N_COLTE  = df_me['COLTE_Density'].values * (u.cm ** -2)

Wco    = df_Andy['Sum_13CO'].values * u.K * dv / eta_mb # Integrated intensity
Wco_me = df_me['total_flux'].values * u.K * u.m / u.s



## Andy COLTE Mass of his sample
Mass = df_Andy['Mass'].values * u.Msun
Mass_Test_py = mu_Andy * c.m_p * R_Andy * dist_Andy ** 2 * df_Andy['n_voxels'].values * Andy_N_COLTE * pixsize_old**2
Mass_Test_py = Mass_Test_py.to('Msun')


## Andy X-factor mass of his sample
massX = mu_Andy * c.m_p * dist_Andy ** 2 * Wco * Xco * pixsize_old  ** 2
massX = massX.to('Msun')


## Our COLTE mass of Andy's sample
# This vs Andy's COLTE should come up to just a factor difference between mu_Andy and R_Andy
M_COLTE_Andy = mu_me * c.m_p * R_me * dist_Andy ** 2 * df_Andy['n_voxels'].values * Andy_N_COLTE * pixsize_old ** 2
M_COLTE_Andy = M_COLTE_Andy.to('Msun')


## Our X-factor mass of Andy's sample
# This should come up to just a factor difference between mu_Andy
M_pixel_Andy = mu_me * c.m_p * dist_Andy ** 2 * pixsize_old ** 2 * Wco * Xco
M_pixel_Andy = M_pixel_Andy.to('Msun')


##  Andy's COLTE mass of our sample
Mass_Test_our = mu_Andy * c.m_p * R_Andy * dist_me ** 2 * Our_N_COLTE * pixsize_new * 2
Mass_Test_our = Mass_Test_our.to('Msun')


##  Andy's X-factor mass of our sample
massX_ours = mu_Andy * c.m_p * dist_me ** 2 * Xco * Wco_me * pixsize_new ** 2
massX_ours = massX_ours.to('Msun')


## Our COLTE mass of our sample
M_COLTE_me = mu_me * c.m_p * R_me * dist_me ** 2 * Our_N_COLTE * pixsize_new ** 2
M_COLTE_me = M_COLTE_me.to('Msun')


## Our X-factor mass of our sample
M_pixel_me = mu_me * c.m_p * dist_me ** 2 * pixsize_new ** 2 * Wco_me * Xco
M_pixel_me = M_pixel_me.to('Msun')

Mass_val          = Mass.value
Mass_Test_val     = Mass_Test_py.value
massX_val         = massX.value
M_COLTE_Andy_val  = M_COLTE_Andy.value
M_pixel_Andy_val  = M_pixel_Andy.value
Mass_Test_our_val = Mass_Test_our.value
massX_ours_val    = massX_ours.value 
M_COLTE_me_val    = M_COLTE_me.value
M_pixel_me_val    = M_pixel_me.value


COLTE_factor_me = Our_N_COLTE * R_me
X_factor_me     = Wco_me * Xco

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(Mass_val, Mass_Test_val, label='Andy COLTE (recomputed)', alpha=0.7)
plt.scatter(Mass_val, M_COLTE_Andy_val, label='Our COLTE', alpha=0.7)
plt.plot([Mass_val.min(), Mass_val.max()], [Mass_val.min(), Mass_val.max()], 'k--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Andy Mass [Msun]")
plt.ylabel("Recomputed COLTE Mass [Msun]")
plt.legend()
plt.title("COLTE Mass Comparisons")

plt.subplot(1, 2, 2)
plt.scatter(Mass_val, massX_val, label='Andy X-factor', alpha=0.7)
plt.scatter(Mass_val, M_pixel_Andy_val, label='Our X-factor', alpha=0.7)
plt.plot([Mass_val.min(), Mass_val.max()], [Mass_val.min(), Mass_val.max()], 'k--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Andy Mass [Msun]")
plt.ylabel("X-factor Mass [Msun]")
plt.legend()
plt.title("X-factor Mass Comparisons")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(massX_val, M_COLTE_Andy_val, alpha=0.7)
plt.plot([massX_val.min(), massX_val.max()], [massX_val.min(), massX_val.max()], 'k--', label='1:1 Line')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("X-factor Mass [M$_\odot$]")
plt.ylabel("COLTE Mass [M$_\odot$]")
plt.title("X-factor vs COLTE Mass")
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()


y = massX_val.value if hasattr(massX_val, 'unit') else massX_val
x = Mass_Test_val.value if hasattr(Mass_Test_val, 'unit') else Mass_Test_val

# Remove non-positive values (log10 is undefined for 0 or negative)
mask = (x > 0) & (y > 0)
x = x[mask]
y = y[mask]

# Take base-10 logarithm
logx = np.log10(x)
logy = np.log10(y)

# Fit line: log10(y) = m * log10(x) + c
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)

# Generate fit line in original units
fit_x = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
fit_y = 10**intercept * fit_x**slope

# Plot
plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.7, label='Data')
plt.plot(fit_x, fit_y, 'r--', label=f'Fit: y = {10**intercept:.2f} × x^{slope:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('X-factor Mass [Msun]')
plt.xlabel('COLTE Mass [Msun]')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.title('Log-Log Fit: X-factor vs COLTE Mass')
plt.tight_layout()
plt.show()

# Optional: print fit parameters
print(f"Slope (power-law index): {slope:.3f}")
print(f"Intercept (log10): {intercept:.3f}")
print(f"Fit equation: y = {10**intercept:.3f} * x^{slope:.3f}")
print(f"R²: {r_value**2:.3f}")
