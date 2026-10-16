#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:40:11 2026

@author: kyriakos.trakakis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reid et al. (2019), Table 2
arms = {
    "3-kpc (near)": (15, 18, 15, 3.52, -4.2, -4.2),
    "Norma":        (5, 54, 18, 4.46, -1.0, 19.5),
    "Scutum-Cen":   (0, 104, 23, 4.91, 14.1, 12.1),
    "Sgr-Car":      (2, 97, 24, 6.04, 17.1, 1.0),
    "Local":        (-8, 34, 9, 8.26, 11.4, 11.4),
    "Perseus":      (-23, 115, 40, 8.87, 10.3, 8.7),
    "Outer":        (-16, 71, 18, 12.24, 3.0, 9.4)
}

def spiral_radius(beta, beta_kink, R_kink, psi_low, psi_high):

    beta = np.asarray(beta, dtype=float)

    psi = np.where(
        beta < beta_kink,
        psi_low,
        psi_high
    )

    delta_beta = np.deg2rad(beta - beta_kink)

    return R_kink * np.exp(
        -delta_beta * np.tan(np.deg2rad(psi))
    )


results = []

for arm, params in arms.items():

    beta_min, beta_max, beta_kink, R_kink, psi_low, psi_high = params

    beta = np.linspace(beta_min, beta_max, 10000)

    radii = spiral_radius(
        beta, beta_kink, R_kink, psi_low, psi_high
    )

    results.append({
        "Arm": arm,
        "R_min": radii.min(),
        "R_max": radii.max(),
    })

arm_df = pd.DataFrame(results)

print(arm_df.round(3))



R0 = 8.15
l_min, l_max = 17, 47

results = []
selected_points = {}

for arm, params in arms.items():

    beta_min, beta_max, beta_kink, R_kink, psi_low, psi_high = params

    # Sample the fitted arm segment
    beta = np.linspace(beta_min, beta_max, 100000)

    R = spiral_radius(
        beta, beta_kink, R_kink, psi_low, psi_high
    )

    # Galactocentric Cartesian coordinates
    beta_rad = np.deg2rad(beta)

    X = R * np.cos(beta_rad)
    Y = R * np.sin(beta_rad)

    # Galactic longitude as seen from the Sun
    lon = np.rad2deg(
        np.arctan2(Y, R0 - X)
    ) % 360
    
    # Heliocentric distance
    d = np.sqrt(
    (R0 - X)**2 + Y**2
    )

    # Restrict to 17 <= l <= 47 degrees
    mask = (lon >= l_min) & (lon <= l_max) & (d > 0)

    if not np.any(mask):
        print(f"{arm}: no points in longitude range")
        continue

    R_selected = R[mask]

    selected_points[arm] = R_selected

    results.append({
        "Arm": arm,
        "R_min": R_selected.min(),
        "R_max": R_selected.max(),
        "N_points": mask.sum(),
        "d_min": d[mask].min(),
        "d_max": d[mask].max()
    })

arm_df = pd.DataFrame(results)

print('\t \t CHIMPS-FGN case')
print(arm_df.round(3))


R0 = 8.15
l_min, l_max = 250, 355

results = []
selected_points = {}

for arm, params in arms.items():

    beta_min, beta_max, beta_kink, R_kink, psi_low, psi_high = params

    # Extend beyond the parallax-source azimuth intervals
    beta = np.linspace(-180, 180, 200000)

    R = spiral_radius(
        beta, beta_kink, R_kink, psi_low, psi_high
    )

    beta_rad = np.deg2rad(beta)

    X = R * np.cos(beta_rad)
    Y = R * np.sin(beta_rad)

    # Reid et al. Galactocentric azimuth convention
    lon = np.rad2deg(
        np.arctan2(-Y, R0 - X)
    ) % 360
    
    d = np.sqrt(
    (R0 - X)**2 + Y**2
    )

    mask = (lon >= l_min) & (lon <= l_max) & (d > 0)

    if not np.any(mask):
        print(f"{arm}: no matches")
        continue

    R_selected = R[mask]

    selected_points[arm] = R_selected

    results.append({
        "Arm": arm,
        "R_min": R_selected.min(),
        "R_max": R_selected.max(),
        "d_min": d[mask].min(),
        "d_max": d[mask].max()
    })

arm_df = pd.DataFrame(results)

# %%
print('\t \t Mopra case')

print(arm_df.round(3))