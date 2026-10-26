#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:28:52 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Galactic parameters
# ============================================================

R0 = 8.15  # kpc


# ============================================================
# 2. Reid et al. (2019) Table 2 spiral-arm parameters
# ============================================================

# Format:
# (beta_min, beta_max, beta_kink, R_kink, pitch_low, pitch_high)
#
# Angles in degrees; radii in kpc.

parameters = {
    "3-kpc":   (15, 18, 15, 3.52, -4.2, -4.2),
    "Norma":   (5, 54, 18, 4.46, -1.0, 19.5),
    "Sct-Cen": (0, 104, 23, 4.91, 14.1, 12.1),
    "Sgr-Car": (2, 97, 24, 6.04, 17.1, 1.0),
    "Local":   (-8, 34, 9, 8.26, 11.4, 11.4),
    "Perseus": (-23, 115, 40, 8.87, 10.3, 8.7),
    "Outer":   (-16, 71, 18, 12.24, 3.0, 9.4),
}

colours = {
    "3-kpc": "orange",
    "Norma": "red",
    "Sct-Cen": "blue",
    "Sgr-Car": "purple",
    "Local": "cyan",
    "Perseus": "black",
    "Outer": "green",
}


# ============================================================
# 3. Generate spiral-arm coordinates
# ============================================================

def reid_arm(name, n=2000, extend_low=0, extend_high=0):

    bmin, bmax, bkink, Rkink, p_low, p_high = parameters[name]

    beta = np.linspace(
        bmin - extend_low,
        bmax + extend_high,
        n
    )

    pitch = np.where(
        beta <= bkink,
        p_low,
        p_high
    )

    R = Rkink * np.exp(
        -np.radians(beta - bkink)
        * np.tan(np.radians(pitch))
    )

    X = R * np.sin(np.radians(beta))
    Y = R * np.cos(np.radians(beta))

    original = (beta >= bmin) & (beta <= bmax)

    return {
        "beta": beta,
        "R": R,
        "X": X,
        "Y": Y,
        "original": original
    }


arms = {
    name: reid_arm(
        name,
        extend_low=0,
        extend_high=0
    )
    for name in parameters
}


# ============================================================
# 4. Download VizieR maser catalogue (Reid et al. 2019)
# ============================================================

vizier = Vizier(columns=["**"], row_limit=-1)

cat = vizier.get_catalogs("J/ApJ/885/131/table1")[0]

print("Number of sources:", len(cat))
print("Catalogue columns:", cat.colnames)


# ============================================================
# 5. Convert maser positions to face-on Galactic coordinates
# ============================================================

sky = SkyCoord(
    ra=cat["RAJ2000"],
    dec=cat["DEJ2000"],
    unit=(u.hourangle, u.deg),
    frame="icrs"
)

l_maser = sky.galactic.l.to_value(u.rad)
b_maser = sky.galactic.b.to_value(u.rad)

plx = np.asarray(cat["plx"], dtype=float)

valid = np.isfinite(plx) & (plx > 0)

d = np.full(len(plx), np.nan)
d[valid] = 1.0 / plx[valid]  # mas -> kpc

# Sun at (0, R0)
X_maser = d * np.cos(b_maser) * np.sin(l_maser)
Y_maser = R0 - d * np.cos(b_maser) * np.cos(l_maser)


# ============================================================
# 6. Plot face-on spiral arms and masers
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

for name, arm in arms.items():

    ax.plot(
        arm["X"],
        arm["Y"],
        color=colours[name],
        linewidth=2,
        label=name
    )

ax.scatter(
    X_maser[valid],
    Y_maser[valid],
    s=18,
    color="gray",
    alpha=0.65,
    edgecolors="none",
    label="Reid et al. (2019) masers",
    zorder=5
)

ax.scatter(
    0, 0,
    marker="*",
    s=200,
    color="red",
    label="Galactic Centre",
    zorder=10
)

ax.scatter(
    0, R0,
    s=120,
    color="orange",
    label="Sun",
    zorder=10
)

ax.set_aspect("equal")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")

ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# ============================================================
# 7. Convert spiral-arm positions to longitude–velocity
# ============================================================

def reid2019_rotation(R, R0=8.15, a2=0.96, a3=1.62):

    R = np.asarray(R, dtype=float)

    lam = (a3 / 1.5)**5
    rho = R / (a2 * R0)
    log_lam = np.log10(lam)

    term1 = 200.0 * lam**0.41

    term2 = np.sqrt(
        0.80 + 0.49 * log_lam
        + 0.75 * np.exp(-0.4 * lam)
        / (0.47 + 2.25 * lam**0.4)
    )

    term3 = (
        (0.72 + 0.44 * log_lam)
        * 1.97 * rho**1.22
        / (rho**2 + 0.61)**1.43
    )

    term4 = (
        1.6 * np.exp(-0.4 * lam)
        * rho**2
        / (rho**2 + 2.25 * lam**0.4)
    )

    return (term1 / term2) * np.sqrt(term3 + term4)


Theta0 = float(reid2019_rotation(R0))

print(f"Circular speed at the Sun: {Theta0:.2f} km/s")

arms_lbv = {}

for name, arm in arms.items():

    X = arm["X"]
    Y = arm["Y"]
    R = arm["R"]

    # Sun at (0, R0)
    # Positive X corresponds to positive Galactic longitude
    l_rad = np.arctan2(X, R0 - Y)

    l = np.degrees(l_rad) % 360
    b = np.zeros_like(l)

    Theta_R = reid2019_rotation(R)

    # Circular-rotation LSR velocity
    v = (Theta_R * R0 / R - Theta0) * np.sin(l_rad)

    arms_lbv[name] = {
        "l": l,
        "b": b,
        "v": v,
        "R": R,
        "beta": arm["beta"]
    }
    
fig, ax = plt.subplots(figsize=(12, 7)) ## Middle plot

for name, arm in arms_lbv.items():

    l = arm["l"]
    v = arm["v"]

    # Only display the Mopra longitude interval
    mask = (l >= 250) & (l <= 355)

    ax.scatter(
        l[mask],
        v[mask],
        s=4,
        color=colours[name],
        label=name
    )

ax.set_xlim(355, 250)
ax.set_ylim(-170, 120)

ax.set_xlabel("Galactic Longitude (deg)")
ax.set_ylabel(r"$V_{\rm LSR}$ (km/s)")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

for name, arm in arms.items():

    original = arm["original"]

    ax.plot(
        arm["X"][original],
        arm["Y"][original],
        color=colours[name],
        linewidth=2,
        label=name
    )

    ax.scatter(
        arm["X"][~original],
        arm["Y"][~original],
        s=1,
        color=colours[name],
        alpha=0.3
    )

ax.scatter(0, 0, marker="*", s=180, color="red")
ax.scatter(0, R0, s=100, color="orange")

ax.set_aspect("equal")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# Output directory
output_dir = Path("/Users/kyriakos.trakakis/Downloads/Spiral_arms/")
output_dir.mkdir(parents=True, exist_ok=True)

for name, arm in arms_lbv.items():

    df = pd.DataFrame({
        "l": arm["l"],
        "b": arm["b"],
        "v": arm["v"]
    })

    filename = output_dir / f"{name}_lbv.csv"

   # df.to_csv(filename, index=False, float_format="%.8f")

    print(f"Saved: {filename}")