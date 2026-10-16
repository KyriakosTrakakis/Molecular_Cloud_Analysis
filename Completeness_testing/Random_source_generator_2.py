#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:40:45 2025

@author: kyriakos.trakakis
"""

import numpy as np
from astropy.io import fits


Mass     = 2 * 10 ** 4 # Msol
distance = 2     # kpc
radius   = (Mass / 228) ** (1 / 2.36)   # pc
v_width  = 1.8 * radius ** 0.27 * 2.5   # km / s 
dv       = 0.5   # km / s
dl       = 6     # arcsec  
X_factor = 2.295 * 10 ** 21 # K * km / s
mH       = 1.67 * 2.33 / 10 ** 24 # g
Msol     = 2 * 10 ** 33 # g

## Calculate secondary parameters

r_deg = radius / distance * 180 / np.pi / 1000
r_arcsec = r_deg * 3600
r_pixel = r_arcsec // dl
M_sol_mult = (np.pi * dl / 180 / 3600) ** 2 * (3.086 * 10 ** 21) ** 2 * X_factor * mH / Msol

v_shape = v_width / dv * 7
r_shape = int(r_pixel)
sigma = r_pixel / 2.354
noise_sigma = 0.44
#location=(107, 102, 180)
location=(234, 182, 880)

template_file = "FGN_027_13CO_23arcsec_aligned_SNR.fits"
output_file   = "gaussian_cube_half_sigma_2_highpeak_test.fits"
noise_file = "027_noise_file_rms044.fits"

def make_stochastic_sphere(shape, center, peak, sigma , radius, rms):
    """
    Create a 3D Gaussian sphere with internal random noise.

    Parameters
    ----------
    shape : tuple of ints
        Dimensions of the sphere cube (nv, nb, nl).
    center : tuple or None
        Center position (defaults to middle of the cube).
    peak : float
        Peak intensity of Gaussian.
    sigma : float
        Gaussian sigma in pixels.
    radius : float or None
        Spherical cutoff radius (in pixels). None = no cutoff.
    rms : float
        RMS of uniform random noise applied only to the sphere.

    Returns
    -------
    sphere : ndarray
        3D array of the stochastic Gaussian sphere.
    """
    nv, nb, nl = shape
    v = np.arange(nv)
    b = np.arange(nb)
    l = np.arange(nl)
    V, B, L = np.meshgrid(v, b, l, indexing="ij")

    # Center defaults to middle
    if center is None:
        cv, cb, cl = (nv - 1) / 2, (nb - 1) / 2, (nl - 1) / 2
    else:
        cv, cb, cl = center

    # Squared distance
    dist2 = (V - cv) ** 2 + (B - cb) ** 2 + (L - cl) ** 2

    # Gaussian profile
    sphere = peak * np.exp(-dist2 / (2.0 * sigma**2))

    # Apply spherical cutoff
    if radius is not None:
        sphere[dist2 > radius**2] = 0.0

    # Add stochastic noise (only within the sphere cube)
    noise = np.random.uniform(-rms, rms, size=sphere.shape)
    sphere *= (1 + noise)
    
    total_sum = sphere.sum()
    
    print("Pre normal peak intensity:", sphere.max())
    print("Pre normal Total integrated value:", total_sum)

    return sphere, total_sum

sphere, total = make_stochastic_sphere(shape=(v_shape, r_shape, r_shape),center = None, peak=4, sigma = 5 * sigma / 12, radius= r_pixel // 2, rms=0.25)

conversion_factor = dv * (distance**2) * M_sol_mult  # arbitrary units to match your setup
sphere *= Mass / (total * conversion_factor)

total_sum = sphere.sum()
print("Cube shape:", sphere.shape)
print("Post normal peak intensity:", sphere.max())
print("Post normal Total integrated value:", total_sum)


def insert_sphere_into_template(template_file, output_file, sphere, location):
    """
    Insert a pre-generated 3D sphere into an empty FITS cube (no noise).

    Parameters
    ----------
    template_file : str
        Path to template FITS cube (defines dimensions & header).
    output_file : str
        Path for output FITS file.
    sphere : ndarray
        3D sphere array.
    location : tuple
        (v0, b0, l0) where the sphere center is placed in the template cube.
    """
    with fits.open(template_file) as hdul:
        header = hdul[0].header.copy()
        template_shape = hdul[0].data.shape

    # Empty cube (no noise)
    new_cube = np.zeros(template_shape, dtype=np.float32)

    # Sphere size & center
    nv, nb, nl = sphere.shape
    cv, cb, cl = (nv - 1) // 2, (nb - 1) // 2, (nl - 1) // 2
    v0, b0, l0 = location

    # Compute slices
    v_slice = slice(v0 - cv, v0 - cv + nv)
    b_slice = slice(b0 - cb, b0 - cb + nb)
    l_slice = slice(l0 - cl, l0 - cl + nl)

    # Boundary check
    if v_slice.start < 0 or v_slice.stop > new_cube.shape[0]:
        raise ValueError(f"Velocity axis out of bounds: {v_slice} vs {new_cube.shape[0]}")
    if b_slice.start < 0 or b_slice.stop > new_cube.shape[1]:
        raise ValueError(f"b axis out of bounds: {b_slice} vs {new_cube.shape[1]}")
    if l_slice.start < 0 or l_slice.stop > new_cube.shape[2]:
        raise ValueError(f"l axis out of bounds: {l_slice} vs {new_cube.shape[2]}")

    # Insert sphere
    new_cube[v_slice, b_slice, l_slice] += sphere

    # Save FITS
    hdu = fits.PrimaryHDU(data=new_cube, header=header)
    hdu.writeto(output_file, overwrite=True)

    print(f"Inserted sphere at {location}, saved to {output_file}")


def make_noise_cube(template_file, output_file, noise_sigma=0.44):
    """
    Create a cube with Gaussian random noise of given sigma.

    Parameters
    ----------
    template_file : str
        Path to template FITS cube (defines dimensions & header).
    output_file : str
        Path for output FITS file.
    noise_sigma : float
        RMS of Gaussian noise.
    """
    with fits.open(template_file) as hdul:
        header = hdul[0].header.copy()
        template_shape = hdul[0].data.shape

    # Gaussian noise
    noise_cube = np.random.normal(loc=0.0, scale=noise_sigma, size=template_shape).astype(np.float32)

    # Save FITS
    hdu = fits.PrimaryHDU(data=noise_cube, header=header)
    hdu.writeto(output_file, overwrite=True)

    print(f"Noise cube with σ={noise_sigma} saved to {output_file}")
    

def vcen_for_distance(d_kpc, l_deg, b_deg,
                      a1=1.0077, a2=0.0394, a3=0.0071,
                      theta0=236.0, R0=8.15):
    """
    Returns V_cen (km/s) that corresponds to a heliocentric distance d_kpc
    at Galactic coordinates (l_deg, b_deg), using your Brand & Blitz params.
    """
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    omega0 = theta0 / R0  # km/s/kpc

    # Galactocentric radius at (d, l, b)
    R = np.sqrt(R0**2 + (d_kpc*np.cos(b))**2 - 2*R0*d_kpc*np.cos(b)*np.cos(l))
    print(R)
    
    # Rotation curve (Brand & Blitz 1993 parameterization)
    wi = omega0 * (a1 * (R/R0)**(a2 - 1.0) + a3 * (R0 / R))
    print(wi)
    
    # Predicted LSR velocity
    V = R0 * (wi - omega0) * np.sin(l) * np.cos(b)  # km/s
    return V



# Insert into template
insert_sphere_into_template(template_file, output_file, sphere,location=location)

# Make noise-only cube
make_noise_cube(template_file, noise_file, noise_sigma=noise_sigma)
