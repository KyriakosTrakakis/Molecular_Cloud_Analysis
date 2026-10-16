#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:09:56 2025

@author: kyriakos.trakakis
"""

import numpy as np
from astropy.io import fits

source_number = 1

if source_number == 1:
    Mass     = 1 * 10 **5 # Msol
    distance = 5     # kpc
    location = (107, 102, 180)
    output_file   = "gaussian_cube_half_sigma_1_3D_newtest.fits"
elif source_number == 2:
    Mass     = 2 * 10 **4 # Msol
    distance = 2     # kpc
    location=(234, 182, 880)
    output_file   = "gaussian_cube_half_sigma_2_3D.fits"


radius   = (Mass / 228) ** (1 / 2.36)   # pc
v_width  = 1.8 * radius ** 0.27   # km / s 
dv       = 0.5   # km / s
dl       = 6     # arcsec  
X_factor = 6.52 * 10 ** 20 # K * km / s
mH       = 1.67 * 2.72 / 10 ** 24 # g
Msol     = 2 * 10 ** 33 # g

## Calculate secondary parameters

r_deg = radius / distance * 180 / np.pi / 1000
r_arcsec = r_deg * 3600
r_pixel = r_arcsec // dl
M_sol_mult = (np.pi * dl / 180 / 3600) ** 2 * (3.086 * 10 ** 21) ** 2 * X_factor * mH / Msol
conversion_factor = dv * (distance**2) * M_sol_mult

v_shape = v_width / dv * 7
r_shape = int(r_pixel)
sigma = r_pixel / 2.354
noise_sigma = 0.44
v_sigma = v_shape / 2.354

template_file = "FGN_027_13CO_23arcsec_aligned_SNR.fits"
noise_file = "027_noise_file_rms044.fits"

def make_3D_gaussian(
    shape,              # (nv, nb, nl)
    center,        # (cv, cb, cl) or None for middle
    peak,
    sigma_spatial,
    sigma_spectral,
    radius,        # spatial cutoff (pixels)
    rms,            # stochastic noise level
    normalize_to   # rescale cube sum to this value
):
    """
    Create a 3D Gaussian source with different sigmas
    along spectral (v) and spatial (b,l) axes.

    Parameters
    ----------
    shape : tuple
        (nv, nb, nl) dimensions of the output cube.
    center : tuple or None
        (cv, cb, cl) center coordinates. Defaults to cube center.
    peak : float
        Peak intensity.
    sigma_spatial : float
        Gaussian sigma in spatial pixels (b,l).
    sigma_spectral : float
        Gaussian sigma in spectral pixels (v).
    radius : float or None
        Spherical cutoff radius in spatial pixels. None = no cutoff.
    rms : float
        RMS amplitude of uniform random noise, applied only inside sphere.
    normalize_to : float or None
        If given, rescales cube so total sum equals this value.

    Returns
    -------
    cube : ndarray
        3D array (nv, nb, nl).
    """

    nv, nb, nl = shape
    v = np.arange(nv)
    b = np.arange(nb)
    l = np.arange(nl)

    # Center
    if center is None:
        cv, cb, cl = (nv - 1)/2, (nb - 1)/2, (nl - 1)/2
    else:
        cv, cb, cl = center

    # Spectral profile (1D Gaussian along v)
    spectral_profile = np.exp(-(v - cv)**2 / (2.0 * sigma_spectral**2))

    # Spatial profile (2D Gaussian in b-l plane)
    B, L = np.meshgrid(b, l, indexing="ij")
    dist2_spatial = (B - cb)**2 + (L - cl)**2
    spatial_profile = np.exp(-dist2_spatial / (2.0 * sigma_spatial**2))

    # Apply spatial cutoff
    if radius is not None:
        spatial_profile[dist2_spatial > radius**2] = 0.0

    # Full cube
    cube = peak * spectral_profile[:, None, None] * spatial_profile[None, :, :]

    # Noise inside non-zero region
    if rms > 0:
        noise = np.random.uniform(-rms, rms, size=cube.shape)
        mask = cube > 0
        cube[mask] *= (1 + noise[mask])
        
    total_sum = cube.sum()
        
    print("Pre normal peak intensity:", cube.max())
    print("Pre normal Total integrated value:", total_sum)

    # Normalize if requested
    if normalize_to is not None:
        total = cube.sum()
        if total > 0:
            cube *= normalize_to / total
            
    total_sum = cube.sum()

    return cube, total_sum

cube, total_sum = make_3D_gaussian(
    shape=(v_shape, r_shape, r_shape),
    center=None,
    peak=4.0,
    sigma_spatial= sigma / 2,
    sigma_spectral=v_sigma / 2,
    radius=r_pixel // 2,
    rms=0.25,
    normalize_to= Mass / conversion_factor
)

print("Post normal peak intensity:", cube.max())
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

insert_sphere_into_template(template_file, output_file, cube,location=location)