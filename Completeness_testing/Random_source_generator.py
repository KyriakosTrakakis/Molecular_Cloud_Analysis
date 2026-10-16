#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:04:51 2025

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

template_file = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_027_13CO_23arcsec_aligned.fits"
output_file   = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/stochastic_gaussian_cube.fits"

## Base parameters
Mass     = 2 * 10 ** 4 # Msol
distance = 2     # kpc
radius   = (Mass / 228) ** (1 / 2.36)   # pc
v_width  = 1.8 * radius ** 0.27 * 2   # km / s 
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

v_shape = v_width / dv * 2
r_shape = int(r_pixel) * 2.
sigma = r_pixel / 2.354 

def stochastic_gaussian_sphere(shape=(v_shape, r_shape, r_shape), center=None, peak= 0.01, sigma = sigma / 2, radius=r_pixel//2, rms = 0.3):
    """
    Create a 3D spherical Gaussian distribution inside a cube grid.

    Parameters
    ----------
    shape : tuple of ints
        Dimensions of the cube (nx, ny, nz).
    center : None or tuple
        Center of the Gaussian. If None, it is set to the middle of the cube.
    peak : float
        Peak intensity at the center (default 7.5).
    sigma : float
        Gaussian sigma in pixels.
    radius : float or None
        Optional spherical cutoff radius (in pixels).
        If set, values outside this radius are set to 0.

    Returns
    -------
    cube : np.ndarray
        3D array containing the spherical Gaussian.
    """
    nx, ny, nz = shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Center defaults to middle
    if center is None:
        cx, cy, cz = (nx-1)/2, (ny-1)/2, (nz-1)/2
    else:
        cx, cy, cz = center

    # Squared distance
    dist2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2

    # Gaussian profile
    cube = peak * np.exp(-dist2 / (2.0 * sigma**2))

    # Apply radiua
    if radius is not None:
        cube[dist2 > radius**2] = 0.0
        
    noise = np.random.uniform(-rms, rms, size=cube.shape)
    cube = cube * (1 + noise)

    # Sum of all values
    total_sum = cube.sum()

    return cube, total_sum

cube, total = stochastic_gaussian_sphere(peak=0.1, radius=r_pixel//2)


# Convert to physical units so that sum gives Mass
conversion_factor = dv * distance**2 * M_sol_mult  # arbitrary units to match your setup
cube *= Mass / (cube.sum() * conversion_factor)

# Midpoints
mid_x, mid_y, mid_z = [s//2 for s in cube.shape]

# Let's also take an off-centre slice, e.g. z = mid_z + 5
off_z = mid_z + int(sigma)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# XY slice (through z mid)
axs[0,0].imshow(cube[:, :, mid_z], origin="lower", cmap="inferno")
axs[0,0].set_title("XY slice (z mid)")

# XY slice (through z mid+5)
#axs[0,1].imshow(cube[:, :, off_z], origin="lower", cmap="inferno")
edge_z = 7  # bottom slice
axs[0,1].imshow(cube[:, :, edge_z], origin="lower", cmap="inferno")
axs[0,1].set_title(f"XY slice (z={off_z})")

# XZ slice (y mid)
axs[1,0].imshow(cube[:, mid_y, :], origin="lower", cmap="inferno")
axs[1,0].set_title("XZ slice (y mid)")

# YZ slice (x mid)
axs[1,1].imshow(cube[mid_x, :, :], origin="lower", cmap="inferno")
axs[1,1].set_title("YZ slice (x mid)")

plt.suptitle(f"3D Gaussian Sphere (sum={total:.2f})", fontsize=14)
plt.tight_layout()
plt.show()

with fits.open(template_file) as hdul:
    header = hdul[0].header.copy()

# Ensure cube is float32 for FITS
cube_fits = cube.astype(np.float32)

# Make new PrimaryHDU with template header
#hdu = fits.PrimaryHDU(data=cube_fits, header=header)
#hdu.writeto(output_file, overwrite=True)

print(f"Saved cube to {output_file}")

total_sum = cube.sum()
print("Cube shape:", cube.shape)
print("Peak intensity:", cube.max())
print("Total integrated value:", total_sum)


def insert_cube_into_fits(template_file, output_file, sphere, location, noise_sigma = 0.44 / 1.4):
    """
    Insert a pre-generated 3D sphere (or any cube) into a FITS cube.

    Parameters
    ----------
    template_file : str
        Path to input FITS cube (l,b,v).
    output_file : str
        Path for output FITS file.
    sphere : np.ndarray
        3D array (the sphere you generated earlier).
    location : tuple (x0,y0,z0)
        Location in the template cube where the sphere center is placed (pixel coords).
    """
    # Load template
    with fits.open(template_file) as hdul:
        header = hdul[0].header.copy()
        template_data = hdul[0].data.astype(np.float32)
        template_shape = template_data.shape  # should be (v,b,l)

    # Make empty cube same as template
    new_cube = np.zeros(template_shape, dtype=np.float32)

    #Optional noise addition, normal distribution
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=new_cube.shape)
    new_cube += noise  
    
    # --- Noise estimation from first 30 velocity slices ---
    noise_region = new_cube[:30, :, :]   # v-axis slices
    noise_calc = np.nanstd(noise_region)     # standard deviation

    # Sphere size & center
    nv, nb, nl = sphere.shape
    cv, cb, cl = (nv-1)//2, (nb-1)//2, (nl-1)//2
    v0, b0, l0 = location

    # Compute slices
    v_slice = slice(v0-cv, v0-cv+nv)
    b_slice = slice(b0-cb, b0-cb+nb)
    l_slice = slice(l0-cl, l0-cl+nl)

    # Boundary check
    if v_slice.start < 0 or v_slice.stop > new_cube.shape[0]:
        raise ValueError(
        f"Sphere does not fit along velocity axis: "
        f"slice=({v_slice.start}:{v_slice.stop}), "
        f"valid range=(0:{new_cube.shape[0]})"
    )
    if b_slice.start < 0 or b_slice.stop > new_cube.shape[1]:
        raise ValueError(
        f"Sphere does not fit along b axis: "
        f"slice=({b_slice.start}:{b_slice.stop}), "
        f"valid range=(0:{new_cube.shape[1]})"
    )
    if l_slice.start < 0 or l_slice.stop > new_cube.shape[2]:
        raise ValueError(
        f"Sphere does not fit along l axis: "
        f"slice=({l_slice.start}:{l_slice.stop}), "
        f"valid range=(0:{new_cube.shape[2]})"
    )

    # Insert the sphere
    new_cube[v_slice, b_slice, l_slice] += sphere # Add a + before = if you wanna add noise

    # Save FITS
    hdu_raw = fits.PrimaryHDU(data=new_cube, header=header)
    hdu_raw.writeto(output_file, overwrite=True)
    
    # Compute SNR cube (avoid div by 0)
    snr_cube = new_cube / noise_calc

    # Save SNR cube
    snr_output = output_file.replace(".fits", "_SNR.fits")
    hdu_snr = fits.PrimaryHDU(data=snr_cube, header=header)
    hdu_snr.writeto(snr_output, overwrite=True)

    print(f"Inserted cube at {location}, saved to {output_file}")
    print("Total inserted flux:", sphere.sum())
    print(f'Input sigma {noise_sigma} gave a sigma {noise_calc}')

template_file = "FGN_027_13CO_23arcsec_aligned_SNR.fits"
output_file   = "stochastic_gaussian_cube_noise_044noise_2_peakier.fits"

# say you already generated `sphere` with shape (21,21,21)
insert_cube_into_fits(template_file, output_file, cube, location=(207, 192, 480)) # (v, b, l)
