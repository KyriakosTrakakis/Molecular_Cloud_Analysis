#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:06:55 2026

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.io.fits.hdu as fits_hdu
import astropy.io.fits.hdu.image as fits_hdu_image
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt

fits_hdu.image = fits_hdu_image

region = "040"
path   = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/"

emission_file = path + f"FGN_13CO_{region}_23arcsec.fits"
noise_file    = path + f"FGN_13CO_{region}_rms_23arcsec.fits"

output_file   = path + f"FGN_13CO_{region}_23arcsec_SNR_spline.fits"

with fits.open(emission_file) as hdul:
    emission = hdul[0].data.astype(float)
    emission_header = hdul[0].header.copy()

with fits.open(noise_file) as hdul:
    noise = np.squeeze(hdul[0].data.astype(float))
    noise_header = hdul[0].header.copy()

noise_wcs = WCS(noise_header).celestial
emission_wcs = WCS(emission_header).celestial

def align_noise_to_emission(noise, noise_header, emission, emission_header):
    """
    Regrid a 2D noise map onto the spatial (l,b) grid of a 3D emission cube
    using low-level WCS transforms.

    Assumes:
        emission.shape = (v, b, l)
        noise.shape    = (b, l)
    """

    noise = np.squeeze(noise)

    # Build celestial WCS objects
    noise_wcs = WCS(noise_header).celestial
    emission_wcs = WCS(emission_header).celestial

    ny, nx = emission.shape[-2:]

    # Target pixel grid
    y_out, x_out = np.indices((ny, nx))

    # Convert emission pixels -> world coordinates
    # origin=0 because NumPy arrays are 0-indexed
    l_world, b_world = emission_wcs.wcs_pix2world(
        x_out,
        y_out,
        0
    )

    # Convert those world coordinates -> noise-map pixels
    x_noise, y_noise = noise_wcs.wcs_world2pix(
        l_world,
        b_world,
        0
    )

    # Interpolate the noise map onto those positions
    noise_aligned = map_coordinates(
        noise,
        [y_noise, x_noise],
        order=1,
        mode="constant",
        cval=np.nan
    )

    return noise_aligned
    
    
## Just checking if they agree
print("Emission:", emission.shape)
print("Noise:", noise.shape)

target_shape = emission.shape[-2:]

noise_aligned = align_noise_to_emission(
    noise,
    noise_header,
    emission,
    emission_header
)

noise = noise_aligned

step = 10

ny, nx = noise.shape

fig, ax = plt.subplots()

im = ax.imshow(
    noise_aligned,
    origin="lower",
    aspect="auto"
)

fig.colorbar(im, ax=ax, label="RMS noise [K]")
ax.set_title("Noise map aligned to emission cube")

plt.show()


x_centres = []
y_centres = []

noise_coarse = []


x_samples = np.arange(step // 2, nx, step)
y_samples = np.arange(step // 2, ny, step)



coarse_noise = np.full((len(y_samples), len(x_samples)), np.nan)

for iy, y in enumerate(y_samples):
    for ix, x in enumerate(x_samples):
        y0, y1 = max(0, y - step // 2), min(ny, y + step // 2)
        x0, x1 = max(0, x - step // 2), min(nx, x + step // 2)

        block = noise[y0:y1, x0:x1]
        good = np.isfinite(block) & (block > 0)

        if np.any(good):
            coarse_noise[iy, ix] = np.median(block[good])

# Fill AFTER the loop
bad = ~np.isfinite(coarse_noise)
print("NaNs in coarse map:", bad.sum())

idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
coarse_noise_filled = coarse_noise[tuple(idx)]

spline = RectBivariateSpline(
    y_samples,
    x_samples,
    coarse_noise_filled,
    kx=3,
    ky=3
)

noise_interp = spline(
    np.arange(ny),
    np.arange(nx)
)

unobserved = ~np.isfinite(noise_aligned)

noise_interp[unobserved] = np.nan

fig, ax = plt.subplots()

im = ax.imshow(
    noise,
    origin="lower",
    aspect="auto"
)

fig.colorbar(im, ax=ax, label="RMS noise [K]")
ax.set_title("Original FUGIN noise map")

plt.show()


fig, ax = plt.subplots()

im = ax.imshow(
    noise_interp,
    origin="lower",
    aspect="auto"
)

fig.colorbar(im, ax=ax, label="RMS noise [K]")
ax.set_title("Spline-interpolated noise map")

plt.show()

correction = noise / noise_interp


snr_new = np.full_like(
    emission,
    np.nan,
    dtype=float
)

valid_noise = (
    np.isfinite(noise_interp) &
    (noise_interp > 0)
)

snr_new[:, valid_noise] = (
    emission[:, valid_noise]
    / noise_interp[valid_noise]
)

fits.writeto(
    output_file,
    snr_new.astype(np.float32),
    emission_header,
    overwrite=True
)

print(f"Saved {output_file}")