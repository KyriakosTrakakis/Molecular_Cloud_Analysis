#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:51:27 2025

@author: kyriakos.trakakis
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from matplotlib.ticker import MultipleLocator

# Load FITS file
fits_path = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/Integrated clusters/FGN 27-37/FGN_027_037_13CO_23arcsec_aligned_SNR_min_del_2_parallel_cluster_17423_integ.fit"
#fits_path = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/Integrated clusters/FGN 17-27/FGN_017_027_13CO_23arcsec_aligned_SNR_min_del_2_parallel_cluster_2452_integ.fit"
hdul = fits.open(fits_path)
hdu = hdul[0]
data = hdu.data.squeeze()
header = hdu.header
wcs = WCS(header)
wcs_2d = wcs.celestial

# Ellipse parameters 17423
major_axis = 173.72  # arcsec
minor_axis = 87.74   # arcsec
pa = 53.919090618899034          # degrees
l = 30.220557796250574           # degrees
b = 0.27187742251388025          # degrees

# Ellipse parameters 2452
#major_axis = 1069.85  # arcsec
#minor_axis = 405.5   # arcsec
#pa = 145.76         # degrees
#l = 24.84           # degrees
#b = -0.051          # degrees

zoom_margin_factor = 3  # how much to zoom out around ellipse size
pa = (180 - pa) % 180


# Convert ellipse axes from arcsec to degrees
major_deg = major_axis / 3600 * 2.354
minor_deg = minor_axis / 3600 * 2.354

# === Create WCS plot ===
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111, projection=wcs_2d)

# Show the image
img = ax.imshow(data, origin="lower", cmap="inferno")

# Plot ellipse in world coordinates (l, b)
ellipse = Ellipse((l, b), width=major_deg, height=minor_deg,
                  angle=pa, edgecolor='cyan', facecolor='none', lw=2, transform=ax.get_transform('world'))
ax.add_patch(ellipse)

# Define zoom region and convert to pixel space
x_margin = major_deg * zoom_margin_factor
y_margin = minor_deg * zoom_margin_factor
l_min, l_max = l - x_margin, l + x_margin
b_min, b_max = b - y_margin, b + y_margin

# Convert (l, b) zoom bounds to pixel coordinates
pix_bounds = wcs_2d.wcs_world2pix([[l_min, b_min], [l_max, b_max]], 0)
(x_min_pix, y_min_pix), (x_max_pix, y_max_pix) = pix_bounds

# Set zoomed-in pixel limits
ax.set_xlim(x_min_pix, x_max_pix)
ax.invert_xaxis()
ax.set_ylim(y_min_pix, y_max_pix)

# Add axis labels and grid
ax.coords[0].set_major_formatter('d.dd')
ax.coords[1].set_major_formatter('d.dd')
ax.coords[0].set_axislabel("Galactic Longitude (deg)")
ax.coords[1].set_axislabel("Galactic Latitude (deg)")
ax.set_title("Zoomed Moment Map with Ellipse", fontsize="x-large")
plt.tight_layout()
plt.show()
# plt.savefig("Ellipse_on_map.png")
