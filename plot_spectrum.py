
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

plt.ion() # Interactive plotting on

#plt.style.use(['chimps', 'pub'])  # Latex font for publication
#plt.style.use(['chimps'])  # Non latex font is quicker

hdu = fits.open('/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/Low_res/G270-280-12CO_giantcube.fits')

data = hdu[0].data
header = hdu[0].header
wcs3 = WCS(header)
wcs2 = wcs3.celestial


def velaxis(header):
    """
    Purpose: return the velocity axis from a given header
    """
    NAX = header['NAXIS3']
    CDELT = header['CDELT3']
    CRPIX = header['CRPIX3']
    CRVAL = header['CRVAL3']
    vaxis = CDELT * (np.arange(NAX) - CRPIX + 1) + CRVAL

    return vaxis * u.Unit(header['BUNIT'])


vax = velaxis(header)

glon, glat = [277.4999, -0.838]  # l, b
#12CO_HIGALBM277.4499-0.8380_good_spectrum.fits

# Convert to integer pixel coors
xcoord, ycoord = wcs2.all_world2pix(glon, glat, 0)
xidx = int(xcoord)
yidx = int(ycoord)

interval=2
## This is a modification to take a region around the pixel instead of the pixel
x_start, x_end = xidx - interval, xidx + interval # Define the x range
y_start, y_end = yidx - interval, yidx + interval  # Define the y range

## This is another modification that makes a circular region around the point
radius = 1  # Radius in pixels
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2

# Apply the mask to the data around the target pixel
x_start, x_end = xidx - radius, xidx + radius
y_start, y_end = yidx - radius, yidx + radius
subset = data[:, y_start:y_end+1, x_start:x_end+1]

# Ensure the mask fits within the subset
masked_region = subset * mask[np.newaxis, :, :]
#spectrum = np.mean(masked_region, axis=(1, 2))  # Mode 2, sphere of radius = radius


# Subset the data within the box
region = data[:, y_start:y_end+1, x_start:x_end+1]

# Average spectrum across the box
#spectrum = np.mean(region, axis=(1, 2))  # Mode 1, rectangle with side = interval

# Extract spectrum
spectrum = data[:, yidx+interval, xidx+interval] 

fig = plt.figure(figsize=(12, 4))
plt.subplots_adjust(bottom=0.20, top=0.90)

ax = fig.add_subplot(111)
ax.set_title(fr'PGCC G224.34-2.00')

ax.step(vax, spectrum)
ax.set_xlabel(r'Velocity [km$\,$s$^{-1}$]')
ax.set_ylabel(r'$T_\mathrm{A}^{*}$ [K]')
ax.minorticks_off()

with fits.open('/Users/kyriakos.trakakis/Desktop/Temp/12CO_HIGALBM250.2784-0.9892_spec.fits') as hdul:
    #Tmb = hdul[0].data
    hd2 = hdul[0].header
header2 = hd2

with fits.open('/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/Low_Res/C18O_HIGALBM279.4687-0.9055_spec.fits') as hdul:
    #Tmb = hdul[0].data
    hd3 = hdul[0].header
header3 = hd3

assert len(spectrum) == len(vax), "Spectrum and velocity axis must have the same length"

# Update the header to reflect the 1D data
header2['NAXIS'] = 1
header2['NAXIS1'] = len(spectrum)
header2['CDELT1'] = (vax[1] - vax[0]).value  # Velocity step
header2['CRPIX1'] = 1                        # Reference pixel
header2['CRVAL1'] = vax[0].value             # Reference value
header2['CUNIT1'] = str(vax.unit)            # Unit of the axis

# Create a primary HDU
hdu = fits.PrimaryHDU(data=spectrum, header=header3)

# Save to a FITS file
output_filename = "12CO_HIGALBM277.4499-0.8380_good_spectrum.fits"
#hdu.writeto(output_filename, overwrite=True)

#plt.savefig('PGCCspectrum.pdf')
