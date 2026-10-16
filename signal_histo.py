import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from astropy.io import fits
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
from scipy.optimize import curve_fit

savefig = True

CO12, hCO12 = fits.getdata('/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_042_13CO_23arcsec_aligned.fits', 0, header=True)

CO12_in = CO12.reshape(-1)

CO12_histo, CO12_bins = np.histogram(CO12_in[~np.isnan(CO12_in)], bins=200, range=(-2,7))

fig = plt.figure(figsize=(3.5, 3.5))
#plt.style.use('chimps')

ax = fig.add_subplot(111)

def Gaussian(x, mu, sigma, amp=None):
    """
    Returns a Gaussian distribution of given mean, standard deviation and
    (optionally) amplitude.

    Arguments:
        x     : float - list or numpy array for the 'x' values
        mu    : float - mean value of the distribution
        sigma : float - standard deviation
        amp   : float - if given, gives the amplitude of the distribution. If 
                not given, the distribution is normalised.

    Returns:
        Gaussian distribution. float.
    """
    term1 = 1 / (sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sigma)**2)
    if amp is None:
        return term1 * term2
    else:
        return amp * term2
    
CO12_popt, CO12_pcov = curve_fit(Gaussian, CO12_bins[0:200]+0.05, CO12_histo)

CO12_fit = Gaussian(CO12_bins[0:200]+0.05, CO12_popt[0], CO12_popt[1], CO12_popt[2])

ax.plot(CO12_bins[0:200]+0.05, CO12_histo,'-b', lw=1)
ax.plot(CO12_bins[0:200]+0.05, CO12_fit,':b', lw=1)
ax.set_yscale('log')
ax.set_xlim(-2, 8)
ax.set_ylim(1, 2e7)
ax.yaxis.set_label_position('left')
ax.set_xlabel(r'T$^{*}_{A}$ [K]')
ax.set_ylabel(r'Number of voxels')

print('CO12', CO12_popt)

#if savefig:
#    fig.savefig('signal_histo_W49.pdf', dpi=300, bbox_inches='tight')
#    fig.savefig('signal_histo_W49.jpg', dpi=300, bbox_inches='tight')
#else:
plt.show()
