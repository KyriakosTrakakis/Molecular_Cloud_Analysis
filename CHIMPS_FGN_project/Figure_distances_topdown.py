import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import pandas as pd

#image = plt.imread('Benjamin_MW_hires_bare.jpg')
image = plt.imread('Top-down_view_of_the_Milky_Way_annotated.jpg')


#cat = fits.open('/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_13CO_017_047_vrms3_mass_catalogue.fits')
#data = cat[1].data
#data  = pd.read_csv("/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/Mopra_Cloud_Mass_Catalogue_Fullinfo.csv")
data  = pd.read_csv("/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/CHIMPS_X2_FUGIN_Master_mass_catalogue_complete_redist_272_ratio326_relabel.csv")  
glon = data['l']
glat = data['b']
vlsr = data['v_cen_km/s']
dist = data['final_distance_kpc']

# Chunk of David's code I don't need
#cat_jps = fits.open('JPS_distances.fits')  
#data_jps = cat_jps[1].data
#glon_jps = data_jps['gal_long']
#glat_jps = data_jps['gal_lat']
#vlsr_jps = data_jps['velocity']
#dist_jps = data_jps['distance']

d2r = np.pi / 180

SunX = 2799
SunY = 3873 - 4  # - 26
GCY = 2795
GCX = SunX

ystart = -7.5
xstart = -4
imsize = 18

R0 = 8.15
pixsize = (R0 / (SunY - GCY))

xmin = 2799 * pixsize
xmax = -2800 * pixsize
ymin = 2799 * pixsize
ymax = -2800 * pixsize

condition = dist > 0
#CloudX = (dist[condition]) * np.sin(glon[condition] * d2r)
#CloudY = R0 - (dist[condition]) * np.cos(glon[condition] * d2r)
CloudX = ((dist[condition]) * np.sin(glon[condition] * d2r)).to_numpy()
CloudY = (R0 - (dist[condition]) * np.cos(glon[condition] * d2r)).to_numpy()

# Chunk of David's code I don't need
#condition_jps = dist_jps > 0
#CloudX_jps = (dist_jps[condition_jps]) * np.sin(glon_jps[condition_jps] * d2r)
#CloudY_jps = R0 - (dist_jps[condition_jps]) * np.cos(glon_jps[condition_jps] * d2r)

fig = plt.figure(figsize=(7, 7))
#plt.style.use('chimps') #This is not a thing, probably something specialised

xy = np.vstack([CloudX, CloudY])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
X, Y, Z = CloudX[idx], CloudY[idx], z[idx]

# Chunk of David's code I don't need
#xy_jps = np.vstack([CloudX_jps, CloudY_jps])
#z_jps = gaussian_kde(xy_jps)(xy_jps)
#idx_jps = z_jps.argsort()
#X_jps, Y_jps, Z_jps = CloudX_jps[idx_jps], CloudY_jps[idx_jps], z_jps[idx_jps]

ax = fig.add_subplot(111)
#ax.imshow(image, extent=[xmax, xmin, ymin, ymax], alpha=1.0) # This mirrors the image of the Galaxy 
ax.imshow(image, extent=[xmin, xmax, ymin, ymax], alpha=1.0)

circle1 = plt.Circle((0, 0), R0, color='w', ls='--', lw=1,
                     fill=False, zorder=1)
circle2 = plt.Circle((0, R0 / 2), R0 / 2, color='w', ls=':', lw=1,
                     fill=False, zorder=1)
ax.add_artist(circle1)
ax.add_artist(circle2)

ax.plot(0, R0, '.', mec='none', mfc='w', ms=8)
ax.plot(0, R0, 'o', mec='w', mfc='none', ms=7, lw=1.0)

#ax.plot(CloudX, CloudY, 'y.', ms=5, mfc='y') # Normally we use this bit, but I'm commenting it out for this GPT modification
has_match = data['has_match'].astype(bool)
CloudX_matched = CloudX[has_match[condition]]
CloudY_matched = CloudY[has_match[condition]]

CloudX_unmatched = CloudX[~has_match[condition]]
CloudY_unmatched = CloudY[~has_match[condition]]

# Plot matched in green
ax.plot(CloudX_matched, CloudY_matched, '+', color='lime', ms=3, label='Matched')

# Plot unmatched in magenta
ax.plot(CloudX_unmatched, CloudY_unmatched, '.', color='magenta', ms=1, label='Unmatched')

# Optional: show legend
ax.legend(loc='lower left', fontsize='small', frameon=True)
# End of GPT modification

#ax.plot(CloudX_jps, CloudY_jps, 'c.', ms=5, mfc='c') # Chunk of David's code I don't need
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([-20,-15,-10,-5, 0, 5, 10, 15,20])
ax.set_yticks([-20, -15,-10,-5, 0, 5, 10, 15, 20])
ax.tick_params(axis='y', which='both', labelleft='off', labelright='on')
ax.tick_params(axis='x', which='major', length=5, width=1., color='silver')
ax.tick_params(axis='y', which='major', length=5, width=1., color='silver')
ax.tick_params(axis='x', which='minor', width=1.)
ax.tick_params(axis='y', which='minor', width=1.)
ax.yaxis.set_label_position('right')
ax.set_xlabel(r'Galactocentric Radius [kpc]')
ax.set_ylabel(r'Galactocentric Radius [kpc]')
ax.text(0.59, 0.63, r'Scutum-Centaurus', rotation=-45, color='w', transform=ax.transAxes)
ax.text(0.73, 0.50, r'Sagittarius', rotation=70, color='w',transform=ax.transAxes)

ax.text(0.60, 0.21, r'Perseus', rotation=20, color='w', transform=ax.transAxes)
ax.text(0.22, 0.43, r'Outer', rotation=90, color='w', transform=ax.transAxes)
ax.text(0.55, 0.27, r'Local', rotation=-20, color='w', transform=ax.transAxes)

plt.tight_layout()
#plt.show()
#plt.savefig('scope_sources_faceon_mirror.png', dpi=200, bbox_inches='tight')
