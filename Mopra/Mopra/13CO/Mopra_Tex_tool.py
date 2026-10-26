#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:08:30 2026

@author: kyriakos.trakakis
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy import stats
import seaborn as sns
import math

## Adding comment to test sync
cube_cache = {}

#regions        = {"250-260", "260-270", "270-280", "280-290", "290-300", "300-310", "310-320", "320-330", "330-340", "340-350", "350-355"}
#backup_regions = {"310", "320", "330", "340", "350"}
#data_path      = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO/"

#regions        = {"10-20", "20-30", "30-40", "40-50"}
#backup_regions = {"20", "30", "40"}
#data_path      = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/"

regions        = {"198-208", "208-218", "218-227", "227-236"}
backup_regions = {"208", "218", "227"}
data_path      = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/FGN/12CO/"

Tex_path       = "/Users/kyriakos.trakakis/COLTE/"

#df = pd.read_csv(data_path + "Mopra_13CO_Full_Cloud_Catalogue_plusratio.csv")
#df = pd.read_csv(data_path + "Inner_FGN_13CO_X_factor_Mass_Catalogue.csv")
df = pd.read_csv(data_path + "Outer_FGN_cloud_catalogue_complete.csv")


for region in regions:
    #data_file   = data_path + f"G{region}-13CO_smallercube_05res.fits"
    #Tex_file    = Tex_path + f"colte_G{region}_smallercube_05res_snr0_Tex.fits"
    #scimes_file = data_path + f"G{region}-13CO_smallercube_05res_min_del_2_parallel_clusters.fits"
    
    #data_file   = data_path + f"FGN_13CO_{region}_23arcsec_trimmed.fits"
    #Tex_file    = Tex_path + f"colte_FGN_{region}_23arcsec_Tex.fits"
    #scimes_file = data_path + f"FGN_13CO_{region}_23arcsec_SNR_spline_min_del_2_parallel_clusters.fits"
    
    data_file   = data_path + f"FGN_{region}_12CO_23arcsec.fits"
    Tex_file    = Tex_path + f"colte_FGN_{region}_23arcsec_Tex.fits"
    scimes_file = data_path + f"FGN_{region}_12CO_23arcsec_SNR_spline_min_del_2_parallel_clusters.fits"
    
    data_cube = fits.getdata(data_file).astype(float)
    tex_cube = fits.getdata(Tex_file).astype(float)
    scimes_cube = fits.getdata(scimes_file)
    
    

    if not (
            data_cube.shape == tex_cube.shape == scimes_cube.shape
    ):
        raise ValueError(
            f"Shape mismatch in region {region}: "
            f"data={data_cube.shape}, "
            f"Tex={tex_cube.shape}, "
            f"SCIMES={scimes_cube.shape}"
        )

    cube_cache[region] = {
        "data": data_cube,
        "tex": tex_cube,
        "scimes": scimes_cube
    }

    print(
        region,
        data_cube.shape
    )
    
backup_cache = {}

for region in backup_regions:

   # data_file = data_path + f"G{region}-13CO_smallercube_05res.fits"
   # tex_file = Tex_path + f"colte_G{region}_smallercube_05res_snr0_Tex.fits"
   # scimes_file = data_path + f"G{region}-13CO_smallercube_05res_min_del_2_parallel_clusters.fits"
    
    data_file = data_path + f"FGN_13CO_{region}_23arcsec_trimmed.fits"
    tex_file    = Tex_path + f"colte_FGN_{region}_23arcsec_Tex.fits"
    scimes_file = data_path + f"FGN_13CO_{region}_23arcsec_SNR_spline_min_del_2_parallel_clusters.fits"

    if not all(os.path.exists(f) for f in
               [data_file, tex_file, scimes_file]):
        print(f"Backup region {region} unavailable")
        continue

    data_cube = fits.getdata(data_file).astype(float)
    tex_cube = fits.getdata(tex_file).astype(float)
    scimes_cube = fits.getdata(scimes_file)

    if not (data_cube.shape == tex_cube.shape == scimes_cube.shape):
        print(f"Shape mismatch in backup region {region}")
        continue

    backup_cache[region] = {
        "data": data_cube,
        "tex": tex_cube,
        "scimes": scimes_cube
    }
    
    

def get_longitude(lon):
    """
    Find the region whose starting longitude is reached
    by searching backwards from the cloud longitude.
    """

    region_starts = {
        int(region.split("-")[0]): region
        for region in regions
    }

    current_lon = int(np.floor(lon))

    while current_lon >= min(region_starts):

        if current_lon in region_starts:
            return region_starts[current_lon]

        current_lon -= 1

    return None


weighted_tex = []
n_voxels_used = []
tex_median = []

for _, row in df.iterrows():

    lon = row["x_cen"]          # change if your longitude column has another name
    cloud_id = int(row["_idx"])

    region = get_longitude(lon)

    if region not in cube_cache:
        print(
            f"Warning: no cube found for longitude {lon}, "
            f"cloud {cloud_id}"
        )
        weighted_tex.append(np.nan)
        continue

    data_cube = cube_cache[region]["data"]
    tex_cube = cube_cache[region]["tex"]
    scimes_cube = cube_cache[region]["scimes"]

    # Voxels belonging to this SCIMES cloud only
    cloud_mask = (scimes_cube == cloud_id)

    if not np.any(cloud_mask):

        print(f"Cloud {cloud_id} not found in {region}. Checking backups...")

    # Example: "300-310" -> ["300", "310"]
        backup_candidates = region.split("-")

        found = False

        for backup_region in backup_candidates:

            if backup_region not in backup_cache:
                continue

            backup_scimes = backup_cache[backup_region]["scimes"]

            backup_mask = (backup_scimes == cloud_id)

            if np.any(backup_mask):

                print(f"Cloud {cloud_id} found in backup {backup_region}")
            
                data_cube = backup_cache[backup_region]["data"]
                tex_cube = backup_cache[backup_region]["tex"]
                scimes_cube = backup_scimes

                cloud_mask = backup_mask
                found = True
                break

        if not found:
            print(f"Warning: cloud {cloud_id} not found in any backup")

            weighted_tex.append(np.nan)
            n_voxels_used.append(np.nan)
            tex_median.append(np.nan)
            continue

    # Require valid emission and Tex values
    valid = (
        cloud_mask
        & np.isfinite(data_cube)
        & np.isfinite(tex_cube)
    )

    if not np.any(valid):
        print(
            f"Warning: no valid voxels for cloud {cloud_id} "
            f"in {region}"
        )
        weighted_tex.append(np.nan)
        tex_median.append(np.nan)
        continue

    weights = data_cube[valid]
    tex_values = tex_cube[valid]

    # Optional but usually sensible:
    # ignore zero/negative emission as a weight
    positive = weights > 0

    weights = weights[positive]
    tex_values = tex_values[positive]

    if len(weights) == 0 or np.sum(weights) <= 0:
        weighted_tex.append(np.nan)
        tex_median.append(np.nan)
        continue

    tex_cloud = np.sum(
        tex_values * weights
    ) / np.sum(weights)

    weighted_tex.append(tex_cloud)
    n_voxels_used.append(len(weights))
    tex_median.append(np.median(tex_values))
    
    
df["Tex_weighted"] = weighted_tex    
df["Tex_nvox"] = n_voxels_used    
df['Tex_median'] = tex_median

p_low = 0
p_high = 80

low = df['Mass_ratio'].quantile(p_low / 100)
high = df['Mass_ratio'].quantile(p_high / 100)

mask = df['Mass_ratio'].between(low, high)
#mask  = df['Mass_ratio'] < 2

df_near = df[mask].copy()
df_far = df[~mask].copy()

Tex_near = df_near['Tex_weighted']
Tex_far = df_far['Tex_weighted']
    

Tex        = df['Tex_weighted']
Tex_near   = df_near['Tex_weighted']
Tex_far    = df_far['Tex_weighted']
Tex_median = df['Tex_median']
    
bins_near = np.linspace(
        Tex_near.min(),
        Tex_near.max(),
        int(np.sqrt(len(Tex_near)))
)

bins_far = np.linspace(
        Tex_far.min(),
        Tex_far.max(),
        int(np.sqrt(len(Tex_far)))
)

bins = np.linspace(
        Tex.min(),
        Tex.max(),
        int(np.sqrt(len(Tex)))
)

bins_median = np.linspace(
    Tex_median.min(),
    Tex_median.max(),
    int(np.sqrt(len(Tex_median)))
)

plt.figure()
plt.hist(Tex_median, bins = bins_median, edgecolor = 'k')
plt.xlabel(r'Median Tex of each cloud')
plt.ylabel('Frequency')


plt.figure()
plt.hist(Tex_near, bins = bins_near, edgecolor = 'k')
plt.xlabel(r'Cloud weighted Tex (K), clouds near ratio = 1')
plt.ylabel('Frequency')

plt.figure()
plt.hist(Tex_far, bins = bins_far, edgecolor = 'k')
plt.xlabel(r'Cloud weighted Tex (K), clouds not near ratio = 1')
plt.ylabel('Frequency')

plt.figure()
plt.hist(Tex, bins = bins, edgecolor = 'k')
plt.xlabel(r'Cloud weighted Tex (K), all clouds')
plt.ylabel('Frequency')


stats = pd.DataFrame({
    'All': Tex,
    'Near': Tex_near,
    'Far': Tex_far
}).describe()

print(stats)


# Remove NaNs
Tex_clean = Tex.dropna()
Tex_near_clean = Tex_near.dropna()
Tex_far_clean = Tex_far.dropna()

# Pairwise KS tests
ks_all_near = ks_2samp(Tex_clean, Tex_near_clean)
ks_all_far  = ks_2samp(Tex_clean, Tex_far_clean)
ks_near_far = ks_2samp(Tex_near_clean, Tex_far_clean)

print("All vs Near:")
print(f"  D = {ks_all_near.statistic:.4f}")
print(f"  p = {ks_all_near.pvalue:.4g}")

print("\nAll vs Far:")
print(f"  D = {ks_all_far.statistic:.4f}")
print(f"  p = {ks_all_far.pvalue:.4g}")

print("\nNear vs Far:")
print(f"  D = {ks_near_far.statistic:.4f}")
print(f"  p = {ks_near_far.pvalue:.4g}")


plt.figure()
ax = sns.ecdfplot(
    Tex_clean,
    label="All clouds",
    linestyle='-'
)

sns.ecdfplot(
    Tex_near_clean,
    label="Ratio near 1",
    linestyle='--',
    ax=ax
)
plt.title('All vs near')
plt.legend()

plt.xlabel("Weighted Tex (K)")
plt.ylabel('Tex CDF')


plt.figure()
ax = sns.ecdfplot(
    Tex_clean,
    label="All clouds",
    linestyle='-'
)

sns.ecdfplot(
    Tex_far_clean,
    label="Ratio not near 1",
    linestyle='--',
    ax=ax
)
plt.title('All vs far')
plt.legend()

plt.xlabel("Weighted Tex (K)")
plt.ylabel('Tex CDF')


plt.figure()
ax = sns.ecdfplot(
    Tex_near_clean,
    label="Ratio near 1",
    linestyle='-'
)

sns.ecdfplot(
    Tex_far_clean,
    label="Ratio not near 1",
    linestyle='--',
    ax=ax
)
plt.title('Far vs near')
plt.legend()

plt.xlabel("Weighted Tex (K)")
plt.ylabel('Tex CDF')