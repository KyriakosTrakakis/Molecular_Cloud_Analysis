#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:48:18 2026

@author: kyriakos.trakakis
"""

import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table

# --- Load catalogues ---
#cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/higal_CSC_distances.csv")  # Example CSV
cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/MOPRA_HiGAL/Full_Mopra/Mopra_HiGal_fits_parameters_subtracting_13CO_post_merge_5kms_properties.csv")


cat_csv = cat_csv[
    (cat_csv['GLON'] >= 250) & 
    (cat_csv['GLAT'] <= 355)
].reset_index(drop=True)

cat_csv['vlsr'] = pd.to_numeric(cat_csv['VLSR'], errors='coerce')


df = pd.read_csv('/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO/Mopra_Full_Cloud_Catalogue_cleaned_angle.csv')

# Rename or assign expected columns for clarity
# CSV (JPS) example

l_higal = cat_csv['GLON']
b_higal = cat_csv['GLAT']
v_higal = cat_csv['VLSR']
d_higal = cat_csv['DIST']


# FITS (GRS) example
l_fgn = df['x_cen']
b_fgn = df['y_cen']
v_fgn = df['v_cen'] / 1000
angle = df['position_angle']

dl = np.maximum(df['major_arcsec'] / 3600 * abs(np.cos(angle*np.pi/180)), df['minor_arcsec'] * abs(np.sin(angle*np.pi/180)) / 3600) / 2 * 2.354
db = np.maximum(df['major_arcsec'] / 3600 * abs(np.sin(angle*np.pi/180)), df['minor_arcsec'] * abs(np.cos(angle*np.pi/180)) / 3600) / 2 * 2.354
dv = df['sigv_kms'] / 2 * 2.354  # Or define your own constant window like ±1 km/s


# --- Match sources ---
matches = []

for i in range(len(df)):
    for j in range(len(cat_csv)):
        if (abs(l_higal.iloc[j] - l_fgn.iloc[i]) <= dl.iloc[i] and
            abs(b_higal.iloc[j] - b_fgn.iloc[i]) <= db.iloc[i] and
            abs(v_higal.iloc[j] - v_fgn.iloc[i]) <= dv.iloc[i]):
            matches.append({
                'fgn_index': df['Universal_ID'].iloc[i],
                'higal_name': cat_csv['DESIGNATION'].iloc[j],  # or gname_jps
                'l_higal': l_higal.iloc[j],
                'b_higal': b_higal.iloc[j],
                'v_higal': v_higal.iloc[j],
                'l_fgn': round(l_fgn.iloc[i],2),
                'b_fgn': round(b_fgn.iloc[i],2),
                'v_fgn': round(v_fgn.iloc[i],2),
                'l_distance': round((l_higal.iloc[j] - l_fgn.iloc[i]) / dl.iloc[i],2),
                'b_distance': round((b_higal.iloc[j] - b_fgn.iloc[i]) / db.iloc[i],2),
                'v_distance': round((v_higal.iloc[j] - v_fgn.iloc[i]) / dv.iloc[i],2),
                'distance': d_higal.iloc[j]
            })
    if i%100 == 0:
        print(f"{i}/{len(df)} clouds checked")


# --- Convert to DataFrame and save ---
df_matches = pd.DataFrame(matches)
print(df_matches)
df_matches.to_csv("Full_+catalogue_HiGAL_testmatch.csv", index=False)

most_common_distances = (
    df_matches.groupby('fgn_index')['distance']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .reset_index()
    .rename(columns={'distance': 'most_common_distance'})
)

print(most_common_distances)
    

# Use these to display long columns

#pd.set_option('display.max_rows', None)     # Show all rows
#pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.width', 0)           # Auto-detect width (no wrapping)
#pd.set_option('display.max_colwidth', None) # Show full column content

# Reset default configuration

#pd.reset_option("all")
