import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table

# --- Load catalogues ---
cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/ATLASGAL_CSC_distances.csv")  # Example CSV

cat_csv = cat_csv[
    (cat_csv['gal_long_peak'] >= 17) & 
    (cat_csv['gal_long_peak'] <= 47)
].reset_index(drop=True)


cat_csv['vlsr'] = pd.to_numeric(cat_csv['vlsr'], errors='coerce')
cat_csv = cat_csv.dropna(subset=['gal_long_peak', 'gal_lat_peak', 'vlsr']).reset_index(drop=True)



table = Table.read("/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_037_047_23arcsec_aligned_SNR_min_del_2_parallel_clusters_table.fits", format='fits')   
df = table.to_pandas()
df = df[(df["minor_arcsec"] > 69 / 2.354) & (df["sigv_kms"] >3 / 2.354)]

# Rename or assign expected columns for clarity
# CSV (JPS) example

l_atlasgal = cat_csv['gal_long_peak']
b_atlasgal = cat_csv['gal_lat_peak']
v_atlasgal = cat_csv['vlsr']
d_atlasgal = cat_csv['distance']


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
        if (abs(l_atlasgal.iloc[j] - l_fgn.iloc[i]) <= dl.iloc[i] and
            abs(b_atlasgal.iloc[j] - b_fgn.iloc[i]) <= db.iloc[i] and
            abs(v_atlasgal.iloc[j] - v_fgn.iloc[i]) <= dv.iloc[i]):
            matches.append({
                'fgn_index': df['_idx'].iloc[i],
                'atlasgal_name': cat_csv['atlas_name'].iloc[j],  # or gname_jps
                'l_atlasgal': l_atlasgal.iloc[j],
                'b_atlasgal': b_atlasgal.iloc[j],
                'v_atlasgal': v_atlasgal.iloc[j],
                'l_fgn': round(l_fgn.iloc[i],2),
                'b_fgn': round(b_fgn.iloc[i],2),
                'v_fgn': round(v_fgn.iloc[i],2),
                'l_distance': round((l_atlasgal.iloc[j] - l_fgn.iloc[i]) / dl.iloc[i],2),
                'b_distance': round((b_atlasgal.iloc[j] - b_fgn.iloc[i]) / db.iloc[i],2),
                'v_distance': round((v_atlasgal.iloc[j] - v_fgn.iloc[i]) / dv.iloc[i],2),
                'distance': d_atlasgal.iloc[j]
            })


# --- Convert to DataFrame and save ---
df_matches = pd.DataFrame(matches)
print(df_matches)
df_matches.to_csv("FGN_17_27_matched_sources.csv", index=False)

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
