import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table

# --- Load catalogues ---
cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/ATLASGAL_CSC_distances.csv")  # Example CSV

cat_csv = cat_csv[
    (cat_csv['gal_long_peak'] >= 37) & 
    (cat_csv['gal_long_peak'] <= 47)
].reset_index(drop=True)


cat_csv['vlsr'] = pd.to_numeric(cat_csv['vlsr'], errors='coerce')
cat_csv = cat_csv.dropna(subset=['gal_long_peak', 'gal_lat_peak', 'vlsr']).reset_index(drop=True)



table = Table.read("/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_037_047_23arcsec_aligned_SNR_min_del_2_parallel_clusters_table.fits", format='fits')   
df = table.to_pandas()
df = df[(df["minor_arcsec"] > 69 / 2.354) & (df["sigv_kms"] >3 / 2.354)]

# Rename or assign expected columns for clarity
# CSV (JPS) example

l_atlasgal  = cat_csv['gal_long_peak']
b_atlasgal  = cat_csv['gal_lat_peak']
v_atlasgal  = cat_csv['vlsr']
d_atlasgal  = cat_csv['distance']
id_atlasgal = cat_csv['atlas_name'] 

# Same thing as above but uses clusters that touch the border, no range
# Saves output


def is_inside_ellipse(l_point, b_point, l0, b0, major_arcsec, minor_arcsec, pa_deg):
    """
    Check if a point (l_point, b_point) lies inside an ellipse.
    
    l0, b0: Center of ellipse (degrees)
    major_arcsec, minor_arcsec: Major and minor axis lengths (arcsec)
    pa_deg: Position angle (degrees) measured from l-axis toward b-axis
    """
    # Convert axes to degrees
    a = major_arcsec / (2 * 3600)
    b = minor_arcsec / (2 * 3600)
    
    # Convert position angle to radians
    theta = np.deg2rad(pa_deg)
    
    # Shift coordinates to ellipse center
    dl = l_point - l0
    db = b_point - b0
    
    # Rotate coordinate system
    x_prime = dl * np.cos(theta) + db * np.sin(theta)
    y_prime = -dl * np.sin(theta) + db * np.cos(theta)
    
    # Check ellipse equation
    value = (x_prime / a)**2 + (y_prime / b)**2
    
    return value <= 1

matches = []

for i_fits, (l0, b0, major_arcsec, minor_arcsec, pa_deg, fits_id, vcen_mps, v_rms) in enumerate(zip(
        df['x_cen'], df['y_cen'], df['major_arcsec']*2.354, df['minor_arcsec']*2.354, df['position_angle'],
        df['_idx'], df['v_cen'], df['v_rms']
    )):
    for i_csv,(l_at, b_at, v_at,atlas_name) in enumerate(zip(l_atlasgal, b_atlasgal, v_atlasgal, id_atlasgal)):
        if is_inside_ellipse(l_at, b_at, l0, b0, major_arcsec, minor_arcsec, pa_deg):
            matches.append({
                "atlasgal_idx": i_csv,
                "atlas_name": atlas_name,
                "atlasgal_l": l_at,
                "atlasgal_b": b_at,
                "atlasgal_vlsr": v_at,
                "fits_idx": i_fits,
                "fits_id": fits_id,
                "fits_l": l0,
                "fits_b": b0,
                "fits_vcen": vcen_mps / 1000.0,  # Convert to km/s
                "major_arcsec": major_arcsec,
                "minor_arcsec": minor_arcsec,
                "position_angle": pa_deg,
                "v_rms": v_rms
            })

print(f"Found {len(matches)} matches!")

# Optional: print matches nicely
for match in matches:
    dv= abs(match['atlasgal_vlsr']-match['fits_vcen'])/match['v_rms']
    if dv <=1:
        
        print(f"{match['atlas_name']} at (l={match['atlasgal_l']:.3f}, b={match['atlasgal_b']:.3f}, v={match['atlasgal_vlsr']:.3f}) "
              f"matched FGN {match['fits_id']} "
              f"at (l={match['fits_l']:.3f}, b={match['fits_b']:.3f}) "
              f" vcen={match['fits_vcen']:.2f} km/s")
