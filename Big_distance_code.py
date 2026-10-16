import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import subprocess
import os
from math import sin, cos, tan, radians, sqrt, pi
from os.path import join
from matplotlib.pylab import *


# Initialisation and parameters
# Defines inputs

cat_csv = pd.read_csv("/Users/kyriakos.trakakis/Downloads/ATLASGAL_CSC_distances.csv")  # Example CSV

cat_csv = cat_csv[
    (cat_csv['gal_long_peak'] >= 17) & 
    (cat_csv['gal_long_peak'] <= 47)
].reset_index(drop=True)

dataset      = "CHIMPS"
region       = "027"
isotopologue = "13CO"
SNR_lim      = 0
v_lim        = 3
v_lim2       = 5
minor_lim    = 69

if dataset == "CHIMPS" or dataset == "COHRS":
    if isotopologue == "12CO":
        dataset = "COHRS"
    elif isotopologue == "13CO":
        dataset = "CHIMPS"


mult_path = f"/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/{isotopologue}/"

cat_csv['vlsr'] = pd.to_numeric(cat_csv['vlsr'], errors='coerce')
cat_csv = cat_csv.dropna(subset=['gal_long_peak', 'gal_lat_peak', 'vlsr']).reset_index(drop=True)

base_prefix     = f"FGN_{region}_13CO_23arcsec_aligned_SNR_min_del_2_parallel_cluster"
input_filename  = base_prefix + "s.fits"
mult_file       = mult_path +f"{dataset}_{region}_{isotopologue}_23arcsec_aligned"
mult_filename   = mult_file + '.fits'
mult_sdf        = mult_file + '.sdf'
SNR_filename    = mult_path + f"{dataset}_{region}_{isotopologue}_23arcsec_aligned_SNR.fits"

clusters_path     = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/'
clusters_file     = f"FGN_{region}_23arcsec_aligned_SNR_min_del_2_parallel_clusters_table.fits"
csv_output        = mult_path +f"{dataset}_{isotopologue}_{region}_vrms_3_X_factor_masses_SNR{SNR_lim}.csv"
csv_output_sub    = mult_path +f"{dataset}_{isotopologue}_{region}_vrms_5_X_factor_masses_SNR{SNR_lim}.csv"
table             = Table.read(clusters_path + clusters_file, format='fits')


df = table.to_pandas()
df = df[(df["minor_arcsec"] > minor_lim / 2.354) & (df["sigv_kms"] > v_lim / 2.354)]
df = df.drop(['luminosity', 'volume', 'flux'], axis = 1)

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

## ---- SECTION 1 : AGAL MATCHES RECTANGLE ----

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

# Now xtract all matched _idx values
matched_ids = {match['fgn_index'] for match in matches}

# Add a 'has_match' column to df
df['has_match'] = df['_idx'].isin(matched_ids)

df_matches = pd.DataFrame(matches)
print(df_matches)
#df_matches.to_csv("FGN_37_47_matched_sources_test.csv", index=False)

most_common_distances = (
    df_matches.groupby('fgn_index')['distance']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .reset_index()
    .rename(columns={'distance': 'most_common_distance'})
)

print(most_common_distances)

# Merge most_common_distance into the main DataFrame on 'fgn_index' and '_idx'
df = df.merge(most_common_distances, how='left', left_on='_idx', right_on='fgn_index')

# Fill NaN (i.e., unmatched sources) with 0
df['AGAL_distance'] = df['most_common_distance'].fillna(0)

# Optionally drop the 'fgn_index' and 'most_common_distance' columns if not needed
df = df.drop(columns=['fgn_index', 'most_common_distance'])

## ---- SECTION 2 : BAYESIAN DISTANCES ----

cluster_ids = df['_idx']
cluster_ids = cluster_ids.to_numpy()

df_inp = pd.DataFrame({
    'Source'       : df['_idx'],
    'Long (deg)'   : df['x_cen'],
    'Lat (deg)'    : df['y_cen'],
    'vlsr (km/s)'  : df['v_cen']/1000,
    'v_rms (km/s)' : df['v_rms']/1000,
    'P': 0.5
})

# Save to .inp file with space-separated columns
output_path = '/Users/kyriakos.trakakis/Downloads/v2.4.1_bundle/'
output_file = 'sources_info.inp'

with open(output_path+output_file, 'w') as f:
    f.write('! Source Long(deg) Lat(deg) vlsr(km/s) v_rms(km/s) P\n')  # comment line
    df_inp.to_csv(f, sep=' ', index=False, header=False, float_format='%.4f')


# Path to the Fortran executable
exe_path = "/Users/kyriakos.trakakis/Downloads/v2.4.1_bundle/distance_calculator"
distance_dir='/Users/kyriakos.trakakis/Downloads/v2.4.1_bundle/'
os.chdir(distance_dir)

try:
    result = subprocess.run([exe_path], check=True, capture_output=True, text=True)
    print("Program output:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Error running Fortran program:")
    print(e.stderr)
    
def plot_distance_pdf(cluster_number,distance_dir):

    data_final     = np.loadtxt(distance_dir + f'{cluster_number}_final_distance_pdf.dat', skiprows=2)  # Skip header: "D(kpc) Probability"
    distances1     = data_final[:, 0]   # Column 1: D(kpc)
    probabilities1 = data_final[:, 1]  # Column 2: Probability

    data_spiral    = np.loadtxt(distance_dir + str(cluster_number)+'_arm_latitude_pdf.dat', skiprows=2, usecols=(0, 1))
    distances2     = data_spiral[:, 0]
    probabilities2 = data_spiral[:, 1]

    data_parallax  = np.loadtxt(distance_dir + str(cluster_number)+'_parallaxes_pdf.dat', skiprows=2)
    distances3     = data_parallax[:, 0]
    probabilities3 = data_parallax[:, 1]

    data_kinematic = np.loadtxt(distance_dir + str(cluster_number)+'_kinematic_distance_pdf.dat', skiprows=2)
    distances4     = data_kinematic[:, 0]
    probabilities4 = data_kinematic[:, 1]

    data_latitude  = np.loadtxt(distance_dir + str(cluster_number)+'_pm_bee_distance_pdf.dat', skiprows=2)
    distances5     = data_latitude[:, 0]
    probabilities5 = data_latitude[:, 1]

    data_longitude = np.loadtxt(distance_dir + str(cluster_number)+'_pm_ell_distance_pdf.dat', skiprows=2)
    distances6     = data_longitude[:, 0]
    probabilities6 = data_longitude[:, 1]
    
    arm_ranges = []
    arm_file = distance_dir + str(cluster_number) + '_arm_ranges.dat'
    with open(arm_file, 'r') as f:
        for line in f:
            if line.strip().startswith('!') or not line.strip():
                continue
            parts = line.split()
            start, end, name = float(parts[0]), float(parts[1]), parts[2]
            if start > 0 and end > 0:
                arm_ranges.append((start, end, name))

# Plotting
    plt.figure(figsize=(10, 6))
    
    ymax = max(probabilities1)  # Adjust if you'd rather base it on other curves or overall max

    for start, end, name in arm_ranges:
    # Shading (transparent and no legend entry)
        plt.axvspan(start, end, color='gray', alpha=0.3, zorder=0)

    # Label the spiral arm at the midpoint
        midpoint = (start + end) / 2
        plt.text(midpoint+2, ymax * 0.9, name, fontsize=12, ha='center', va='bottom', color='gray')
    
    plt.plot(distances2, probabilities2, label='Spiral arm distance', color='red', lw=2, linestyle='--')
    plt.plot(distances3, probabilities3, label='Parallax distance', color='blue', lw=2, linestyle='--')
    plt.plot(distances4, probabilities4, label='Kinematic distance', color='green', lw=2, linestyle='--')
    plt.plot(distances5, probabilities5, label='Proper motion b', color='cyan', lw=2, linestyle='--')
    plt.plot(distances6, probabilities6, label='Proper motion l', color='purple', lw=2, linestyle='--')
    plt.plot(distances1, probabilities1, label='Combined probability', color='black', lw=2)

    plt.xlabel('Distance (kpc)', fontsize=14)
    plt.ylabel('Probability density', fontsize=14)
    plt.title(f'Distance PDFs cluster {cluster_number}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for id in cluster_ids:
    plot_distance_pdf(id, distance_dir)
    
## ---- SECTION 3 : ROTATION CURVE ----

os.chdir(clusters_path)

# Constants
a1, a2, a3 = 1.0077, 0.0394, 0.0071  # Brand & Blitz (1993)
theta0     = 236.0              # km/s
R0         = 8.15               # kpc
omega0     = theta0 / R0
dtor       = pi / 180

# Initialize outputs
n              = len(df)
R              = np.zeros(n)
Vcalc          = np.zeros(n)
d_near         = np.zeros(n)
d_far          = np.zeros(n)
distance       = np.full(n, np.nan)
dist_method    = np.zeros(n, dtype=int)

# Step 1: Rotation Curve + Near/Far Distance Calculation
for i in range(n):
    l, b, V = df['x_cen'][i], df['y_cen'][i], df['v_cen'][i]/1000
    
    Ri = 0.5
    while True:
        Ri += 0.01
        wi = omega0 * ((a1 * (Ri/R0)**(a2 - 1)) + a3 * (R0 / Ri))
        Vtest = R0 * (wi - omega0) * sin(radians(l)) * cos(radians(b))
        if abs(Vtest - V) < 0.4:
            break
    
    R[i] = Ri
    Vcalc[i] = Vtest

    A = cos(radians(b))**2
    B = 2 * R0 * cos(radians(b)) * cos(radians(l))
    C = R0**2 - Ri**2
    delta = B**2 - 4 * A * C

    if delta <= 0:
        d1 = d2 = R0 * cos(radians(l)) / cos(radians(b))
    else:
        sqrt_delta = sqrt(delta)
        d1 = -( -B + sqrt_delta ) / (2 * A)
        d2 = -( -B - sqrt_delta ) / (2 * A)

    d_near[i], d_far[i] = min(d1, d2), max(d1, d2)

    # Outer Galaxy check
    if R[i] >= R0 or d_near[i] <= 0:
        distance[i] = d_far[i]
        dist_method[i] = 2  # Outer Galaxy

# Step 2: Tangent Point Method
# Apply only to unresolved sources (dist_method = 0)
for i in range(n):
    if dist_method[i] != 0:
        continue
    l = df['x_cen'][i]
    V = df['v_cen'][i]
    tan_vel = theta0 * sin(radians(l))
    if abs(V - tan_vel) < 10 and (d_far[i] - d_near[i]) <= 1.0:
        distance[i] = (d_near[i] + d_far[i]) / 2
        dist_method[i] = 3  # Tangent method

# Step 3: Scale Height Method
# Still unresolved sources
for i in range(n):
    if dist_method[i] != 0:
        continue
    b = df['y_cen'][i]
    z = d_far[i] * 1000 * tan(radians(b))  # pc
    if abs(z) > 120:
        distance[i] = d_near[i]
        dist_method[i] = 4  # Scale height method

# Step 4: Append Results
df['R_GC'] = R
df['d_near'] = d_near
df['d_far'] = d_far
df['distance'] = distance
df['dist_method'] = dist_method

## ---- SECTION 4 : DISTANCE SELECTION ----

def determine_distance_assignment(txt_path, d_near, d_far):
    components = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Filter lines that contain "Probability component"
    prob_lines = [line.strip() for line in lines if line.strip().startswith("Probability component")]

    for line in prob_lines:
        parts = line.split()
        try:
            dist = float(parts[3])
            prob = float(parts[5])
            components.append((dist, prob))
        except (IndexError, ValueError):
            continue

    if not components:
        raise ValueError("No valid probability components found.")

    # Get distance of component with highest probability
    best_dist, _ = max(components, key=lambda x: x[1])
    
    all_distances = [comp[0] for comp in components]
    min_distance = min(all_distances)
    max_distance = max(all_distances)

    if abs(best_dist - d_near) <= abs(best_dist - d_far):
            return 'near'
    else:
            return 'far'

prt_dir = '/Users/kyriakos.trakakis/Downloads/v2.4.1_bundle/'  # Update if needed
final_distances = []
distance_flags = []  # Optional: stores 'near' or 'far'

# Selects near or far distances

for idx, row in df.iterrows():
    obj_id = int(row['_idx'])
    prt_file = os.path.join(prt_dir, f"{obj_id}.prt")

    try:
        choice = determine_distance_assignment(prt_file, row['d_near'], row['d_far'])
        if choice == 'near':
            final_distances.append(row['d_near'])
            distance_flags.append('near')
        else:
            final_distances.append(row['d_far'])
            distance_flags.append('far')
    except Exception as e:
        print(f"Warning: Failed to process {prt_file} due to {e}")
        final_distances.append(np.nan)
        distance_flags.append('error')

df['distance'] = final_distances
df['distance_flag'] = distance_flags  # Optional for debugging
df['structure_type'] = df['structure_type'].str.decode('utf-8').str.strip() 

## ---- SECTION 5 : FLUXES AND PIXELS ----

def run_command(command, cwd=None):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
        
# --- Get the numbers of the clusters we work with ---
cluster_numbers = df['_idx']

# --- Grab the SNR mask ---
with fits.open(SNR_filename) as hdul_SNR:
    data_SNR = hdul_SNR[0].data
    header_SNR = hdul_SNR[0].header
    

mask_SNR = (data_SNR > SNR_lim)


# --- Load original data ---
with fits.open(os.path.join(clusters_path, input_filename)) as hdul:
    data = hdul[0].data
    header = hdul[0].header


starlink_path=  '/usr/local/star-2023A/bin/convert/fits2ndf'
source_command = f"source {starlink_path}"

run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {mult_filename} {mult_sdf}")

nonzero_pixels = []
total_flux     = []

# --- Loop over each number and create individual FITS files ---
for num in cluster_numbers:
    mask = (data == num)
    new_data = np.where(mask, data, 0)
    new_data = np.where(mask_SNR, new_data, 0)

    
    output_filename = f"{base_prefix}_{num}.fits"
    output_filename = os.path.join(clusters_path, output_filename)
    
    hdu = fits.PrimaryHDU(new_data, header=header)
    hdu.writeto(output_filename, overwrite=True)
    print(f"Written: {output_filename}")
    
    ndffile         = f"{base_prefix}_{num}.sdf"
    run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {output_filename} {ndffile}")
    os.system('rm -rf ' + output_filename)
    
    ndffile_2       = f"{base_prefix}_{num}_masked.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/mult IN1={mult_sdf} IN2={ndffile} OUT={ndffile_2}")
    os.system('rm -rf '+ndffile)
    
    ndffile_3       = f"{base_prefix}_{num}_data.sdf"
    run_command(f"/usr/local/star-2023A/bin/kappa/cdiv {ndffile_2} SCALAR={num} OUT={ndffile_3}")
    os.system('rm -rf '+ndffile_2)
    
  #  run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {ndffile_3} OUT={'*'}")
    
    ndffile_4           = f"{base_prefix}_{num}_integ.sdf"
    fits_file_int       = f"{base_prefix}_{num}_integ.fit"
    run_command(f"/usr/local/star-2023A/bin/kappa/collapse estimator=integ {ndffile_3} AXIS={'VRAD'} OUT={ndffile_4}")
    run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {ndffile_4} OUT={'*'}")
    os.system('rm -rf '+ndffile_3)
    os.system('rm -rf '+ndffile_4)

    print(num)
    
    clust_data, headerr = fits.getdata(fits_file_int, 0, header=True)
    clust_good          = clust_data[~isnan(clust_data)]
    clust_good          = clust_good[clust_good>=0]
    pixels_nonzero      = np.count_nonzero(clust_good)
    integ_flux          = np.sum(clust_good)
    os.system('rm -rf '+fits_file_int)
    
    if header['CUNIT3'] == 'm/s':
        integ_flux=integ_flux/1000

    total_flux.append(integ_flux)
    nonzero_pixels.append(pixels_nonzero)
    print(integ_flux, pixels_nonzero)

os.system('rm -rf ' +mult_sdf) 


df['total_flux']   = total_flux
df['pixel_number'] = nonzero_pixels 

## ---- SECTION 6 : MASS CALCULATION ----

final_distance = []
for i in range(len(df)):
    if df['has_match'][i] and df['AGAL_distance'][i] != '---':
        final_distance.append(float(df['AGAL_distance'][i]))
    else:
        final_distance.append(df['distance'][i])
        
df['final_distance'] = final_distance

df['distance_cm'] = df['final_distance'] * 3.086 * pow(10,21)
df['Mean_flux']   =df['total_flux'] / df['pixel_number']

# Constants
if isotopologue == "13CO":
    if dataset == "CHIMPS" or dataset == "COHRS":
        X_factor = 4.3 * pow(10,21)
    elif dataset == "FGN":
        X_factor = 2.295 * pow(10,21)
elif isotopologue == "12CO":
    if dataset == "CHIMPS" or dataset == "COHRS":
        X_factor = 4 * pow(10,20)
    elif dataset == "FGN":
        X_factor = 2.295 * pow(10,21) / 8.5
elif isotopologue == "C18O":
    X_factor       = 2.295 * pow(10,21) * 6.5
    
mH2            = 1.67 * 2.33 * pow(10,-24)
Msol           = 2 * pow(10,33)
pixel_area_arc = pow(1.66*3.6,2)
arc_2_sr       = pow(np.pi/180/3600,2)

df['density']=X_factor*df['Mean_flux']

# Ellipse masses
df['semi_major_deg']= df['major_arcsec'] / 7200  # Divide by 7200 to turn major to semi-major too
df['semi_minor_deg']= df['minor_arcsec'] / 7200
df['semi_major_cm'] = df['semi_major_deg'] * df['distance_cm'] * np.pi / 180 * 2.354
df['semi_minor_cm'] = df['semi_minor_deg'] * df['distance_cm'] * np.pi / 180 * 2.354
df['NH2_ellipse']   = df['density'] * np.pi * df['semi_major_cm'] * df['semi_minor_cm']
df['M_ellipse_g']   = df['NH2_ellipse'] * mH2


# Pixel area masses
df['angular_size_sr']       = df['pixel_number'] * pixel_area_arc * arc_2_sr
df['area_cm2_pixel_method'] = df['angular_size_sr'] * pow(df['distance_cm'],2)
df['NH2_pixel']             = df['area_cm2_pixel_method'] * df['density']
df['M_pixel_g']             = df['NH2_pixel'] * mH2

df['M_ellipse_Sol'] = df['M_ellipse_g'] / Msol   
df['M_pixel_Sol']   = df['M_pixel_g'] / Msol

df['area_ellipse']   = df['area_ellipse'].round(1)  # Round to 1 decimal place
df['area_exact']     = df['area_exact'].round(1)
df['major_sigma']    = df['major_sigma'].round(1)
df['minor_sigma']    = df['minor_sigma'].round(1)
df['position_angle'] = df['position_angle'].round(1)
df['radius']         = df['radius'].round(1)
df['v_cen']          = df['v_cen'].round(2)
df['v_rms']          = df['v_rms'].round(2)
df['x_cen']          = df['x_cen'].round(3)
df['y_cen']          = df['y_cen'].round(3)
df['radius_arcsec']  = df['radius_arcsec'].round(1)
df['sigv_kms']       = df['sigv_kms'].round(2)
df['xcens']          = df['xcens'].round(3)
df['ycens']          = df['ycens'].round(3)
df['vcens']          = df['vcens'].round(3)
df['major_arcsec']   = df['major_arcsec'].round(1)
df['minor_arcsec']   = df['minor_arcsec'].round(1)
df['R_GC']           = df['R_GC'].round(2)
df['d_near']         = df['d_near'].round(2)
df['d_far']          = df['d_far'].round(2)
df['distance']       = df['distance'].round(2)
df['final_distance'] = df['final_distance'].round(2)


df['distance_cm']           = df['distance_cm'].apply(lambda x: f'{x:.2e}') # Write to format 1e7
df['Mean_flux']             = df['Mean_flux'].round(2)
df['density']               = df['density'].apply(lambda x: f'{x:.2e}')
df['semi_major_deg']        = df['semi_major_deg'].round(3)
df['semi_minor_deg']        = df['semi_minor_deg'].round(3)
df['semi_major_cm']         = df['semi_major_cm'].apply(lambda x: f'{x:.2e}')
df['semi_minor_cm']         = pd.to_numeric(df['semi_minor_cm']).apply(lambda x: f'{x:.2e}')
df['NH2_ellipse']           = df['NH2_ellipse'].apply(lambda x: f'{x:.2e}')
df['M_ellipse_g']           = df['M_ellipse_g'].apply(lambda x: f'{x:.2e}')
df['M_ellipse_Sol']         = df['M_ellipse_Sol'].apply(lambda x: f'{x:.2e}')
df['angular_size_sr']       = df['angular_size_sr'].apply(lambda x: f'{x:.2e}')
df['area_cm2_pixel_method'] = df['area_cm2_pixel_method'].apply(lambda x: f'{x:.2e}')
df['NH2_pixel']             = df['NH2_pixel'].apply(lambda x: f'{x:.2e}')
df['M_pixel_g']             = df['M_pixel_g'].apply(lambda x: f'{x:.2e}')
df['M_pixel_Sol']           = df['M_pixel_Sol'].apply(lambda x: f'{x:.2e}')

df.to_csv(csv_output, index= False)

df_sub = df[df["sigv_kms"] > v_lim2 / 2.354]

#df_sub.to_csv(csv_output_sub, index= False)


