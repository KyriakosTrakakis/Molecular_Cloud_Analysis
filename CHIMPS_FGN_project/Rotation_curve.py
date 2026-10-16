import numpy as np
from astropy.io import fits
from astropy.table import Table
from math import sin, cos, tan, radians, sqrt, pi

# Load FITS table
with fits.open("FGN_027_037_23arcsec_aligned_SNR_min_del_2_parallel_clusters_table.fits") as hdul:
    data = Table(hdul[1].data)

# Constants
a1, a2, a3 = 1.0077, 0.0394, 0.0071  # Brand & Blitz (1993)
theta0 = 236.0                      # km/s
R0 = 8.15                           # kpc
omega0 = theta0 / R0
dtor = pi / 180

# Initialize outputs
n = len(data)
R = np.zeros(n)
Vcalc = np.zeros(n)
near_dist = np.zeros(n)
far_dist = np.zeros(n)
distance = np.full(n, np.nan)
dist_method = np.zeros(n, dtype=int)

# Step 1: Rotation Curve + Near/Far Distance Calculation
for i in range(n):
    l, b, V = data['x_cen'][i], data['y_cen'][i], data['v_cen'][i]/1000
    
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

    near_dist[i], far_dist[i] = min(d1, d2), max(d1, d2)

    # Outer Galaxy check
    if R[i] >= R0 or near_dist[i] <= 0:
        distance[i] = far_dist[i]
        dist_method[i] = 2  # Outer Galaxy

# Step 2: Tangent Point Method
# Apply only to unresolved sources (dist_method = 0)
for i in range(n):
    if dist_method[i] != 0:
        continue
    l = data['x_cen'][i]
    V = data['v_cen'][i]
    tan_vel = theta0 * sin(radians(l))
    if abs(V - tan_vel) < 10 and (far_dist[i] - near_dist[i]) <= 1.0:
        distance[i] = (near_dist[i] + far_dist[i]) / 2
        dist_method[i] = 3  # Tangent method

# Step 3: Scale Height Method
# Still unresolved sources
for i in range(n):
    if dist_method[i] != 0:
        continue
    b = data['y_cen'][i]
    z = far_dist[i] * 1000 * tan(radians(b))  # pc
    if abs(z) > 120:
        distance[i] = near_dist[i]
        dist_method[i] = 4  # Scale height method

# Step 4: Append Results
data['R_GC'] = R
data['near_dist'] = near_dist
data['far_dist'] = far_dist
data['distance'] = distance
data['dist_method'] = dist_method

# Save output
#data.write("FGN_027_037_kinematic_distances_all_methods.fits", overwrite=True)
