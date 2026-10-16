#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:52:23 2025

@author: kyriakos.trakakis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import subprocess
import os

def run_command(command, cwd=None):
    """Run a command and print its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    else:
        print(result.stdout)

# === Read FWHM values (should now contain 29 entries) ===
fwhm_values = np.loadtxt("/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/Fake_sources/fwhm_values.txt")

# === Generate d values ===
# first 19: 1.0 → 10.0 (step 0.5)
# next 10:  11 → 20 (step 1)
d_values = [1 + 0.5 * i for i in range(19)] + [10 + i for i in range(1, 11)]

# Quick sanity check
assert len(fwhm_values) == len(d_values) == 29, "Number of FWHM values must match d_values (29)."

# === Fixed parameters ===
fwhm3 = 7
like_file = "makeclump_01.sdf"
pardist = "uniform"
nclump = 1
peak = "[5]"
shape = "ellipse"
lbnd = "[-599,-299,-191]"
ubnd = "[601,301,193]"

makeclumps_path = "/usr/local/star-2023A/bin/cupid/makeclumps"
ndf2fits_path   = "/usr/local/star-2023A/bin/convert/ndf2fits"

# === Loop through all runs ===
for i, d in enumerate(d_values):
    fwhm = fwhm_values[i]
    fwhm1 = fwhm2 = fwhm

    # Nice consistent naming for files
    d_float = float(d)
    name = f"M1e5_r10_s10_peak5_d{int(d_float) if d_float.is_integer() else str(d_float).replace('.', '')}"

    # --- build makeclumps command ---
    cmd = (
        f"bash -c 'source /usr/local/star-2023A/etc/profile && "
        f"{makeclumps_path} like={like_file} pardist={pardist} "
        f"fwhm1=[{fwhm1}] fwhm2=[{fwhm2}] fwhm3=[{fwhm3}] "
        f"model={name}.sdf nclump={nclump} "
        f"out={name}_noisy.sdf outcat={name}_cat "
        f"peak={peak} shape={shape} lbnd={lbnd} ubnd={ubnd}'"
    )

    print(f"\n=== makeclumps for d={d:.2f}, FWHM={fwhm} ===")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f" makeclumps failed for {name}: {e}")
        continue


    # --- Convert to FITS ---
    run_command(
    f"bash -c 'source /usr/local/star-2023A/etc/profile && "
    f"{ndf2fits_path} {name}.sdf {name}.fits'"
)

    # --- Clean up ---
    os.system(f"rm -rf {name}.sdf {name}_noisy.sdf")
