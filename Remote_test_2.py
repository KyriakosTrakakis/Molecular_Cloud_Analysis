#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:06:48 2025

@author: kyriakos.trakakis
"""
import subprocess
import os
import runpy
import sys

def run_command(command, cwd=None):
    """Run a command and print its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    else:
        print(result.stdout)
        
# === Data and rms files ===
data_file = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_027_13CO_23arcsec_aligned_SNR.fits'
data_injected = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_027_13CO_23arcsec_aligned_SNR_300_sources_injected.fits'
        
# === Makeclumps parameter list ===
fwhm1 = "25"
fwhm2 = "25"
fwhm3 = "4"
like_file = "makeclump_01.sdf"
pardist = "normal"
nclump = "300"
peak = "[6, 2]"
shape = "ellipse"
lbnd = "[-599,-299,-191]"
ubnd = "[601,301,193]"
vgrad1 = "[0,0]"
vgrad2 = "[0,0]"

makeclumps_path = "/usr/local/star-2023A/bin/cupid/makeclumps"
ndf2fits_path   = "/usr/local/star-2023A/bin/convert/ndf2fits"

name = '300_sources_test'

cmd = (
    "bash -c 'source /usr/local/star-2023A/etc/profile && "
    f"{makeclumps_path} "
    f"like={like_file} "
    f"pardist={pardist} "
    f"fwhm1={fwhm1} "
    f"fwhm2={fwhm2} "
    f"fwhm3={fwhm3} "
    f"vgrad1={vgrad1} "
    f"vgrad2={vgrad2} "
    f"model={name}.sdf "
    f"nclump={nclump} "
    f"out={name}_noisy.sdf "
    f"outcat={name}_cat "
    f"peak={peak} "
    f"shape={shape} "
    f"lbnd={lbnd} "
    f"ubnd={ubnd}'"
)


try:
    subprocess.run(cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f" makeclumps failed for {name}: {e}")
    

run_command(
    f"bash -c 'source /usr/local/star-2023A/etc/profile && "
    f"{ndf2fits_path} in={name} out={name}.fits'"
)



data_sdf = data_file.replace(".fits", ".sdf")
injected_sdf  = data_injected.replace(".fits",".sdf")

# FITS → NDF

run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {data_file} {data_sdf}")



# Cleanup optional
#os.remove(reference_sdf)
#os.remove(aligned_sdf)

run_command(f"/usr/local/star-2023A/bin/kappa/add {data_sdf} {name}.sdf {injected_sdf}")

run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {injected_sdf} {data_injected}")


os.system(f"rm -rf {name}.sdf {name}_noisy.sdf {injected_sdf} {data_sdf}")

sys.path.insert(0, "/Users/kyriakos.trakakis/COLTE")

scimes_data_name = os.path.splitext(data_injected)[0]

runpy.run_path(
    "/Users/kyriakos.trakakis/COLTE/scimes1.py",
    init_globals={"filename": scimes_data_name}
)
