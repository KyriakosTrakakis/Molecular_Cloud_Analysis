#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:06:48 2025

@author: kyriakos.trakakis
"""
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
        
# === Data and rms files ===
data_file = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_027_13CO_23arcsec_aligned.fits'
rms_aligned = '/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_027_13CO_rms_23arcsec_realigned.fits'
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



reference_sdf = data_file.replace(".fits", "_remade.sdf")
aligned_sdf   = rms_aligned.replace(".fits", ".sdf")
injected_sdf  = data_injected.replace(".fits",".sdf")

# FITS → NDF

run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {data_file} {reference_sdf}")



# NDF → FITS
cmd = "/usr/local/star-2023A/bin/convert/ndf2fits"

process = subprocess.Popen(
    f"{cmd} in={aligned_sdf} out={rms_aligned}",
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# ORIGIN prompt → we send a blank line (accept default)
process.stdin.write("\n")
process.stdin.flush()

# Cleanup optional
#os.remove(reference_sdf)
#os.remove(aligned_sdf)

run_command(f"/usr/local/star-2023A/bin/kappa/add {reference_sdf} {name}.sdf {injected_sdf}")

run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {injected_sdf} {data_injected}")


os.system(f"rm -rf {name}.sdf {name}_noisy.sdf ")

code = open("/Users/kyriakos.trakakis/COLTE/scimes1.py").read()

# Create a dictionary for variables the script should see
params = {"filename": data_injected}

#exec(code, params)
