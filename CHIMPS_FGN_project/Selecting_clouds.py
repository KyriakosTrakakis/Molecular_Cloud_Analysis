#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:10:49 2026

@author: kyriakos.trakakis
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path


# Directory containing the CSV and FITS files
data_dir = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/"
#csv_dir  = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/Raf_mask_tests/Full_region/"


for region in ('027_037', '037_047'):

    csv_file = data_dir + f"FGN_{region}_13CO_Mass_catalogue_COLTE.csv"
    fits_file = data_dir + f"FGN_{region}_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters.fits"

    output_file = data_dir + f"FGN_{region}_13CO_23arcsec_aligned_SNR_min_del_2_parallel_clusters_selected.fits"

    print(f"Processing Region {region}...")

    # --------------------------------------------------
    # 1. Read indices of clouds that we want to retain
    # --------------------------------------------------
    df = pd.read_csv(csv_file)
    
    cloud_ids = (
    df["idx"]
    .dropna()
    .astype(int)
    .to_numpy()
)

    print(f"  Keeping {len(cloud_ids)} cloud IDs")


    # --------------------------------------------------
    # 2. Read SCIMES label cube
    # --------------------------------------------------
    with fits.open(fits_file) as hdul:

        data = hdul[0].data
        header = hdul[0].header.copy()

        # --------------------------------------------------
        # 3. Find voxels belonging to selected clouds
        # --------------------------------------------------
        mask = np.isin(data, cloud_ids)

        # --------------------------------------------------
        # 4. Keep selected clouds only
        # --------------------------------------------------
        selected_data = np.where(mask, data, -1)

        # --------------------------------------------------
        # 5. Write new FITS cube
        # --------------------------------------------------
        fits.writeto(
            output_file,
            selected_data,
            header,
            overwrite=True
        )
        
    present_ids = np.intersect1d(
    cloud_ids,
    np.unique(data)
    )

    missing_ids = np.setdiff1d(
    cloud_ids,
    np.unique(data)
    )

    print(f"  Requested clouds: {len(cloud_ids)}")
    print(f"  Found in cube:     {len(present_ids)}")
    print(f"  Missing from cube: {len(missing_ids)}")
    print(f"  Selected voxels:   {np.count_nonzero(mask):,}")

    print(f"  Saved: {output_file}")
