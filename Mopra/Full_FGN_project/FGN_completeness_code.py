#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:45:58 2026

@author: kyriakos.trakakis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:44:25 2026

@author: kyriakos.trakakis
"""

import subprocess
import os
import runpy
import sys
import numpy as np
import collections.abc
from matplotlib.pylab import *
from astropy.io import fits
from astropy.table import vstack, Table, hstack
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def run_command(command, cwd=None):
    """Run a command and print its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    else:
        print(result.stdout)
        
N_LOOK = 4
n_iter = 5
completeness_flux_final = []
test = "sensitivity"

if test == 'completeness':
    data_file = '/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_030_23arcsec_SNR_spline.fits'
    data_injected = '/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_030_23arcsec_SNR_injected.fits'
    
    #=== Makeclumps parameter list, completeness ===
    fwhm1 = "18"
    fwhm2 = "18"
    fwhm3 = "4"
    like_file = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_30_23arcsec_trimmed.sdf"
    pardist = "normal"
    nclump = "250"
    peak = "[6, 2]"
    shape = "ellipse"
    lbnd = "[1,4,108]"
    ubnd = "[848,845,354]"
    vgrad1 = "[0,0]"
    vgrad2 = "[0,0]"

if test == "sensitivity":
    data_file = '/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_30_23arcsec_trimmed_rms_cube.fits'
    data_injected = '/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_30_23arcsec_trimmed_rms_injected.fits'
    
    # === Makeclumps parameter list, sensitivity ===
    fwhm1 = "[8, 2]"
    fwhm2 = "[8, 2]"
    fwhm3 = "[3, 1]"
    like_file = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_30_23arcsec_trimmed.sdf"
    pardist = "normal"
    nclump = "2000"
    peak = "[3, 0.35]"
    shape = "ellipse"
    lbnd = "[1,4,108]"
    ubnd = "[848,845,354]"
    vgrad1 = "[0,0]"
    vgrad2 = "[0,0]"



makeclumps_path = "/usr/local/star-2023A/bin/cupid/makeclumps"
ndf2fits_path   = "/usr/local/star-2023A/bin/convert/ndf2fits"

def match_catalogues_full_fluxfiltered(
        t1, t2,
        l1_col="x_cen", b1_col="y_cen", v1_col="v_cen",
        l2_col="Cen1", b2_col="Cen2", v2_col="Cen3",
        id1_col="_idx", id2_col="PIDENT",
        flux1_col="flux",
        l_tol=0.07, b_tol=0.07, v_tol=2000.0):
    """
    Match two FITS tables on (l,b,v), combine full columns,
    and if multiple t1 entries map to the same t2 entry,
    keep only the t1 entry with the highest flux.
    """

    # Extract columns
    l1, b1, v1 = t1[l1_col], t1[b1_col], t1[v1_col]
    l2, b2, v2 = t2[l2_col], t2[b2_col], t2[v2_col]

    # Build KD-tree in scaled coordinate space
    coords1 = np.vstack([l1/l_tol, b1/b_tol, v1/v_tol]).T
    coords2 = np.vstack([l2/l_tol, b2/b_tol, v2/v_tol]).T

    tree = cKDTree(coords2)
    dist, idx = tree.query(coords1, k=1)

    # Scaled distance filter
    scaled_dist = np.sqrt(
        ((l1 - l2[idx]) / l_tol) ** 2 +
        ((b1 - b2[idx]) / b_tol) ** 2
    )

    v_dist = np.sqrt(((v1 - v2[idx]) / v_tol) ** 2)
    
    good = (scaled_dist < 1.0) & (v_dist < 1.0)

    # Extract matched subsets
    t1_match = t1[good]
    t2_match = t2[idx[good]]

    # Add temporary index column to allow grouping
    t1_ids = t1_match[id1_col]
    t2_ids = t2_match[id2_col]

    merged = Table()
    merged["t1_id"] = t1_ids
    merged["t2_id"] = t2_ids
    merged["flux"] = t1_match[flux1_col]

    # Build full merged rows
    t1_prefixed = t1_match.copy()
    t2_prefixed = t2_match.copy()
    t1_prefixed.rename_columns(t1_prefixed.colnames,
                               [f"{c}" for c in t1_prefixed.colnames])
    t2_prefixed.rename_columns(t2_prefixed.colnames,
                               [f"{c}" for c in t2_prefixed.colnames])

    full = hstack([t1_prefixed, t2_prefixed])

    # ---- FILTER: keep only highest-flux match per t2_id ----
    unique_t2 = {}
    for row in full:
        t2_id = row[id2_col]
        flux_val = row[flux1_col]

        if t2_id not in unique_t2:
            unique_t2[t2_id] = row
        else:
            if flux_val > unique_t2[t2_id][flux1_col]:
                unique_t2[t2_id] = row

    filtered = vstack(list(unique_t2.values()))

    return filtered

def fill_nans_with_neighbor_mean(arr):
    arr = np.asarray(arr, dtype=float)

    x = np.arange(len(arr))
    good = np.isfinite(arr)

    # Linear interpolation across NaNs
    arr_filled = np.interp(x, x[good], arr[good])

    return arr_filled

all_completeness_curves = []

for i in range(n_iter):
    name = f'50_sources_test_{i}'
    
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
    
    run_command(f"/usr/local/star-2023A/bin/convert/fits2ndf {data_file} {data_sdf}")
    
    run_command(f"/usr/local/star-2023A/bin/kappa/add {data_sdf} {name}.sdf {injected_sdf}")
    
    run_command(f"/usr/local/star-2023A/bin/convert/ndf2fits {injected_sdf} {data_injected}")
    
    os.system(f"rm -rf {name}.sdf {name}_noisy.sdf {injected_sdf} {data_sdf}")
    
    sys.path.insert(0, "/Users/kyriakos.trakakis/COLTE")
    
    scimes_data_name = os.path.splitext(data_injected)[0]
    
    runpy.run_path(
        "/Users/kyriakos.trakakis/COLTE/scimes1.py",
        init_globals={"filename": scimes_data_name}
    )
    
    if test == "completeness":
        scimes_data_name = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_030_23arcsec_SNR_injected"
    elif test == "sensitivity":
        scimes_data_name = "/Users/kyriakos.trakakis/Downloads/FUGIN_data/13CO/FGN_13CO_30_23arcsec_trimmed_rms_injected"
    
    clusters_name = scimes_data_name + "_min_del_2_parallel_clusters.fits"
    clust, h_clust = fits.getdata(clusters_name, 0, header=True)
    
    clust_good=clust[~isnan(clust)]
    clust_good=clust_good[clust_good>=0]
    clust_found=np.unique(clust_good)
    counted=np.count_nonzero(clust_found)
    if clust_found[0]==0:
        counted=counted+1
        
    table_c = Table([clust_found], names=['idx'])
    
    leaves_name = scimes_data_name + "_min_del_2_leaves.fits"
    leaf, h_leaf = fits.getdata(leaves_name, 0, header=True)
    
    l_good=leaf[~isnan(leaf)]
    l_good= l_good[l_good>=0]
    l_found=np.unique(l_good)
    counted_l=np.count_nonzero(l_found)
    if l_found[0]==0:
        counted_l=counted_l+1
        
    table_l = Table([l_found], names=['idx'])
    
    table = vstack([table_c, table_l])
        
    catalogue_name = scimes_data_name + "_min_del_2_new_catalog.fits"
    cat = Table.read(catalogue_name)
    
    mask = np.isin(cat['_idx'], table['idx'])
    
    scimes_matched_cat = cat[mask]
    
    t_makeclumps = Table.read(f"{name}_cat.FIT")
    
    matched = match_catalogues_full_fluxfiltered(scimes_matched_cat, t_makeclumps)
    
    sums_input = t_makeclumps['Sum']
    sums_matched = matched['Sum']

    bin_range = range(10, 51)  # From 10 to 30 bins
    medians_95 = []
    bins_used = []
    
    for nbins in bin_range:
        # Compute bins
        bins = np.histogram_bin_edges(sums_input, bins=nbins)

        # Compute median_sums_input per bin
        median_all = []
        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            in_all = (sums_input >= lo) & (sums_input < hi)
            if np.any(in_all):
                median_all.append(np.median(sums_input[in_all]))
            else:
                median_all.append(np.nan)
        median_all = np.array(median_all)
        median_sums_input = fill_nans_with_neighbor_mean(median_all)

        # Compute counts
        sum_counts_input, _ = np.histogram(sums_input, bins=bins)
        sum_counts_matched, _ = np.histogram(sums_matched, bins=bins)

        # Compute completeness
        with np.errstate(divide='ignore', invalid='ignore'):
            completeness_ratio = np.where(
                sum_counts_input > 0,
                sum_counts_matched / sum_counts_input,
                0
            )

        index_95 = None

        for k in range(len(completeness_ratio)):
            if completeness_ratio[k] >= 0.95:
                end = min(k + 1 + N_LOOK, len(completeness_ratio))
                future = completeness_ratio[k+1:end]

                no_collapse = not np.any(future < 0.5 * completeness_ratio[k])
                recovery = np.any(future >= 0.95)

                if no_collapse and recovery:
                    index_95 = k
                    break
              
        if index_95 is not None:
            medians_95.append(median_sums_input[index_95])
            bins_used.append(nbins)
        else:
            medians_95.append(np.nan)
            bins_used.append(nbins)
            
    medians_95 = np.array(medians_95)
    bins_used = np.array(bins_used)

    # Remove any NaNs for mean calculation
    valid_medians = medians_95[~np.isnan(medians_95)]
    median_median = np.median(valid_medians)

    # Find the bin count that gives median closest to the mean
    closest_idx = np.argmin(np.abs(medians_95 - median_median))
    best_nbins = bins_used[closest_idx]
    
    nbins = best_nbins

    sums_input = t_makeclumps['Sum']
    sums_matched = matched['Sum']
    bins = np.histogram_bin_edges(sums_input, bins=nbins)

    median_all = []
    
    for m in range(len(bins) - 1):
        # Bin limits
        lo, hi = bins[m], bins[m + 1]

        # Values in this bin
        in_all = (sums_input >= lo) & (sums_input < hi)
        in_mat = (sums_matched >= lo) & (sums_matched < hi)

        # Median for full sample
        if np.any(in_all):
            median_all.append(np.median(sums_input[in_all]))
        else:
            median_all.append(np.nan)
        

    median_all = np.array(median_all)
    
    median_sums_input = fill_nans_with_neighbor_mean(median_all)

    sum_counts_input, bin_edges = np.histogram(sums_input, bins=nbins)  # Find the right bin edges
    sum_counts_input, _ = np.histogram(sums_input, bins=bin_edges)   # Use the bin edges to find populations in each bin
    sum_counts_matched, _ = np.histogram(sums_matched, bins=bin_edges)

    with np.errstate(divide='ignore', invalid='ignore'):
        completeness_ratio = np.where(
            sum_counts_input > 0,
            sum_counts_matched / sum_counts_input,
            0
        )
        
    all_completeness_curves.append(
    (median_sums_input.copy(), completeness_ratio.copy())
)

   # index_95 = np.where(completeness_ratio>=0.95)[0][0] # old code
   
    index_95 = None
    for n in range(len(completeness_ratio)):
        if completeness_ratio[n] >= 0.95:
            end = min(n + 1 + N_LOOK, len(completeness_ratio))
            future = completeness_ratio[n+1:end]

            no_collapse = not np.any(future < 0.5 * completeness_ratio[n])
            recovery = np.any(future >= 0.95)

            if no_collapse and recovery:
                index_95 = n
                break
    iteration_tested_completeness = median_sums_input[index_95]
    completeness_flux_final.append(iteration_tested_completeness)
    
    
    plt.figure()
    # Full sample
    plt.hist(
    sums_input,
    bins=bin_edges,
    histtype='step',
    linewidth=1,
    label='Input'
)

# Subset
    plt.hist(
    sums_matched,
    bins=bin_edges,
    histtype='step',
    linewidth=1,
    label='Matched'
)

# Axis labels
    plt.xlabel('Flux (K)')
    plt.ylabel('Frequency')

# Show legend
    plt.legend()
    
    plt.figure()
    plt.plot(median_sums_input,completeness_ratio)
    plt.title(f'Run number {i+1}')
    plt.xlabel('Flux (K)')
    plt.ylabel('Recovered sources per input sources')
    
    os.system(f"rm -rf {name}.fits {name}_cat.FIT")
    os.system(f"rm -rf {catalogue_name} {clusters_name} {leaves_name} {data_injected}")
    
    plt.show()
    
    
    
print(completeness_flux_final)


def collapse_to_median(lst):
    out = []
    for x in lst:
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float).ravel()
            out.append(np.nanmedian(arr) if arr.size else np.nan)
        else:
            out.append(x)
    return out

completeness_flux_final = collapse_to_median(completeness_flux_final)


plt.figure()
plt.plot(completeness_flux_final)
plt.xlabel('Run number')
plt.ylabel('Completeness flux (K)')
plt.ylim(0, 1300)   

plt.figure(figsize=(7, 5))

for x, y in all_completeness_curves:
    plt.scatter(x, y, color='gray', alpha=0.3)

plt.xlabel('Flux (K)')
plt.ylabel('Recovered / Input')
plt.title('Completeness curves (30 iterations)')
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)

plt.show()

rows = []   # This is if you want to get all points from all iterations into one array

for median_sums, completeness in all_completeness_curves:
    rows.append(
        np.column_stack([median_sums, completeness])
    )
all_points = np.vstack(rows)

s1 = int(fwhm1)
p1 = int(peak[1])
p2 = float(peak[4:8])

np.savetxt(
    f"FGN_all_completeness_curves_10000_fwhm{s1}_peak{p1}_{p2}.csv", # 10000 sources, peak distribution [6,2]
    all_points,
    delimiter=",",
    header="median_sum,completeness",
    comments=""
)