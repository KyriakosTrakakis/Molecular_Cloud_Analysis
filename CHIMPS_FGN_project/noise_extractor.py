import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def calculate_std_in_range(fits_file, l_range, b_range):
    """
    Calculate the standard deviation of pixel values within the specified range.

    Parameters:
        fits_file (str): Path to the FITS file.
        l_range (tuple): Range of l-coordinates as (l_min, l_max).
        b_range (tuple): Range of b-coordinates as (b_min, b_max).

    Returns:
        float: Standard deviation of the selected pixel values.
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Assuming the data is in the primary HDU
        data = hdul[0].data

    # Check if the data is 3D
    if data.ndim != 3:
        raise ValueError("The FITS file does not contain 3D data.")

    # Extract the specified range
    l_min, l_max = l_range
    b_min, b_max = b_range
    selected_data = data[:, b_min:b_max+1, l_min:l_max+1]

    # Flatten the selected data and calculate the standard deviation
    std_dev = np.nanstd(selected_data)

    return std_dev

def calculate_std_for_multiple_ranges(fits_file, l_start, l_end, b_start, b_end, step, l_steps, b_steps):
    """
    Calculate the standard deviation for multiple ranges in a grid-like pattern.

    Parameters:
        fits_file (str): Path to the FITS file.
        l_start (int): Starting l-coordinate.
        l_end (int): Ending l-coordinate.
        b_start (int): Starting b-coordinate.
        b_end (int): Ending b-coordinate.
        step (int): Step size for increasing the ranges.
        l_steps (int): Number of longitude steps.
        b_steps (int): Number of latitude steps.

    Returns:
        list: A list of tuples containing the range, the calculated standard deviation,
              and the corresponding mean world coordinates.
    """
    results = []

    # Open the FITS file to read WCS and header
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        wcs = WCS(header)

    # Debug: Print WCS axis types
    print("WCS Axis Types:", header['CTYPE1'], header['CTYPE2'], header['CTYPE3'])

    for b_step in range(b_steps):
        for l_step in range(l_steps):
            # Calculate ranges
            l_range = (l_start + l_step * 2 * step, l_end + l_step * 2 * step)
            b_range = (b_start + b_step * step, b_end + b_step * step)

            # Calculate standard deviation for the current range
            std_dev = calculate_std_in_range(fits_file, l_range, b_range)

            # Calculate the mean pixel coordinates
            mean_l = (l_range[0] + l_range[1]) / 2
            mean_b = (b_range[0] + b_range[1]) / 2

            # Convert mean pixel coordinates to world coordinates
            mean_world = wcs.pixel_to_world_values([[mean_l, mean_b, 0]])[0]

            # Extract l and b from the world coordinates
            mean_world_l = round(mean_world[0], 3)  # l (longitude)
            mean_world_b = round(mean_world[1], 3)  # b (latitude)

            results.append((l_range, b_range, std_dev, mean_world_l, mean_world_b))

            print(f"l_range: {l_range}, b_range: {b_range}, std_dev: {std_dev}, "
                  f"mean_world_l: {mean_world_l}, mean_world_b: {mean_world_b}")

    return results

# Example usage
#fits_file = "/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/CHIMPS_027_037_13CO_23arcsec_aligned.fits"  # Replace with your FITS file path
fits_file = "/Users/kyriakos.trakakis/Downloads/Mopra_FGN_project/Mopra/13CO/G320-13CO_smallercube_05res.fits"
l_start, l_end = 50, 250  # Initial l-coordinate range
b_start, b_end = 50, 250  # Initial b-coordinate range
v_start, v_end = 0, 40
step = 300  # Step size for increasing the ranges
l_steps = 10  # Number of longitude steps
b_steps = 2   # Number of latitude steps

results = calculate_std_for_multiple_ranges(fits_file, l_start, l_end, b_start, b_end, step, l_steps, b_steps)

# Find the maximum standard deviation
max_result = max(results, key=lambda x: x[2])  # x[2] is the standard deviation
max_l_range, max_b_range, max_std_dev, max_world_l, max_world_b = max_result

print(f"Maximum standard deviation: {max_std_dev}")
print(f"Corresponding l_range: {max_l_range}")
print(f"Corresponding b_range: {max_b_range}")
print(f"Mean world coordinates: l = {max_world_l}, b = {max_world_b}")



def calculate_std_in_range_3D(fits_file, l_range, b_range, v_range):
    """
    Calculate the standard deviation of pixel values within the specified range.

    Parameters:
        fits_file (str): Path to the FITS file.
        l_range (tuple): Range of l-coordinates as (l_min, l_max).
        b_range (tuple): Range of b-coordinates as (b_min, b_max).

    Returns:
        float: Standard deviation of the selected pixel values.
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Assuming the data is in the primary HDU
        data = hdul[0].data

    # Check if the data is 3D
    if data.ndim != 3:
        raise ValueError("The FITS file does not contain 3D data.")

    # Extract the specified range
    l_min, l_max = l_range
    b_min, b_max = b_range
    v_min, v_max = v_range
    selected_data = data[v_min:v_max+1, b_min:b_max+1, l_min:l_max+1]

    # Flatten the selected data and calculate the standard deviation
    std_dev = np.nanstd(selected_data)

    return std_dev

l_range_3D = [l_start, l_end]
b_range_3D = [b_start, b_end]
v_range_3D = [v_start, v_end]

def calculate_cube_noise(fits_file, v_range):
    """
    Calculate the standard deviation of pixel values within the specified range.

    Parameters:
        fits_file (str): Path to the FITS file.
        l_range (tuple): Range of l-coordinates as (l_min, l_max).
        b_range (tuple): Range of b-coordinates as (b_min, b_max).

    Returns:
        float: Standard deviation of the selected pixel values.
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Assuming the data is in the primary HDU
        data = hdul[0].data

    # Check if the data is 3D
    if data.ndim != 3:
        raise ValueError("The FITS file does not contain 3D data.")

    # Extract the specified range
    v_min, v_max = v_range
    selected_data = data[v_min:v_max+1,:,:]

    # Flatten the selected data and calculate the standard deviation
    std_dev = np.nanstd(selected_data)

    return std_dev

