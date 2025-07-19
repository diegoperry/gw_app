#%% Imports
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%###############################################################################
##################################  FUNCTIONS   ##################################
##################################################################################

# Function to reduce data by averaging over time period
def time_avg_reduce(data_dict, day_avg):
    """
    Splits time series data at large gaps and averages over a given time period.

    Parameters:
        data_dict (dict): Dictionary containing astrometric data.
        day_avg (float): Averaging period in days.

    Returns:
        dict: A new dictionary with the averaged time and the averaged data arrays for the used keys.
    """

    # Extract the time array
    time = data_dict["time"].copy()

    # Initialize lists to hold the data arrays and their corresponding keys
    data_arrays = []
    used_keys = []

    # Loop through the data_dict and collect the relevant 2D arrays
    for key, val in data_dict.items():
        if key == "time":
            continue
        if val.ndim == 2 and val.shape[1] == time.shape[0]:
            data_arrays.append(val)
            used_keys.append(key)


    # Define time interval threshold for splitting (e.g., 2x median interval)
    gap_threshold = 2 * np.median(np.diff(time))

    # Compute time differences and find large gaps
    dt = np.diff(time)
    gap_indices = np.where(dt > gap_threshold)[0] + 1  # Find gap positions

    # Split data based on gaps
    split_time = np.split(time, gap_indices)
    split_data_arrays = [np.split(arr, gap_indices, axis=1) for arr in data_arrays]

    # Initialize lists for storing averaged results
    averaged_time = []
    averaged_data_arrays = [[] for _ in data_arrays]

    # Process each chunk
    for t_chunk, *data_chunks in zip(split_time, *split_data_arrays):
        if len(t_chunk) == 0:
            continue  # Skip empty chunks

        # Determine dynamic chunk size for this segment
        chunk_size = np.searchsorted(t_chunk, t_chunk[0] + day_avg, side='right')

        # Compute full chunks
        num_full_chunks = len(t_chunk) // chunk_size
        remaining = len(t_chunk) % chunk_size

        # Compute mean for full chunks
        full_chunk_avg_time = t_chunk[:num_full_chunks * chunk_size].reshape(num_full_chunks, chunk_size).mean(axis=1)
        full_chunk_avg_data = [
            arr[:, :num_full_chunks * chunk_size].reshape(arr.shape[0], num_full_chunks, chunk_size).mean(axis=2)
            for arr in data_chunks
        ]

        # Handle remaining elements
        if remaining > 0:
            last_chunk_avg_time = t_chunk[num_full_chunks * chunk_size:].mean()
            last_chunk_avg_data = [arr[:, num_full_chunks * chunk_size:].mean(axis=1, keepdims=True) for arr in data_chunks]

            # Append last chunk separately
            full_chunk_avg_time = np.hstack([full_chunk_avg_time, last_chunk_avg_time])
            full_chunk_avg_data = [np.hstack([full_arr, last_arr]) for full_arr, last_arr in zip(full_chunk_avg_data, last_chunk_avg_data)]

        # Store results
        averaged_time.append(full_chunk_avg_time)
        for avg_list, avg_arr in zip(averaged_data_arrays, full_chunk_avg_data):
            avg_list.append(avg_arr)

    # Concatenate results from all chunks
    final_time = np.hstack(averaged_time)
    final_data_arrays = [np.hstack(avg_list) for avg_list in averaged_data_arrays]

    # Build the new data_dict with averaged results
    red_data_dict = data_dict.copy()
    red_data_dict["time"] = final_time
    for key, avg_arr in zip(used_keys, final_data_arrays):
        red_data_dict[key] = avg_arr

    return red_data_dict

# Plot x and y centroids of the i'th star vs time
def plot_star_centroids(data_dict, i, data_type="raw", fig=None, ax=None, **plot_kwargs):
    """
    Plot the x and y centroids of the i'th star vs time.

    Parameters:
        data_dict (dict): Dictionary containing astrometric data.
        i (int): Index of the star to plot.
        data_type (str): Type of data to plot ("raw", "res", "poscorr").
        fig (matplotlib.figure.Figure, optional): Existing figure to add the plot to. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Existing axes to add the plot to. Defaults to None.
        **plot_kwargs: Additional keyword arguments for customization (e.g., label, color, marker, size).

    Returns:
        fig, ax: The figure and axes objects for further plotting or modification.
    """

    # Set the data and units based on the selected data_type
    time = data_dict["time"].copy()
    if data_type == "raw":
        data_x = data_dict["rawx"].copy()
        data_y = data_dict["rawy"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    elif data_type == "res":
        data_x = data_dict["res_x"].copy()
        data_y = data_dict["res_y"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    elif data_type == "poscorr":
        data_x = data_dict["poscorrx"].copy()
        data_y = data_dict["poscorry"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Choose from 'raw', 'res', or 'poscorr'.")

    # Calculate the average for the selected data
    avg_x = np.mean(data_x[i])
    avg_y = np.mean(data_y[i])

    # If no figure or axes are provided, create new ones without separation between the subplots
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})  # hspace=0 removes space between the subplots

    # Plot the x and y positions of the i'th star vs time with additional plot arguments
    ax[0].scatter(time, data_x[i] - avg_x, **plot_kwargs)
    ax[0].set_ylabel(f"X [{unit_x}]")
    ax[0].tick_params(axis="x", which="both", direction='in')
    ax[0].legend()

    ax[1].scatter(time, data_y[i] - avg_y, **plot_kwargs)
    ax[1].set_xlabel("Time [days]")
    ax[1].set_ylabel(f"Y [{unit_y}]")
    ax[1].legend()

    plt.suptitle(f"Centroids of KICID: {data_dict['kicids'][i]}")

    return fig, ax

# Plot x and y centroids of all stars at a given time
def plot_all_star_centroids(data_dict, t_indx, data_type="raw", fig=None, ax=None, **plot_kwargs):
    """
    Plot the x and y centroids of all stars at a given time.

    Parameters:
        data_dict (dict): Dictionary containing astrometric data.
        t (int): Index of the time step to plot.
        data_type (str): Type of data to plot ("raw", "res", "poscorr").
        fig (matplotlib.figure.Figure, optional): Existing figure to add the plot to. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Existing axes to add the plot to. Defaults to None.
        **plot_kwargs: Additional keyword arguments for customization (e.g., label, color, marker, size).

    Returns:
        fig, ax: The figure and axes objects for further plotting or modification.
    """

    # Set the data and units based on the selected data_type
    if data_type == "raw":
        data_x = data_dict["rawx"].copy()
        data_y = data_dict["rawy"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    elif data_type == "res":
        data_x = data_dict["res_x"].copy()
        data_y = data_dict["res_y"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    elif data_type == "res_fit":
        data_x = data_dict["res_x_fit"].copy()
        data_y = data_dict["res_y_fit"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    elif data_type == "poscorr":
        data_x = data_dict["poscorrx"].copy()
        data_y = data_dict["poscorry"].copy()
        unit_x = "pixels"
        unit_y = "pixels"
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Choose from 'raw', 'res', or 'poscorr'.")
    
    # If no figure or axes are provided, create new ones without separation between the subplots
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    # Plot the x y positions of all stars
    ax.scatter(data_x[:,t_indx], data_y[:,t_indx], **plot_kwargs)

    return fig, ax

# Function to unpack fit parameters
def unpack_params(params, num_epochs, num_stars, pm=False):
    """
    Unpack model parameters into arrays of transformation parameters and proper motion parameters.

    Inputs:
    - params: array of model parameters, structured as follows:
        [param1_epoch1, param1_epoch2, ..., param1_epochN, param2_epoch1, ..., param2_epochN, ...]
        if pm=True, the proper motion parameters are appended at the end of the array as:
        [mu_xi_star1, mu_xi_star2, ..., mu_xi_starN, mu_eta_star1, mu_eta_star2, ..., mu_eta_starN]
    - num_epochs: number of epochs
    - num_stars: number of stars
    - pm: whether to include proper motion parameters

    Returns:
    - params: array of transformation parameters (shape: (num_params, num_epochs))
    - mu_xi: array of proper motion parameters for xi (shape: (num_stars,))
    - mu_eta: array of proper motion parameters for eta (shape: (num_stars,))
    """

    if pm:
        mu_xi, mu_eta = np.array(params[-2*num_stars:]).reshape(2, num_stars)
        params = params[:-2*num_stars]
    else:
        mu_xi = mu_eta = np.zeros(num_stars)

    params = np.array(params).reshape(-1, num_epochs)

    return params, mu_xi, mu_eta

# >>> Models for fitting the data <<<

def model_1(x, y, delta_t, params):
    """
    Generic rotation model + shift:
    xi = a * x + b * y + c
    eta = -b * x + a * y + f

    Parameters:
    - x, y: arrays of star positions at each epoch (shape: can be (N_stars, N_epochs) or (N_epochs,) or (N_stars,))
    - delta_t: array of time differences between epochs (shape: (N_epochs,) or a single value)
    - params: array of model parameters (shape: 6*N_epochs), ordered as follows:
        - 4 parameters for each of the N_epochs: a, b, c, d

    Returns:
    - xi, eta: arrays of transformed star positions at each epoch (shape: (N_stars, N_epochs))
    """
    
    # Ensure delta_t is a 1D array
    delta_t = np.atleast_1d(delta_t)
    num_epochs = len(delta_t)

    # If x and y were provided as 1D arrays, reshape them to 2D arrays:
    if x.ndim == 1:
        # (x,y) are 1D arrays for a single star at multiple epochs or for multiple stars at a single epoch
        x = x.reshape(1, -1) if x.shape[0] == num_epochs else x.reshape(-1, 1)
        y = y.reshape(1, -1) if y.shape[0] == num_epochs else y.reshape(-1, 1)

    # Unpack transformation parameters
    a, b, c, d = np.array(params).reshape(4,num_epochs)

    # Compute xi and eta at each epoch
    xi = np.array([a[t] * x[:, t] + b[t] * y[:, t] + c[t] for t in range(num_epochs)])
    eta = np.array([-b[t] * x[:, t] + a[t] * y[:, t] + d[t] for t in range(num_epochs)])

    return xi.T, eta.T  # Shape: (num_stars, num_epochs)

def model_2(x, y, delta_t, params):
    """
    Generic rotation model + shift and proper motion correction:
    xi = a * x + b * y + c - mu_xi * delta_t
    eta = -b * x + a * y + d - mu_eta * delta_t

    Parameters:
    - x, y: arrays of star positions at each epoch (shape: can be (N_stars, N_epochs) or (N_epochs,) or (N_stars,))
    - delta_t: array of time differences between epochs (shape: (N_epochs,) or a single value)
    - params: array of model parameters (shape: 6*N_epochs + 2*N_stars), ordered as follows:
        - 4 parameters for each of the N_epochs: a, b, c, d
        - 1 proper motion parameter for each of the N_stars: mu_xi
        - 1 proper motion parameter for each of the N_stars: mu_eta
    """

    # Ensure delta_t is a 1D array
    delta_t = np.atleast_1d(delta_t)
    num_epochs = len(delta_t)

    # If x and y were provided as 1D arrays, reshape them to 2D arrays:
    if x.ndim == 1:
        # (x,y) are 1D arrays for a single star at multiple epochs or for multiple stars at a single epoch
        x = x.reshape(1, -1) if x.shape[0] == num_epochs else x.reshape(-1, 1)
        y = y.reshape(1, -1) if y.shape[0] == num_epochs else y.reshape(-1, 1)

    num_stars = x.shape[0]

    # Extract proper motion parameters (one per star)
    mu_xi, mu_eta = np.array(params[-2*num_stars:]).reshape(2, num_stars)

    # Unpack transformation parameters
    params = params[:-2*num_stars]
    a, b, c, d = np.array(params).reshape(4, num_epochs)

    # Compute xi and eta at each epoch
    xi = np.array([a[t] * x[:, t] + b[t] * y[:, t] + c[t] - mu_xi * delta_t[t] for t in range(num_epochs)])
    eta = np.array([-b[t] * x[:, t] + a[t] * y[:, t] + d[t] - mu_eta * delta_t[t] for t in range(num_epochs)])

    return xi.T, eta.T  # Shape: (num_stars, num_epochs)

def model_3(x, y, delta_t, params):
    """
    Schmidt–Cassegrain design model:
    xi = a * x + b * y + c * x * y + d * x^2 + e * y^2 + f * x * (x^2 + y^2) + g
    eta = ap * x + bp * y + cp * x * y + dp * x^2 + ep * y^2 + fp * y * (x^2 + y^2) + gp

    Parameters:
    - x, y: arrays of star positions at each epoch (shape: (N_stars, N_epochs) or (N_epochs,) or (N_stars,))
    - delta_t: array of time differences between epochs (shape: (N_epochs,) or a single value)
    - params: array of model parameters (shape: 14*N_epochs), ordered as:
        - 14 parameters for each of the N_epochs: a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp

    Returns:
    - xi, eta: arrays of transformed star positions at each epoch (shape: (N_stars, N_epochs))
    """

    # Ensure delta_t is a 1D array
    delta_t = np.atleast_1d(delta_t)
    num_epochs = len(delta_t)

    # If x and y are 1D arrays, reshape them to 2D arrays:
    if x.ndim == 1:
        x = x.reshape(1, -1) if x.shape[0] == num_epochs else x.reshape(-1, 1)
        y = y.reshape(1, -1) if y.shape[0] == num_epochs else y.reshape(-1, 1)

    # Unpack transformation parameters
    params = np.array(params).reshape(14, num_epochs)
    a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp = params

    # Compute xi and eta at each epoch
    xi = np.array([
        a[t] * x[:, t] + b[t] * y[:, t] + c[t] * x[:, t] * y[:, t] +
        d[t] * x[:, t] ** 2 + e[t] * y[:, t] ** 2 +
        f[t] * x[:, t] * (x[:, t] ** 2 + y[:, t] ** 2) + g[t]
        for t in range(num_epochs)
    ])

    eta = np.array([
        ap[t] * x[:, t] + bp[t] * y[:, t] + cp[t] * x[:, t] * y[:, t] +
        dp[t] * x[:, t] ** 2 + ep[t] * y[:, t] ** 2 +
        fp[t] * y[:, t] * (x[:, t] ** 2 + y[:, t] ** 2) + gp[t]
        for t in range(num_epochs)
    ])

    return xi.T, eta.T  # Shape: (num_stars, num_epochs)


def model_4(x, y, delta_t, params):
    """
    Schmidt–Cassegrain design model with star-specific proper motion correction:
    xi = a * x + b * y + c * x * y + d * x^2 + e * y^2 + f * x * (x^2 + y^2) + g - mu_xi * delta_t
    eta = ap * x + bp * y + cp * x * y + dp * x^2 + ep * y^2 + fp * y * (x^2 + y^2) + gp - mu_eta * delta_t

    Parameters:
    - x, y: arrays of star positions (in mas) at each epoch (shape: can be (N_stars, N_epochs) or (N_epochs,) or (N_stars,))
    - delta_t: array of time differences between epochs (shape: (N_epochs,) or a single value)
    - params: array of model parameters (shape: 14*N_epochs + 2*N_stars), ordered as follows:
        - 7 parameters for each of the N_epochs: a, b, c, d, e, f, g
        - 7 parameters for each of the N_epochs: ap, bp, cp, dp, ep, fp, gp
        - 1 proper motion parameter for each of the N_stars: mu_xi (mas/yr)
        - 1 proper motion parameter for each of the N_stars: mu_eta (mas/yr)
    """
    
    # Ensure delta_t is always a 1D array
    delta_t = np.atleast_1d(delta_t)
    num_epochs = len(delta_t)

    # If x and y were provided as 1D arrays, reshape them to 2D arrays:
    if x.ndim == 1:
        # (x,y) are 1D arrays for a single star at multiple epochs or for multiple stars at a single epoch
        x = x.reshape(1, -1) if x.shape[0] == num_epochs else x.reshape(-1, 1)
        y = y.reshape(1, -1) if y.shape[0] == num_epochs else y.reshape(-1, 1)

    num_stars = x.shape[0]

    # Unpack transformation parameters and proper motion parameters
    params, mu_xi, mu_eta = unpack_params(params, num_epochs, num_stars, pm=True)
    params = np.array(params).reshape(14, num_epochs)
    a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp = params
    
    # Compute xi and eta at each epoch
    xi = np.array([
        a[t] * x[:, t] + b[t] * y[:, t] + c[t] * x[:, t] * y[:, t] +
        d[t] * x[:, t] ** 2 + e[t] * y[:, t] ** 2 +
        f[t] * x[:, t] * (x[:, t] ** 2 + y[:, t] ** 2) + g[t] - mu_xi * delta_t[t]
        for t in range(num_epochs)
    ])

    eta = np.array([
        ap[t] * x[:, t] + bp[t] * y[:, t] + cp[t] * x[:, t] * y[:, t] +
        dp[t] * x[:, t] ** 2 + ep[t] * y[:, t] ** 2 +
        fp[t] * y[:, t] * (x[:, t] ** 2 + y[:, t] ** 2) + gp[t] - mu_eta * delta_t[t]
        for t in range(num_epochs)
    ])

    return xi.T, eta.T  # Shape: (num_stars, num_epochs)

# Dictionary of models
MODEL_REGISTRY = {
    1: {"model": model_1, "n_params": 4, "pm": False},
    2: {"model": model_2, "n_params": 4, "pm": True},
    3: {"model": model_3, "n_params": 14, "pm": False},
    4: {"model": model_4, "n_params": 14, "pm": True}
}

# Function to fit model across multiple time steps
def fit_model(model_index, x_data, y_data, xi_data, eta_data, delta_t, xi_errors=None, eta_errors=None):
    """
    Fit model across multiple time steps, enforcing proper motions to be the same for all epochs.
    
    x_data, y_data: shape (N_stars, N_epochs)
    xi_data, eta_data: shape (N_stars, N_epochs)
    delta_t: shape (N_epochs,)
    xi_errors, eta_errors: shape (N_stars, N_epochs)

    Returns:
    - params: array of fitted model parameters
    - xi_fit, eta_fit: arrays of transformed star positions at each epoch (shape: (N_stars, N_epochs))
    - covariance: covariance matrix of the fitted parameters
    """
    if model_index not in MODEL_REGISTRY:
        raise ValueError(f"Model index {model_index} is not defined.")

    model_info = MODEL_REGISTRY[model_index]
    model_func = model_info["model"]
    
    N_stars, N_epochs = x_data.shape
    num_params = model_info["n_params"]*N_epochs + 2*N_stars*model_info["pm"]   # 2 parameters per star for proper motion

    # Flatten data for fitting
    xi_eta_data_flat = np.concatenate([xi_data.flatten(), eta_data.flatten()])
    sigmas = np.concatenate([xi_errors.flatten(), eta_errors.flatten()]) if xi_errors is not None else None

    def wrapped_model(inputs_array, *params):
        x = inputs_array[0].reshape(N_stars, N_epochs)
        y = inputs_array[1].reshape(N_stars, N_epochs)
        delta_t = inputs_array[2][:N_epochs]  # Ensure delta_t remains a 1D array

        xi_fit, eta_fit = model_func(x, y, delta_t, params)
        return np.concatenate([xi_fit.flatten(), eta_fit.flatten()])

    # Initial guess for parameters
    p0 = np.ones(num_params)

    # Prepare input array for curve_fit
    # First, normalize x, y, and delta_t
    x_scale = np.abs(x_data).mean()
    y_scale = np.abs(y_data).mean()
    t_scale = delta_t.max()
    input_array = np.vstack([x_data.flatten()/x_scale, y_data.flatten()/y_scale, np.tile(delta_t, N_stars)]/t_scale)

    # Fit model
    params, covariance = curve_fit(wrapped_model, input_array, xi_eta_data_flat, p0=p0, sigma=sigmas, absolute_sigma=True)

    # Compute fitted values
    xi_fit, eta_fit = model_func(x_data, y_data, delta_t, params)

    return params, xi_fit, eta_fit, covariance

#################################################################################
################################   MAIN SCRIPT   ################################
#################################################################################

#%% Set paths and flags
fitsfn = "/Users/taladi/codes/kepler_astrometry_catalog/data/maestro/clean-analyze_20250317-152902/make_clean/bndct_kicid_list.fits"
data_test_plot = True
REDUCE = True
day_avg = 1.5

#%% Load the catalog
with fits.open(fitsfn) as hdu:
    data_dict = {
        "time": hdu["TIME"].data["time"][0],
        "kicids": hdu["PATHS"].data["kicid"],
        "rawx": hdu["RAW_CENTROIDS"].data["RAW_CENTROID_X"],    # in pixels (ntimes, nstars)
        "rawy": hdu["RAW_CENTROIDS"].data["RAW_CENTROID_Y"],    # in pixels (ntimes, nstars)
        # "res_x": hdu["RESIDUALS"].data["CENTROID_X"],   # in mas (ntimes, nstars)
        # "res_y": hdu["RESIDUALS"].data["CENTROID_Y"],   # in mas (ntimes, nstars)
        # "poscorrx": hdu["POSCORR"].data["CORR_X"],  # in pixels (ntimes, nstars)
        # "poscorry": hdu["POSCORR"].data["CORR_Y"]   # in pixels (ntimes, nstars)
    }

# Test plots the raw and poscorr centroids of the i'th star
i = 0
if data_test_plot:
    # Create a figure and axes for plotting
    fig, ax = plot_star_centroids(data_dict, i, data_type="raw", marker='.', label="raw")  # First plot with "raw" data
    fig, ax = plot_star_centroids(data_dict, i, data_type="poscorr", fig=fig, ax=ax, marker='+', label="poscorr")  # Add plot for the next star with "poscorr" data

# Reduce data by averaging over time period
if REDUCE:
    red_data_dict = time_avg_reduce(data_dict, day_avg)
    data_dict = red_data_dict

#%% Fit the model to the data

# Select model
model_index = 4
model_info = MODEL_REGISTRY[model_index]

# Prepare data
time = data_dict["time"].copy()
time -= time[0]
x_data = data_dict["rawx"].copy()
y_data = data_dict["rawy"].copy()

# Shift each star to the origin
x_data -= np.mean(x_data, axis=1)[:, np.newaxis]
y_data -= np.mean(y_data, axis=1)[:, np.newaxis]

# Generate zero array with the same shape as x_data
nstars, ntimes = x_data.shape
zeros = np.zeros((nstars, ntimes))

# determine errors in positions residuals
pct_err = 1e-7
xi_errors = np.abs(pct_err * x_data)
eta_errors = np.abs(pct_err * y_data)

#%% Fit model
all_params_fit, x_fit, y_fit, covariance = fit_model(model_index,
                                                        x_data=x_data, y_data=y_data,
                                                        xi_data=zeros, eta_data=zeros,
                                                        delta_t=time,
                                                        # xi_errors=xi_errors, eta_errors=eta_errors
                                                        )
data_dict["res_x_fit"] = x_fit
data_dict["res_y_fit"] = y_fit

#%% Plot
data_dict["res_x"] = x_data
data_dict["res_y"] = y_data

tindx = 3
fig, ax = plot_all_star_centroids(data_dict, tindx, "res")
fig, ax = plot_all_star_centroids(data_dict, tindx, "res_fit", fig, ax, marker='*')

#%% Unpack the fitted parameters
params, mu_xi, mu_eta = unpack_params(all_params_fit, ntimes, nstars, pm=True)

# Reshape the parameters for each epoch
params = params.reshape(14, ntimes)

mu_xi - mu_xi.mean()
mu_eta

#%% Create a video of the all star centroids plot
import matplotlib.animation as animation

# Create a figure and axes for plotting
fig, ax = plt.subplots()
ax.set_xlim(x_data.min(), x_data.max())
ax.set_ylim(y_data.min(), y_data.max())

# Initialize the scatter plot
scat = ax.scatter(x_data[:, 0], y_data[:, 0], c='b', label="raw")
scat_fit = ax.scatter(x_fit[:, 0], y_fit[:, 0], c='r', label="fit")

# Update function for the animation
def update_plot(frame):
    scat.set_offsets(np.column_stack([x_data[:, frame], y_data[:, frame]]))
    scat_fit.set_offsets(np.column_stack([x_fit[:, frame], y_fit[:, frame]]))
    return scat, scat_fit

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=range(ntimes), interval=100, blit=True)

ani.save("star_centroids.mp4", writer="ffmpeg")

#%% 

import pickle

# Save Gaia dictionary to a pickle file
pickle.dump(gaia_data, open("gaia_data.pkl", "wb"))