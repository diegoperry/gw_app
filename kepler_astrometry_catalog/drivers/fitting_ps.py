#%% Imports
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize
import sys

sys.path.append("/Users/taladi/codes/kepler_astrometry_catalog")
from kepler_astrometry_catalog.clean import Clean

# Constants & conversion factors
bjdrefi = 2454833
deg_to_mas = 3.6e6
yr_to_day = 365.25

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

# Function to load Gaia data
def load_gaia_data(kicids):
    """
    Load Gaia data for a list of KIC IDs.

    Parameters:
    - kicids: list of KIC IDs

    Returns:
    - g_dict: dictionary of Gaia data for the given KIC IDs
    """

    import warnings
    from astropy.units import UnitsWarning
    warnings.simplefilter('ignore', category=UnitsWarning)
    fits_path = '/Users/taladi/codes/Kepler_Gaia/fits_files/'
    filename = 'kepler_dr3_good.fits'
    data = Table.read(fits_path+filename, format='fits')

    # Get the quantities of interest from the Kepler-Gaia file (numpy precision of the data is set to 64-bit)
    kep_mags,gaia_pm_ra,gaia_pm_ra_err,gaia_pm_dec,gaia_pm_dec_err = np.zeros([5,len(kicids)], dtype=np.float64)
    parallax, parallax_err = np.zeros([2,len(kicids)], dtype=np.float64)
    gaia_ra, gaia_dec, kic_ra, kic_dec = np.zeros([4,len(kicids)], dtype=np.float64)

    for i, k in enumerate(kicids):
        idx = np.where(data['kepid'] == k)[0][0]
        kep_mags[i] = data['kepmag'][idx]
        gaia_pm_ra[i] = data['pmra'][idx]              # in mas/yr
        gaia_pm_ra_err[i] = data['pmra_error'][idx]    # in mas/yr
        gaia_pm_dec[i] = data['pmdec'][idx]            # in mas/yr
        gaia_pm_dec_err[i] = data['pmdec_error'][idx]  # in mas/yr
        parallax[i] = data['parallax'][idx]             # in mas
        parallax_err[i] = data['parallax_error'][idx]   # in mas
        gaia_ra[i] = data['ra'][idx]           # in deg
        gaia_dec[i] = data['dec'][idx]         # in deg
        kic_ra[i] = data['ra_kic'][idx]     # in deg
        kic_dec[i] = data['dec_kic'][idx]   # in deg

    gaia_dict = {
        "kep_mags": kep_mags,
        "gaia_pm_ra": gaia_pm_ra,
        "gaia_pm_ra_err": gaia_pm_ra_err,
        "gaia_pm_dec": gaia_pm_dec,
        "gaia_pm_dec_err": gaia_pm_dec_err,
        "parallax": parallax,
        "parallax_err": parallax_err,
        "gaia_ra": gaia_ra,
        "gaia_dec": gaia_dec,
        "kic_ra": kic_ra,
        "kic_dec": kic_dec
    }

    # Gaia DR3 time in BJD
    jd_2000 = 2451545.0     # Julian date of J2000 (days)
    gaia_dr3_yr = 2016.0       # Gaia DR3 year
    gaia_bjd_time = jd_2000 + (gaia_dr3_yr - 2000.0) * 365.25  # in BJD

    gaia_dict["gaia_bjd_time"] = gaia_bjd_time

    return gaia_dict

# -----------------------------------
# >>> Models for fitting the data <<<
# -----------------------------------

def model_1(x, y, delta_t, params):
    """
    Generic rotation model + shift:
    xi = a * x + b * y + c
    eta = -b * x + a * y + f

    Parameters:
    - x, y: arrays of star positions at each epoch (shape: can be (nstars, ntimes) or (ntimes,) or (nstars,))
    - delta_t: array of time differences between epochs (shape: (ntimes,) or a single value)
    - params: array of model parameters (shape: 6*ntimes), ordered as follows:
        - 4 parameters for each of the ntimes: a, b, c, d

    Returns:
    - xi, eta: arrays of transformed star positions at each epoch (shape: (nstars, ntimes))
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
    - x, y: arrays of star positions at each epoch (shape: can be (nstars, ntimes) or (ntimes,) or (nstars,))
    - delta_t: array of time differences between epochs (shape: (ntimes,) or a single value)
    - params: array of model parameters (shape: 6*ntimes + 2*nstars), ordered as follows:
        - 4 parameters for each of the ntimes: a, b, c, d
        - 1 proper motion parameter for each of the nstars: mu_xi
        - 1 proper motion parameter for each of the nstars: mu_eta
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
    - x, y: arrays of star positions at each epoch (shape: (nstars, ntimes) or (ntimes,) or (nstars,))
    - delta_t: array of time differences between epochs (shape: (ntimes,) or a single value)
    - params: array of model parameters (shape: 14*ntimes), ordered as:
        - 14 parameters for each of the ntimes: a, b, c, d, e, f, g, ap, bp, cp, dp, ep, fp, gp

    Returns:
    - xi, eta: arrays of transformed star positions at each epoch (shape: (nstars, ntimes))
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
    - x, y: arrays of star positions (in mas) at each epoch (shape: can be (nstars, ntimes) or (ntimes,) or (nstars,))
    - delta_t: array of time differences between epochs (shape: (ntimes,) or a single value)
    - params: array of model parameters (shape: 14*ntimes + 2*nstars), ordered as follows:
        - 7 parameters for each of the ntimes: a, b, c, d, e, f, g
        - 7 parameters for each of the ntimes: ap, bp, cp, dp, ep, fp, gp
        - 1 proper motion parameter for each of the nstars: mu_xi (mas/yr)
        - 1 proper motion parameter for each of the nstars: mu_eta (mas/yr)
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
# def fit_model(model_index, x_data, y_data, xi_data, eta_data, delta_t, xi_errors=None, eta_errors=None):
#     """
#     Fit model across multiple time steps, enforcing proper motions to be the same for all epochs.
    
#     x_data, y_data: shape (nstars, ntimes)
#     xi_data, eta_data: shape (nstars, ntimes)
#     delta_t: shape (ntimes,)
#     xi_errors, eta_errors: shape (nstars, ntimes)

#     Returns:
#     - params: array of fitted model parameters
#     - xi_fit, eta_fit: arrays of transformed star positions at each epoch (shape: (nstars, ntimes))
#     - covariance: covariance matrix of the fitted parameters
#     """
#     if model_index not in MODEL_REGISTRY:
#         raise ValueError(f"Model index {model_index} is not defined.")

#     model_info = MODEL_REGISTRY[model_index]
#     model_func = model_info["model"]
    
#     nstars, ntimes = x_data.shape
#     num_params = model_info["n_params"]*ntimes
#     total_num_params = num_params + 2*nstars*model_info["pm"]   # 2 parameters per star for proper motion

#     # Flatten data for fitting
#     xi_eta_data_flat = np.concatenate([xi_data.flatten(), eta_data.flatten()])
#     sigmas = np.concatenate([xi_errors.flatten(), eta_errors.flatten()]) if xi_errors is not None else None

#     def wrapped_model(inputs_array, *params):
#         x = inputs_array[0].reshape(nstars, ntimes)
#         y = inputs_array[1].reshape(nstars, ntimes)
#         delta_t = inputs_array[2][:ntimes]  # Ensure delta_t remains a 1D array

#         xi_fit, eta_fit = model_func(x, y, delta_t, params)
#         return np.concatenate([xi_fit.flatten(), eta_fit.flatten()])

#     # Initial guess for parameters (all zeros)
#     p0 = np.zeros(total_num_params)
#     p0[:ntimes] = 1
#     if model_index in [3,4]:
#         p0[num_params//2+ntimes:num_params//2+2*ntimes] = 1

#     # Prepare input array for curve_fit
#     # First, normalize x, y, and delta_t
#     # x_scale = np.abs(x_data).mean()
#     # y_scale = np.abs(y_data).mean()
#     # t_scale = delta_t.max()
#     # input_array = np.vstack([x_data.flatten()/x_scale, y_data.flatten()/y_scale, np.tile(delta_t, nstars)]/t_scale)
#     input_array = np.vstack([x_data.flatten(), y_data.flatten(), np.tile(delta_t, nstars)])

#     # Fit model
#     params, covariance = curve_fit(wrapped_model, input_array, xi_eta_data_flat, p0=p0,
#                                 #    sigma=sigmas, absolute_sigma=True,
#                                 #    ftol=1e-12, xtol=1e-12, gtol=1e-12, maxfev=10000
#                                    )

#     # Compute fitted values
#     xi_fit, eta_fit = model_func(x_data, y_data, delta_t, params)

#     return params, xi_fit, eta_fit, covariance

def fit_model(model_index, x_data, y_data, xi_data, eta_data, delta_t, xi_errors=None, eta_errors=None):
    """
    Fit model across multiple time steps, enforcing proper motions to be the same for all epochs.
    
    x_data, y_data: shape (nstars, ntimes)
    xi_data, eta_data: shape (nstars, ntimes)
    delta_t: shape (ntimes,)
    xi_errors, eta_errors: shape (nstars, ntimes)

    Returns:
    - params: array of fitted model parameters
    - xi_fit, eta_fit: arrays of transformed star positions at each epoch (shape: (nstars, ntimes))
    - covariance: covariance matrix of the fitted parameters
    """
    if model_index not in MODEL_REGISTRY:
        raise ValueError(f"Model index {model_index} is not defined.")

    model_info = MODEL_REGISTRY[model_index]
    model_func = model_info["model"]
    
    nstars, ntimes = x_data.shape
    num_params = model_info["n_params"]*ntimes
    total_num_params = num_params + 2*nstars*model_info["pm"]   # 2 parameters per star for proper motion

    # Flatten data for fitting
    xi_eta_data_flat = np.concatenate([xi_data.flatten(), eta_data.flatten()])
    sigmas = np.concatenate([xi_errors.flatten(), eta_errors.flatten()]) if xi_errors is not None else None

    def wrapped_model(inputs_array, *params):
        """
        Wrap the model function for fitting by `minimize`.
        """
        x = inputs_array[0].reshape(nstars, ntimes)
        y = inputs_array[1].reshape(nstars, ntimes)
        delta_t = inputs_array[2][:ntimes]  # Ensure delta_t remains a 1D array

        xi_fit, eta_fit = model_func(x, y, delta_t, params)
        return np.concatenate([xi_fit.flatten(), eta_fit.flatten()])

    # Initial guess for parameters (all zeros)
    p0 = np.zeros(total_num_params)
    p0[:ntimes] = 1
    if model_index in [3,4]:
        p0[num_params//2+ntimes:num_params//2+2*ntimes] = 1

    # Prepare input array for fitting function
    input_array = np.vstack([x_data.flatten(), y_data.flatten(), np.tile(delta_t, nstars)])

    # Define objective function (sum of squared residuals)
    def objective(params):
        # Compute the model output
        xi_eta_fit_flat = wrapped_model(input_array, *params)
        
        # Compute residuals (difference between data and model)
        residuals = xi_eta_data_flat - xi_eta_fit_flat
        
        # Optionally weight by errors if provided
        if sigmas is not None:
            residuals /= sigmas
        
        # Return the sum of squared residuals
        return np.sum(residuals**2)

    # Perform the minimization
    result = minimize(objective, p0, method='L-BFGS-B')  # You can try other solvers as well

    # Retrieve the fitted parameters
    params = result.x
    
    # Compute fitted values (model output using the optimized parameters)
    xi_fit, eta_fit = model_func(x_data, y_data, delta_t, params)

    # Calculate covariance matrix from the inverse of the Hessian (if available)
    try:
        hessian_inv = result.hess_inv.toarray() if hasattr(result.hess_inv, 'toarray') else None
    except AttributeError:
        hessian_inv = None
    
    return params, xi_fit, eta_fit, hessian_inv

# Function to plot selected fit parameters vs. time
def plot_params_vs_time(params, time, sel_params, fig=None, ax=None, plot_args={}):
    # If params is a 1D array, raise an error
    if params.ndim == 1:
        raise ValueError("params should be a 2D array with shape (n_params, ntimes)."
                        "Run unpack_params() first to extract the parameters.")

    n_params = len(sel_params)
    if n_params > 1:
        if fig is None or ax is None:
            fig, ax = plt.subplots(n_params, 1, figsize=(10, 8), sharex=True)

        for i, p_i in enumerate(sel_params):
            ax[i].plot(time, params[p_i], **plot_args)
            ax[i].set_ylabel(f"Parameter No. {p_i}")
        ax[-1].set_xlabel("Time [days]")
        if "label" in plot_args:
            ax[0].legend()
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, params[sel_params[0]], **plot_args)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(f"Parameter No. {sel_params[0]}")
        if "label" in plot_args:
            ax.legend()
    
    plt.tight_layout()

    return fig, ax

# Function to plot RA and Dec positions
def plot_positions(
        x_data: np.ndarray,
        y_data: np.ndarray,
        time: np.ndarray = None,
        type="stars", indx=0,
        fig=None, ax=None,
        plot_args: dict = {}
):
    """
    Plot RA and Dec positions of stars or Gaia data.

    Parameters:
        x_data (ndarray): RA data (shape: (nstars, ntimes))
        y_data (ndarray): Dec data (shape: (nstars, ntimes))
        time (ndarray): Time data (shape: (ntimes,))
        type (str): Type of data to plot ('stars' or 'time')
        indx (int): Index of the star or epoch to plot
        plot_args (dict): Additional arguments for plotting

    Returns:
        fig, ax: Figure and axis objects
    """
    if type == "time":
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
        ax[0].plot(time, x_data[indx], **plot_args)
        ax[0].set_ylabel("RA")
        ax[1].plot(time, y_data[indx], **plot_args)
        ax[1].set_ylabel("Dec")
        ax[1].set_xlabel("Time [days]")
        if "label" in plot_args:
            ax[0].legend()
    elif type == "stars":
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_data[:, indx], y_data[:, indx], **plot_args)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        if "label" in plot_args:
            ax.legend()

    # If labels are provided, plot legend
    return fig, ax

# Function to animate star centroids
def animate_star_centroids(x_data, y_data, xi_data, eta_data, x_fit, y_fit, output_file="~/star_centroids.mp4"):
    """
    Create an animation of star centroids over time.

    Parameters:
    x_data, y_data : np.ndarray
        Arrays of shape (N, ntimes) representing star centroids.
    xi_data, eta_data : np.ndarray
        Reference arrays of shape (N, ntimes) for additional points.
    x_fit, y_fit : np.ndarray
        Arrays of shape (N, ntimes) for fitted centroids.
    ntimes : int
        Number of time steps in the animation.
    output_file : str, optional
        File name for saving the animation (default: "star_centroids.mp4").
    """
    import matplotlib.animation as animation
    ntimes = x_data.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(y_data.min(), y_data.max())
    
    # Initialize the scatter plots
    scat_true = ax.scatter(xi_data[:, 0], eta_data[:, 0], marker="*", c="k", alpha=0.4, label="Gaia")
    scat_data = ax.scatter(x_data[:, 0], y_data[:, 0], marker="*", s=10, c="b", label="Kepler")
    scat_fit = ax.scatter(x_fit[:, 0], y_fit[:, 0], marker="x", c='r', s=30, label="fit")
    ax.legend(loc="upper right")
    
    # Update function for animation
    def update_plot(frame):
        scat_true.set_offsets(np.column_stack([xi_data[:, frame], eta_data[:, frame]]))
        scat_data.set_offsets(np.column_stack([x_data[:, frame], y_data[:, frame]]))
        scat_fit.set_offsets(np.column_stack([x_fit[:, frame], y_fit[:, frame]]))
        return scat_data, scat_fit
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, frames=range(ntimes), interval=100, blit=True)
    
    # Save animation
    ani.save(output_file, writer="ffmpeg")
    plt.close(fig)  # Close the figure to prevent display issues

# Function to run the Clean script for a chosen catalog
def run_clean_script(sampleid, quarter, max_nstars):
    """
    Run the Clean script and return the cleaned data.

    Parameters:
    sampleid : str
        Sample ID for the data catalog.
    quarter : int
        Kepler quarter to process.
    max_nstars : int
        Maximum number of stars to process.

    Returns:
    dict
        Dictionary containing the cleaned data:
        - time: Time array (in days)
        - kicids: KIC IDs of the stars
        - rawx: Raw x positions (in pixels)
        - rawy: Raw y positions (in pixels)
        - res_x: Residual x positions (in mas)
        - res_y: Residual y positions (in mas)
        - ra: Right Ascension (in degrees)
        - dec: Declination (in degrees)
    """
    cl = Clean(
        quarter=quarter,
        sampleid=sampleid,
        max_nstars=max_nstars,
        dvacorr=False,
        pca=False,
        conv2glob=True,
    )

    # Load the cleaned data
    data_dict = {
        "time": cl.time.mean(axis=0),   # in days (ntimes,)
        "kicids": cl.kicid, # KIC IDs of the stars (nstars,)
        # "rawx": cl.rctd_x,  # in pixels (nstars, ntimes)
        # "rawy": cl.rctd_y,  # in pixels (nstars, ntimes)
        # "res_x": cl.ctd_x,  # in mas (nstars, ntimes)
        # "res_y": cl.ctd_y,  # in mas (nstars, ntimes)
        "ra": cl.ra,  # in degrees (nstars, ntimes)
        "dec": cl.dec,  # in degrees (nstars, ntimes)
    }
    
    return data_dict

# Function to load cleaned data from a FITS file
def load_cleaned_fits(fitsfn):
    """
    Load cleaned data from a FITS file.

    Parameters:
    fitsfn : str
        Path to the FITS file.

    Returns:
    dict
        Dictionary containing the cleaned data.
    """
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
    
    return data_dict


#%%##############################################################################
################################   MAIN SCRIPT   ################################
#################################################################################



#%% Run the clean script
sampleid = 'bndct_kicid_list'
quarter = 6
max_nstars = 100
# Flags
REDUCE = True
day_avg = 2

# Run the Clean script
data_dict = run_clean_script(sampleid, quarter, max_nstars)

# Reduce data by averaging over time period
if REDUCE:
    red_data_dict = time_avg_reduce(data_dict, day_avg)
    data_dict = red_data_dict


# Load Gaia data
kicids = data_dict["kicids"]
gaia_dict = load_gaia_data(kicids)

# Project Gaia positions to Kepler time
gaia_in_kepler_time = gaia_dict["gaia_bjd_time"] - bjdrefi  # in days
kep_gaia_dt = data_dict["time"] - gaia_in_kepler_time       # in days
g_pm_ra = gaia_dict["gaia_pm_ra"] / deg_to_mas / yr_to_day      # convert to deg/day
g_pm_dec = gaia_dict["gaia_pm_dec"] / deg_to_mas / yr_to_day    # convert to deg/day

gaia_proj_ra = gaia_dict["gaia_ra"][:, None] + kep_gaia_dt[None, :] * g_pm_ra[:, None]      # in deg
gaia_proj_dec = gaia_dict["gaia_dec"][:, None] + kep_gaia_dt[None, :] * g_pm_dec[:, None]   # in deg

# Plot the Gaia and Kepler positions at the i'th epoch
i = 0

fig, ax = plot_positions(gaia_proj_ra, gaia_proj_dec, type="stars", indx=i,
                         plot_args={"s": 150, "c": "b", "label": "Gaia", "marker": "*"})
# ax.scatter(data_dict["ra"], data_dict["dec"], marker="*", s=20, c="r", label="Kepler")
fig, ax = plot_positions(data_dict["ra"], data_dict["dec"], type="stars", indx=i,
                            plot_args={"s": 20, "c": "r", "label": "Kepler", "marker": "*"},
                            fig=fig, ax=ax)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
plt.show()


# Fit the model to the data

# Select model
model_index = 3
model_info = MODEL_REGISTRY[model_index]

# Prepare data
time = data_dict["time"].copy()
delta_t = time - time[0]
x_data = data_dict["ra"].copy()
y_data = data_dict["dec"].copy()

xi_data = np.tile(gaia_proj_ra[:,0, np.newaxis], len(time))
eta_data = np.tile(gaia_proj_dec[:,0, np.newaxis], len(time))

# Shift to reference point
ref_ra = x_data.mean()
ref_dec = y_data.mean()
# ref_ra = x_data.mean()
# ref_dec = y_data.mean()
x_data -= ref_ra
y_data -= ref_dec
xi_data -= ref_ra
eta_data -= ref_dec
shifted_gaia_proj_ra = gaia_proj_ra - ref_ra
shifted_gaia_proj_dec = gaia_proj_dec - ref_dec

# rescale the data
# x_scale = np.abs(x_data).mean()
# y_scale = np.abs(y_data).mean()
# x_data /= x_scale
# y_data /= y_scale
# xi_data /= x_scale
# eta_data /= y_scale

# determine errors in positions residuals
pct_err = 1e-6
xi_errors = np.abs(pct_err * x_data)
eta_errors = np.abs(pct_err * y_data)

all_params_fit, x_fit, y_fit, covariance = fit_model(model_index,
                                                        x_data=x_data, y_data=y_data,
                                                        xi_data=xi_data, eta_data=eta_data,
                                                        delta_t=delta_t,
                                                        # xi_errors=xi_errors, eta_errors=eta_errors
                                                        )

# x_fit *= x_scale
# y_fit *= y_scale
# x_data *= x_scale
# y_data *= y_scale
# xi_data *= x_scale
# eta_data *= y_scale

# Print the xi2 value for the fit
nstars, ntimes = x_data.shape
dof = nstars * ntimes * 2 - all_params_fit.size
pre_red_xi2 = np.sum((x_data - xi_data)**2 + (y_data - eta_data)**2) / dof
red_xi2 = np.sum((x_fit - xi_data)**2 + (y_fit - eta_data)**2) / dof

print(f"reduced pre_xi2 = {pre_red_xi2}")
print(f"reduced xi2 = {red_xi2}")

# Unpack the fitted parameters
params, mu_xi, mu_eta = unpack_params(all_params_fit, ntimes, nstars, pm=model_info["pm"])


# Plot and animate the fitted star positions

indx = 0
fig, ax = plot_positions(xi_data, eta_data, type="stars", indx=indx,
                         plot_args={"s": 150, "c": "k", "label": "Gaia", "marker": "*"})
fig, ax = plot_positions(x_data, y_data, type="stars", indx=indx,
                         plot_args={"s": 50, "c": "b", "label": "Kepler", "marker": "*"},
                         fig=fig, ax=ax)
fig, ax = plot_positions(x_fit, y_fit, type="stars", indx=indx,
                         plot_args={"s": 30, "c": "r", "label": "fit", "marker": "x"},
                         fig=fig, ax=ax)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
plt.show()


# Residuals

x_data_res = x_data - xi_data
y_data_res = y_data - eta_data
x_fit_res = x_fit - xi_data
y_fit_res = y_fit - eta_data
gaia_proj_res = shifted_gaia_proj_ra - xi_data
gaia_proj_dec_res = shifted_gaia_proj_dec - eta_data

#%% Print mean and median of residuals absolute values
print("\t\t RA\t\t Dec")
print(f"Mean: \t\t {np.abs(x_data_res).mean():.3e}\t {np.abs(y_data_res).mean():.3e}")
print(f"Median: \t {np.median(np.abs(x_data_res)):.3e}\t {np.median(np.abs(y_data_res)):.3e}")

#%%
animate_star_centroids(x_data_res, y_data_res, x_data_res*0, x_data_res*0, x_fit_res, y_fit_res,
                        output_file="RA_Dec_fit.mp4")
#%% Plot residuals
indx = 2
fig, ax = plot_positions(gaia_proj_res, gaia_proj_dec_res, time=time, type="time", indx=indx,
                            plot_args={"label": "Gaia"})
fig, ax = plot_positions(x_data_res, y_data_res, time=time, type="time", indx=indx,
                         plot_args={"label": "Kepler"}, fig=fig, ax=ax)
fig, ax = plot_positions(x_fit_res, y_fit_res, time=time, type="time", indx=indx,
                         plot_args={"label": "fit"}, fig=fig, ax=ax)
for i in range(len(ax)):
    ax[i].axhline(0, color="k", linestyle="--", alpha=0.4)
    ax[i].set_ylabel("RA residuals [deg]")
ax[-1].set_xlabel("RA residuals [deg]")
ax[-1].set_ylabel("Dec residuals [deg]")
plt.show()


#%% Plot the fitted parameters vs. time
sel_params = [i for i in range(0, model_info["n_params"]//2)]
fig, ax = plot_params_vs_time(params, time, sel_params,
                             plot_args={"label": "fit", "marker": "o", "linestyle": "--"})

########################################################################
########################################################################

#%% Testing mock data
trns_params_prt_scale = 1e-4
noise_scale = 1e-10

# Select model
model_index = 4
model_info = MODEL_REGISTRY[model_index]

# Generate number of stars and epochs if is not defined
nstars = 50# if "nstars" not in locals() else nstars
ntimes = 48 #if "ntimes" not in locals() else ntimes

# Generate mock data
x_data0 = np.random.uniform(-1, 1, (nstars,))    # in deg
y_data0 = np.random.uniform(-1, 1, (nstars,))    # in deg
pm_params = np.random.uniform(-1, 1, 2*nstars) * 1e-9  # in deg/day
delta_t = np.linspace(0., 90., ntimes)  # in days
x_data = x_data0[:, None] + delta_t[None, :] * pm_params[:nstars, None]  # in deg
y_data = y_data0[:, None] + delta_t[None, :] * pm_params[nstars:, None]  # in deg

# Generate mock parameters
num_params = model_info["n_params"]*ntimes
trns_params = np.zeros(num_params)
trns_params[:ntimes] = 1
if model_index in [3,4]:
    trns_params[num_params//2+ntimes:num_params//2+2*ntimes] = 1
trns_params += np.random.uniform(-1, 1, num_params)*trns_params_prt_scale
all_trns_params = np.concatenate([trns_params, pm_params])


# Compute mock xi and eta data
model_func = MODEL_REGISTRY[model_index]["model"]
xi_data, eta_data = model_func(x_data, y_data, delta_t, all_trns_params)

# Generate errors (uncertainties for observed positions)
error_scale = 0.000001
xi_errors = np.random.normal(0.3, error_scale, size=(nstars, ntimes))
eta_errors = np.random.normal(0.3, error_scale, size=(nstars, ntimes))

# Add noise to x_data and y_data
# Normal distribution noise
x_noisy = x_data * (1 + np.random.normal(0, noise_scale, x_data.shape))
y_noisy = y_data * (1 + np.random.normal(0, noise_scale, y_data.shape))
# Uniform distribution noise
x_noisy = x_data + np.random.uniform(-noise_scale, noise_scale, x_data.shape)
y_noisy = y_data + np.random.uniform(-noise_scale, noise_scale, y_data.shape)

all_params_fit, x_fit, y_fit, covariance = fit_model(model_index,
                                                    x_data=x_noisy, y_data=y_noisy,
                                                    xi_data=xi_data, eta_data=eta_data,
                                                    delta_t=delta_t,
                                                    # xi_errors=xi_errors, eta_errors=eta_errors
                                                    )

# Reduced xi2 value for the fit
dof = nstars * ntimes * 2 - all_params_fit.size
pre_red_xi2 = np.sum((x_noisy - xi_data)**2 + (y_noisy - eta_data)**2) / dof
red_xi2 = np.sum((x_fit - xi_data)**2 + (y_fit - eta_data)**2) / dof
print(f"reduced pre_xi2 = {pre_red_xi2}")
print(f"reduced xi2 = {red_xi2}")

# Unpack the fitted parameters
params, mu_xi, mu_eta = unpack_params(all_params_fit, ntimes, nstars, pm=True)

# Plot the mock data positions
i = 0
fig, ax = plot_positions(xi_data, eta_data, type="stars", indx=i,
                         plot_args={"s": 150, "c": "b", "label": "mock", "marker": "*"})
fig, ax = plot_positions(x_noisy, y_noisy, type="stars", indx=i,
                            plot_args={"s": 20, "c": "r", "label": "data", "marker": "*"},
                            fig=fig, ax=ax)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
plt.show()

# Animate the mock data
# animate_star_centroids(x_noisy, y_noisy, xi_data, eta_data, x_fit, y_fit,
#                         output_file="RA_Dec_fit_mock.mp4")

# Plot residuals for the i'th star
x_data_res = x_noisy - xi_data
y_data_res = y_noisy - eta_data
x_fit_res = x_fit - xi_data
y_fit_res = y_fit - eta_data

#%% Print the average and median of the residuals absolute values
print("\t\t RA residuals \t\t Dec residuals")
print("Data")
print(f"Average residuals: {np.abs(x_data_res).mean()}, {np.abs(y_data_res).mean()}")
print(f"Median residuals: {np.median(np.abs(x_data_res))}, {np.median(np.abs(y_data_res))}")
print("Fit")
print(f"Average residuals: {np.abs(x_fit_res).mean()}, {np.abs(y_fit_res).mean()}")
print(f"Median residuals: {np.median(np.abs(x_fit_res))}, {np.median(np.abs(y_fit_res))}")

indx = 0
fig, ax = plot_positions(x_data_res, y_data_res, time=delta_t, type="time", indx=indx,
                         plot_args={"label": "data"})
fig, ax = plot_positions(x_fit_res, y_fit_res, time=delta_t, type="time", indx=indx,
                         plot_args={"label": "fit"}, fig=fig, ax=ax)
for i in range(len(ax)):
    ax[i].axhline(0, color="k", linestyle="--", alpha=0.4)
    ax[i].set_ylabel("RA residuals [deg]")
ax[-1].set_xlabel("RA residuals [deg]")
ax[-1].set_ylabel("Dec residuals [deg]")
plt.show()

# Plot the fitted parameters vs. time
sel_params = [i for i in range(0, 7)]
trns_params, pm_1, pm_2 = unpack_params(all_trns_params, ntimes, nstars, pm=True)
fig, ax = plot_params_vs_time(params, delta_t, sel_params, plot_args={"label": "fit", "marker": "x", "linestyle": "--"})
fig, ax = plot_params_vs_time(trns_params, delta_t, sel_params,
                                plot_args={"label": "mock", "marker": "o", "linestyle": "--"},
                                fig=fig, ax=ax)
plt.show()

############################################################
############################################################
############################################################



#%% Create a dictionary with all quarters for the sampleid 'bndct_kicid_list'
quarters = [i for i in range(0, 18)]

all_data_dict = {}
for q in quarters:
    cl = Clean(
        quarter=q,
        sampleid=sampleid,
        max_nstars=max_nstars,
        dvacorr=False,
        pca=False,
        conv2glob=True,
        )

    # Load the cleaned data
    data_dict = {
        "quarter": q,
        "time": cl.time.mean(axis=0),   # in days (ntimes,)
        "kicids": cl.kicid, # KIC IDs of the stars (nstars,)
        "ra": cl.ra,  # in degrees (nstars, ntimes)
        "dec": cl.dec,  # in degrees (nstars, ntimes)
        "rawx": cl.rctd_x,  # in pixels (nstars, ntimes)
        "rawy": cl.rctd_y,  # in pixels (nstars, ntimes)
        "res_x": cl.ctd_x,  # in mas (nstas, ntimes)
        "res_y": cl.ctd_y,  # in mas (nstas, ntimes)
    }

    all_data_dict[q] = data_dict

#%%Find kicids that are common in all quarters
common_kicids = list(set(all_data_dict[0]["kicids"]))

# Concatenate all data for the i'th star in the common kicids
i = 4
kicid = common_kicids[i]

ra_total = []
dec_total = []
time_total = []
for i, q in enumerate(quarters[3:]):
    data_dict = all_data_dict[q].copy()
    kicid_index = np.where(data_dict["kicids"] == kicid)[0][0]
    ra = np.array(data_dict["ra"][kicid_index])
    dec = np.array(data_dict["dec"][kicid_index])
    if i>0:
        ra_diff = ra[0] - ra_total[-1]
        dec_diff = dec[0] - dec_total[-1]
        ra -= ra_diff
        dec -= dec_diff
    ra_total.extend(ra)
    dec_total.extend(dec)
    time_total.extend(data_dict["time"])

ra_total = np.array(ra_total)
dec_total = np.array(dec_total)
time_total = np.array(time_total)

ra_total -= ra_total.mean()
dec_total -= dec_total.mean()

# Plot the total data with colorfunction for time
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(ra_total*deg_to_mas, dec_total*deg_to_mas, c=time_total, cmap="viridis", s=10)
ax.set_xlabel("RA [mas]")
ax.set_ylabel("Dec [mas]")
plt.colorbar(sc, label="Time [days]")
plt.show()