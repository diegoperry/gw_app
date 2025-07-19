import os

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

from estoiles import errors
from estoiles import gw_source
from estoiles import paths
from estoiles import star_catalog


KEPLER_STAR_CAT_PATH = os.path.join(paths.CATALOGDIR, 'stars_observed_by_kepler_X_KIC.csv')


def kepler_mock_gaussian_err(gw_source, nstars, obs_time_len,
                             star_cat=None,
                             sigma=None, target_nstars_sigma_scaling=None, cadence=30 * u.min, random_seed=None):
    '''Generates mock Kepler data, assuming only errors are Gaussian, based on star magnitude.

    Args:
        gw_source: GWSource object.
        nstars: Number of stars
        obs_time_len: Total observing window; must include astropy unit.
        star_cat: estoiles.star_catalog.StarCatalog object. If None, defaults to Kepler catalog for backwards compat.
        sigma: Manual override on noise level, instead of using magnitude-fitted curve. Defaults to None.
        target_nstars_sigma_scaling: If not none, multiply the errors by sqrt(N) / sqrt(N_target). Defaults to None.
        cadence: Time between observations; must include unit. Defaults to 30*u.min.
        random_seed: Random seed. Defaults to None.

    Returns:
        Tuple of:
          - Time (in seconds, with no attached unit)
          - True star coordinates (in radians, no attached unit) without noise. Shape (N_stars, time, x/y/z).
          - Total noise, with same shape and units as true star coordinates.
          - error sigma (in radians, no attached unit)
          - Array of cartesian (in telescope-centered frame) star coordinates selected from the Kepler catalog,
            with shape (N_stars, 3).
    '''
    if star_cat is None:
        star_cat = star_catalog.KeplerStars(cat_path=KEPLER_STAR_CAT_PATH, minmag=11.4, maxmag=16, nstars=nstars, random_seed=random_seed)

    error_source = errors.ErrorStochasticAstrometry(star_cat.mag, random_seed=random_seed)
    if sigma is not None:
        error_source.sigmas = np.full_like(error_source.sigmas, sigma)
    if target_nstars_sigma_scaling is not None:
        error_source.sigmas *= (nstars / target_nstars_sigma_scaling)**0.5

    n_obs = int((obs_time_len // cadence).value)
    time_arr = np.linspace(0, 1, n_obs) * obs_time_len

    # Move the star axis to be the zeroth axis, to match semantics of loaded data (which have shape (N_stars, time, x/y/z))).
    move_star_axis = lambda dn: np.moveaxis(dn, -1, 0)

    # Both output in radians already (without attached astropy unit).
    true_star_pos = move_star_axis(gw_source.dn(star_cat.starcoords, time_arr) + star_cat.starcoords)
    noise_star_pos = move_star_axis(error_source.dn(star_cat.starcoords, time_arr))
    return time_arr.to(u.second).value, true_star_pos.value, noise_star_pos, error_source.sigmas, star_cat.starcoords.T

def random_skycoord(telcoord, rng):
    '''Generate a SkyCoord drawn from a uniform random distribution on the sky.

    Args:
        telcoord: Telescope's SkyCoord; needed to properly set the random coord's frame. 
        rng: RandomState PRNG.

    Returns:
        Uniformly sampled SkyCoord.
    '''
    unit_vec = rng.normal(size=(3,))
    unit_vec /= np.sum(unit_vec**2)**0.5
    coord = SkyCoord(x=unit_vec[0], y=unit_vec[1], z=unit_vec[2], representation_type='cartesian')
    coord = coord.transform_to(telcoord)
    return coord
