"""

Classes to do various things with parallax and proper motion

"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from importlib import reload
from astropy.table import Table
import pandas as pd
from astropy.io import fits
from astropy.time import Time
import astropy.coordinates as coords
import emcee
import corner

from kepler_astrometry_catalog.paths import *
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict
from kepler_astrometry_catalog.clean import Clean
from kepler_astrometry_catalog.get_centroids import get_raw_centroids


@dataclass(kw_only=True)
class StarMotionModelGaia:
    ## uses Gaia parallax and proper motion to make a model for star motion in Kepler data
    ## follows:  https://gist.github.com/neilzim/e838297bcfbaed6162c9701ddfd0bb7d
    ra: float  ##  initial ra. must be J2000 if t0 isn't set
    dec: float  ## initial dec. must be J2000 if t0 isn't set
    parallax: float
    pmra: float  ## should include cos\delta correction
    pmdec: float
    times: float = None  ## kepler times. given in typical kepler LC times.
    t0: float = None

    def __post_init__(self):
        self.T_j2000 = Time(
            "2000-01-01T00:00:00", format="isot", scale="utc"
        )  ##useful for other things.
        try:
            self.T_obs = Time(self.times + 2454833.0, format="jd")  ## convert to JD
            print("using user-specified time")
        except:
            ## set 1 year of obs time if none given
            print("setting time to 1 year")
            self.T_obs = (
                Time("2012-01-01T00:00:00", format="isot", scale="utc")
                + np.linspace(0, 365, 1000) * u.day
            )
        if self.t0 is None:
            self.t0 = self.T_obs[0]
        ## set coords at t0
        self.coord_t0 = coords.SkyCoord(
            ra=self.ra,
            dec=self.dec,
            distance=coords.Distance(parallax=self.parallax),
            pm_ra_cosdec=self.pmra,
            pm_dec=self.pmdec,
            obstime=self.t0,
        )
        ## set coords for other times
        self.coord_obs = coords.SkyCoord(ra=self.ra, dec=self.dec, obstime=self.T_obs)

        # get modeled ra and dec
        self.get_total_motion()
        return

    def get_pm_shift(self):
        ## uses astropy.apply_space_motion..but doesn't work due to a nonmatching frame issue?
        # new_coord = self.coord_t0.apply_space_motion(self.T_obs)
        # print(new_coord)
        # offsets = np.ndarray([self.coord_t0.spherical_offsets_to(new_coord[i]) for i in range(len(self.T_obs))])
        # self.pmshiftra = np.ndarray([offsets[i].ra for i in range(len(self.T_obs))])
        # self.pmshiftdec = np.ndarray([offsets[i].ra for i in range(len(self.T_obs))])
        ## old code. by hand implementation, which wouldn't handle radial velocities and L.O.S. effects. but it works!
        ## FIXME
        self.pmshiftra = (
            (self.T_obs - self.t0).value * u.day * self.pmra / np.cos(self.coord_t0.dec)
        ).to(u.deg)
        self.pmshiftdec = ((self.T_obs - self.t0).value * u.day * self.pmdec).to(u.deg)
        return

    def get_plx_shift(self):
        if self.parallax.value == 0.0:
            self.parra = 0.0 * u.deg * np.zeros((len(self.times)))
            self.pardec = 0.0 * u.deg * np.zeros((len(self.times)))
            return
        # Get geocentric ecliptic coordinates of Sun
        sun_loc = coords.get_sun(self.T_obs)
        sun_skycoord = coords.SkyCoord(
            frame="gcrs", obstime=self.T_obs, ra=sun_loc.ra, dec=sun_loc.dec
        )
        sun_eclon = sun_skycoord.geocentrictrueecliptic.lon
        sun_eclat = sun_skycoord.geocentrictrueecliptic.lat
        star_eclon = self.coord_t0.geocentrictrueecliptic.lon
        star_eclat = self.coord_t0.geocentrictrueecliptic.lat
        # This is a low-precision approximation for the annual parallax offset.
        # These forumlae assume the observer is geocentric. They ingore
        # the eccentricity of Earth's orbit, and the distinction between
        # the Sun and the Solar System barycenter.
        # It is correct to within ~2% of the parallax amplitude.
        # See, e.g., Chapter 8 of Spherical Astronomy by Robin M. Green for a derivation.
        plx_delta_eclon = (
            -self.parallax * np.sin(star_eclon - sun_eclon) / np.cos(star_eclat)
        )
        plx_delta_eclat = (
            -self.parallax * np.cos(star_eclon - sun_eclon) * np.sin(star_eclat)
        )

        ## convert to ra,dec
        coord_plx = coords.GeocentricTrueEcliptic(
            lon=star_eclon + plx_delta_eclon, lat=star_eclat + plx_delta_eclat
        ).transform_to(coords.ICRS)

        self.parra = self.coord_t0.spherical_offsets_to(coord_plx)[0]
        self.pardec = self.coord_t0.spherical_offsets_to(coord_plx)[1]

        self.parra -= self.parra[0]
        self.pardec -= self.pardec[0]
        return

    def get_total_motion(self):
        self.get_pm_shift()
        self.get_plx_shift()
        self.tot_ra = (self.ra + self.pmshiftra + self.parra).to(u.deg)
        self.tot_dec = (self.dec + self.pmshiftdec + self.pardec).to(u.deg)
        return


@dataclass(kw_only=True)
class GaiaParallaxSampler:
    ra: float  ##  initial ra. must be J2000 if t0 isn't set
    dec: float  ## initial dec. must be J2000 if t0 isn't set
    parallax: float
    pmra: float  ## should include cos\delta correction
    pmdec: float
    times: float  ## kepler times. given in typical kepler LC times.
    data: np.ndarray
    error: float = 2.0 * u.mas
    ndim: int = 2  ## number of mcmc params
    nwalkers: int = 16
    nburnin: int = 1000
    nsteps: int = 10000
    nthin: int = 15
    e: float = 0.0  ## eccentricity

    def __post_init__(self):
        self.labels = ["pmra", "pmdec"]
        return

    def log_prior(self, params):
        pmra, pmdec = params
        if (np.abs(pmra) > 300) or (np.abs(pmdec) > 300):
            return -np.inf
        else:
            return 0.0

    def log_like(self, params):
        prior = self.log_prior(params)
        if np.isinf(prior):
            return -np.inf
        pmra, pmdec = params
        smm = StarMotionModelGaia(
            ra=self.ra,
            dec=self.dec,
            parallax=self.parallax,
            pmra=pmra * u.mas / u.yr,
            pmdec=pmdec * u.mas / u.yr,
            times=self.times,
            t0=self.times[0],
        )
        modelx, modely = smm.tot_ra, smm.tot_dec
        model = np.vstack([modelx, modely])
        chisq = (
            -0.5
            * np.sum((model.to(u.mas) - self.data.to(u.mas)) ** 2)
            / (self.error) ** 2
        )
        return chisq.to("").value

    def run(self):
        p0 = (
            np.random.randn(self.nwalkers, self.ndim) * 0.1 * self.pmra.value
            + self.pmra.value
        )
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_like)
        state = self.sampler.run_mcmc(p0, self.nburnin, progress=True)
        self.sampler.reset()
        state = self.sampler.run_mcmc(state, self.nsteps, progress=True)
        self.samples = self.sampler.get_chain()
        if self.nthin == 0:
            self.flatchain = self.sampler.get_chain(discard=self.nburnin, flat=True)
        else:
            self.flatchain = self.sampler.get_chain(
                discard=self.nburnin, thin=self.nthin, flat=True
            )
        self.best_fit = np.zeros(self.ndim)
        for i in range(self.ndim):
            self.best_fit[i] = np.percentile(self.flatchain[:, i], 50)
        return

    def make_plots(self):
        self.make_fullchain_plot()
        self.make_model_data_plot()
        self.make_corner_plot()
        return

    def make_fullchain_plot(self):
        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        fig.show()
        return

    def best_fit_model(self):
        pmra = self.best_fit[0] * u.mas / u.yr
        pmdec = self.best_fit[1] * u.mas / u.yr
        smm = StarMotionModelGaia(
            ra=self.ra,
            dec=self.dec,
            parallax=self.parallax,
            pmra=pmra,
            pmdec=pmdec,
            times=self.times,
            t0=self.times[0],
        )
        return smm.tot_ra, smm.tot_dec

    def make_model_data_plot(self):
        f = plt.figure()
        plt.scatter(self.data[0], self.data[1])
        xgaia, ygaia = self.best_fit_model()
        plt.plot(xgaia, ygaia, color="k")
        f.show()
        return

    def make_corner_plot(self):
        fig = corner.corner(
            self.flatchain,
            truths=[self.pmra.value, self.pmdec.value],
            labels=self.labels,
        )
        fig.show()
        return
