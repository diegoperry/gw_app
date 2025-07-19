from dataclasses import dataclass
import os

import numpy as np
import astropy.units as u
import pandas as pd

from estoiles.interfaces import DeflectionSourceInterface
import estoiles.paths


'''
Median astrometric precision, estimated (after cleaning) to be 2 mas.
Gaussian stochastic error. Assumes time steps are ~30 min apart; doesn't explicitly check that.

This is a clunky hack to get around the fact that the dn interface only takes in coordinates,
but this class will assume that the kep_mag passed in at instantiation has magnitudes in the same order 
(and same number) as coords later passed into dn. It will draw different errors for the different coords
according to the magnitude associated with each.

Errors are uncorrelated with each other.
'''
@dataclass
class ErrorStochasticAstrometry(DeflectionSourceInterface):
    kep_mag: np.ndarray
    astrometric_precision_csvpath: str = os.path.join(estoiles.paths.CATALOGDIR, 'monet2010_astrometric_precision_lowerenvelope.csv')
    random_seed: int = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.random_seed)
        # Units are mas, so convert to radians.
        self.sigmas = self.get_astrometric_precision(self.kep_mag, self.astrometric_precision_csvpath) * 1e-3 * u.arcsec
        self.sigmas = self.sigmas.to(u.radian).value
        assert not np.any(np.isnan(self.sigmas))

    # Output units are in radians (without attached unit).
    def dn(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
        assert n_.shape[-1] == len(self.kep_mag)
        return self.rng.normal(loc=0, scale=self.sigmas, size=(len(t_), 3, n_.shape[-1]))

    @staticmethod
    def get_astrometric_precision(kepmag, csvpath):
        """
        Interpolate the Monet+2010 astrometric precision plot.
        If kepmag includes stars with magnitudes >16 or <11.4, their values will be
        filled as NaN.

        <11.4 is saturation
            (Gilliland+10 https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.160G/abstract)
        >16 is beyond the Monet+2010 limits.

        Args:
            kepmag (np.ndarray): of Kepler magnitudes
        Returns:
            astrom_precision_mas (np.ndarray): astrometric precision, in
            milliarcseconds, in a 30-minute stack.
        """

        # csvpath = os.path.join(
        #     DATADIR, "processed", "monet2010_astrometric_precision_lowerenvelope.csv"
        # )
        df = pd.read_csv(csvpath)

        eps = np.random.uniform(low=-1e-8, high=1e-8, size=len(df))
        df['kic_mag'] = df['kic_mag'] + eps

        df = df.sort_values(by='kic_mag', ascending=True)

        x = np.array(df.kic_mag)
        y = np.array(df.astrometric_error) * 1e3

        z = np.polyfit(x, y, 3)
        fn = np.poly1d(z)

        astrom_precision_mas = fn(kepmag)

        sel = (kepmag > 16) | (kepmag < 11.4)
        astrom_precision_mas[sel] = np.nan

        return astrom_precision_mas