'''
This class creates a population of gw sources.
'''


from dataclasses import dataclass, asdict
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from astroquery.vizier import Vizier
import wget
import os
from astropy.io import fits
import astropy.constants as const

from estoiles.gw_source import GWSource
import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
import estoiles.interfaces
from estoiles.paths import CATALOGDIR

@dataclass
class Population(estoiles.interfaces.DeflectionSourceInterface):
    nsource: int
    catalog: pd.DataFrame = None
    randomvars: dict = None
    twomass: bool = False
    sloan: bool = False
    Mc: float = None
    freq: float = None
    c: float = 2.99e8*u.m/u.s
    H0: float = (70*u.km).to(1*u.m)/u.s/u.Mpc
    _time: float | np.ndarray = 0 * u.s

    def __post_init__(self):
        if (self.catalog is None) and (self.twomass == False) and (self.sloan == False):
            poppars = self.make_random_vars()
        elif self.twomass:
            if self.catalog is None:
                self.catalog = Population.get_twomass()
            poppars = self.make_twomass_vars()
        elif self.sloan:
            poppars = self.make_sloan_vars()
        else:
            poppars = self.make_catalog_vars()

        self.make_pop(poppars)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t_):
        self._time = t_
        for src in self.gwss:
            src.time = t_

    def make_pop(self, poppars):
        self.gwss = [GWSource(freq=poppars['freq'][i], Mc=poppars['Mc'][i], sourcecoord=poppars['sourcecoord'][i], dl=poppars['dist'][i]) for i in range(self.nsource)]
        self.srcindetarray = [g.srcindet for g in self.gwss]
        self.pop = [asdict(g) for g in self.gwss]
        for k in self.pop[0].keys():
            setattr(self, k, [self.pop[i][k] for i in range(len(self.pop))])
        return self.pop

    @property
    def harray(self):
        return [g.h for g in self.gwss]

    def make_random_vars(self):
        if self.randomvars is None:

            if self.freq is None:
                minlogfreq = np.log10(7.8e-8)
                maxlogfreq = np.log10(Population.get_ISCO(self.Mc)/u.Hz)
                logfreqs = minlogfreq+(maxlogfreq-minlogfreq)*np.random.rand(self.nsource)
                freqs = 10.**logfreqs*u.Hz
            else:
                freqs = np.ones(self.nsource)*self.freq

            if self.Mc is None:
                logMcs = np.random.rand(self.nsource)*4+5.
                Mcs = 10**logMcs*u.Msun
            else:
                Mcs = np.ones(self.nsource)*self.Mc

            # logfreqs = np.random.rand(self.nsource)*3-9
            # freqs = 10.**logfreqs*u.Hz
            ls = np.random.rand(self.nsource)*360*u.deg
            bs = np.random.rand(self.nsource)*180.*u.deg - 90.*u.deg
            sourcecoord = [SkyCoord(l=l, b=b,frame='galactic') for l,b in zip(ls,bs)]
            # logMcs = np.random.rand(self.nsource)*4+5.
            # Mcs = 10**logMcs*u.Msun
            dists = np.random.rand(self.nsource)*250.*u.Mpc
            randpars = {'freq' : freqs,
                    'sourcecoord' : sourcecoord,
                    'Mc' : Mcs,
                    'dist' : dists}
        else:
            randpars = self.randomvars
        return randpars

    def make_catalog_vars(self):
        pass

    def make_twomass_vars(self):
        cat = self.catalog
        if self.nsource <= len(cat):
            inds = np.random.choice(np.arange(len(cat)),size=self.nsource,replace=False)
        else:
            self.nsource = len(cat)
            inds = np.arange(len(cat))
        sources = cat[cat.index.isin(inds)]
        #so the catalog contains the ra/DEC info, and the distance.
        sourcecoord = [SkyCoord(ra=r*u.deg, dec=d*u.deg).galactic for r,d in
                zip(sources['ra'], sources['dec'])]
        dists = [ i*u.Mpc for i in sources['dist']]
        ## make random vars for all other vars.
        randpars = self.make_random_vars()
        randpars['sourcecoord'] = sourcecoord
        randpars['dist'] = dists
        return randpars

    def make_sloan_vars(self):
        cat = fits.open(CATALOGDIR + 'nasa_sloan_atlas.fits')[1].data
        if self.nsource <= len(cat):
            inds = np.random.choice(np.arange(len(cat)),size=self.nsource,replace=False)
            sources = cat[inds]
        else:
            self.nsource = len(cat)
            sources = cat
        r = sources['RA']
        d = sources['DEC']
        sourcecoord = SkyCoord(ra=r*u.deg, dec=d*u.deg).galactic
        dists = sources['ZDIST']*self.c/self.H0
        randpars = self.make_random_vars()
        randpars['sourcecoord'] = sourcecoord
        randpars['dist'] = dists
        return randpars

    @staticmethod
    def get_twomass(catalog=None):
        file = CATALOGDIR + 'mingarelli_twomass.dat'
        ## first, get 2M++ catalog
        if os.path.exists(file):
            catalog = pd.read_csv(file)
        else:
            catalog = pd.read_csv(
                    '''https://raw.githubusercontent.com/ChiaraMingarelli/nanohertz_GWs/master/galaxy_data/2mass_galaxies.lst''',
                    delimiter='\s+',
                    names=['name', 'ra', 'dec', 'dist', 'kmag', 'nicename'])
            catalog.to_csv(file)
        return catalog

    def get_ISCO(Mc):
        fmax_pf = (const.c**3/(6**1.5*np.pi*const.G)).to(u.kg/u.s)
        Mc = Mc.to(1*u.kg)
        return fmax_pf/Mc*2**.3

    # Array of deflections of shape (S, ...) where S is the number of sources. Sum along axis 0 for total deflection.
    # Implemented for legacy purposes (things in sampler.py use the raw array)
    def _dn_raw(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
        # Set times for all sources to t_.
        self.time = t_
        # Then regenerate h matrices for new times. Do this once since self.harray is a property; don't want to generate that every iter
        # in the loop below.
        harray = self.harray
        return np.array([cdn.dn(harray[i], self.srcindetarray[i], n_) for i in range(self.nsource)])

    def dn(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
        return np.sum(self._dn_raw(n_, t_), axis=0)