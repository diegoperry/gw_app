'''
This class returns a standard star catalog with coordinates, within a specific field of view in galactic coordinates (l, b).
'''
from dataclasses import dataclass, field
import os

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import pandas as pd

from estoiles.paths import *
import estoiles.gw_calc

@dataclass(kw_only=True)
class StarCatalog():
    telcoord: SkyCoord = field(init=False)
    star_skycoords: SkyCoord = field(init=False)
    mag: np.ndarray = field(init=False)
    nstars: int = None
    minmag: float = -np.inf
    maxmag: float = np.inf
    fov_l: Quantity[u.deg] = None
    fov_b: Quantity[u.deg] = None
    b_span: Quantity[u.deg] = None
    l_span: Quantity[u.deg] = None
    
    rng: np.random.RandomState = field(init=False)
    random_seed: int = None
    
    @property
    def l(self):
        return self.star_skycoords.galactic.l.to(u.deg)
    @property
    def b(self):
        return self.star_skycoords.galactic.b.to(u.deg)
    @property
    def minb(self):
        if self.fov_b is not None and self.b_span is not None:
            return (self.fov_b-0.5*self.b_span)
        else:
            return -90 * u.deg
    @property
    def maxb(self):
        if self.fov_b is not None and self.b_span is not None:
            return (self.fov_b+0.5*self.b_span)
        else:
            return 90 * u.deg
    @property
    def minl(self):
        if self.fov_l is not None and self.l_span is not None:
            return (self.fov_l-0.5*self.l_span)
        else:
            return 0 * u.deg
    @property
    def maxl(self):
        if self.fov_l is not None and self.l_span is not None:
            return (self.fov_l+0.5*self.l_span)
        else:
            return 360 * u.deg
        
    # Duck-typing with setup_starcoords: transform into telescope frame.
    # Returned shape is (3, N).
    @property
    def starcoords(self):
        return estoiles.gw_calc.GWcalc.coordtransform(self.telcoord, self.star_skycoords.galactic)

    def __post_init__(self):
        self.rng = np.random.RandomState(seed=self.random_seed)
        self.telcoord = SkyCoord(l=self.fov_l,b=self.fov_b,frame='galactic')
        # assert self.minb >= -90 * u.deg and self.maxb <= 90 * u.deg
        self.load_coords()
        
    def load_coords(self):
        pass

@dataclass(kw_only=True)
class KeplerStars(StarCatalog):
    DEFAULT_TELCOORD = SkyCoord(ra='19h22m40s', dec='+44:30:00', unit=(u.hourangle, u.deg), frame='icrs')
    
    cat_path: str
    fov_l: Quantity[u.deg] = DEFAULT_TELCOORD.galactic.l.to(u.deg)
    fov_b: Quantity[u.deg] = DEFAULT_TELCOORD.galactic.b.to(u.deg)
        
    def load_coords(self):
        cat = pd.read_csv(self.cat_path)
        # Strip out NaNs in coordinates.
        cat.dropna(subset=['kic_degree_ra', 'kic_dec'], inplace=True)
        
        cat_all_star_skycoords = SkyCoord(ra=cat['kic_degree_ra']*u.deg, dec=cat['kic_dec']*u.deg)
        
        mask = np.logical_and.reduce([
            cat_all_star_skycoords.galactic.l >= self.minl,
            cat_all_star_skycoords.galactic.l <= self.maxl,
            cat_all_star_skycoords.galactic.b >= self.minb,
            cat_all_star_skycoords.galactic.b <= self.maxb,
            cat['kic_kepmag'] >= self.minmag,
            cat['kic_kepmag'] <= self.maxmag
        ])
        
        subsample_ind = np.arange(0, np.sum(mask))
        if self.nstars is not None and self.nstars < np.sum(mask):
            self.rng.shuffle(subsample_ind)
            subsample_ind = subsample_ind[:self.nstars]
        
        self.star_skycoords = cat_all_star_skycoords[mask][subsample_ind]
        self.mag = cat['kic_kepmag'].loc[mask].iloc[subsample_ind]
        self.nstars = len(self.mag)

@dataclass(kw_only=True)
class RomanStars(StarCatalog):
    DEFAULT_TELCOORD = SkyCoord(ra='19h22m40s', dec='+44:30:00', unit=(u.hourangle, u.deg), frame='icrs')
    
    fov_l: Quantity[u.deg] = DEFAULT_TELCOORD.galactic.l.to(u.deg)
    fov_b: Quantity[u.deg] = DEFAULT_TELCOORD.galactic.b.to(u.deg)
    
    l_span: Quantity[u.deg] = 0.5 * u.deg
    b_span: Quantity[u.deg] = 0.5 * u.deg
        
    def load_coords(self):
        telcoord_offset_frame = self.telcoord.galactic.skyoffset_frame()
        cat_all_star_skycoords = SkyCoord(lon=self.rng.uniform(-self.l_span.to(u.deg).value / 2, self.l_span.to(u.deg).value / 2, self.nstars) * u.deg,
                                          lat=self.rng.uniform(-self.b_span.to(u.deg).value / 2, self.b_span.to(u.deg).value / 2, self.nstars) * u.deg,
                                          frame=telcoord_offset_frame).icrs
        mask = np.ones(len(cat_all_star_skycoords), dtype=bool)
        
        subsample_ind = np.arange(0, np.sum(mask))
        if self.nstars is not None and self.nstars < np.sum(mask):
            self.rng.shuffle(subsample_ind)
            subsample_ind = subsample_ind[:self.nstars]
        
        self.star_skycoords = cat_all_star_skycoords[mask][subsample_ind]
        # TODO: We never actually use this right now, so it's a nonsense value.
        self.mag = np.ones(len(self.star_skycoords)) * 14
        self.nstars = len(self.mag)
    
@dataclass(kw_only=True)
class GaiaStars(StarCatalog):
    QUERYALL: bool = False

    def load_coords(self):
        fname = DATADIR+'GAIA/GaiaStars_l'+'{:.0f}'.format(self.fov_l.value)+'b'+'{:.0f}'.format(self.fov_b.value)+'_mag'+str(self.minmag)+'_'+str(self.maxmag)+'.npy'
        if os.path.exists(fname):
            print('file exists.')
            r = np.load(fname,allow_pickle=True)
        else:
            print('file not found, querying now...')
            if self.minl.to(u.deg).value < 0:
                l_string = " AND (g.l BETWEEN "+str(360 + self.minl.to(u.deg).value)+" AND 360 \
                             OR g.l BETWEEN 0 AND "+str(self.maxl.to(u.deg).value)+")"
            elif self.maxl.to(u.deg).value > 360:
                l_string = " AND (g.l BETWEEN "+str(self.minl.to(u.deg).value)+" AND 360 \
                             OR g.l BETWEEN 0 AND "+str(self.maxl.to(u.deg).value-360)+")"
            else:
                l_string = " AND g.l BETWEEN "+str(self.minl.to(u.deg).value)+" AND "+str(self.maxl.to(u.deg).value)

            job = Gaia.launch_job_async("SELECT l,b,phot_g_mean_mag \
                    FROM gaiadr2.gaia_source AS g  \
                    WHERE g.phot_g_mean_mag \
                    BETWEEN " + str(self.minmag) + " \
                    AND " + str(self.maxmag) + "\
                    AND g.b BETWEEN "+ str(self.minb.to(u.deg).value) +" \
                    AND "+ str(self.maxb.to(u.deg).value)+l_string, dump_to_file=False)
            r = job.get_results()
            if len(r)!=0:
                print('query finished, '+str(len(r))+' stars were found')
                np.save(fname,r)
            else:
                print('no qualified stars found.')
                return

        if not self.QUERYALL:
            if len(r)<= self.nstars:
                print('requested star number exceeds catalog size')
            else:
                ind = self.rng.randint(len(r),size=self.nstars)
                # ind = np.arange(self.nstars)
                r = r[ind]

        self.star_skycoords = SkyCoord(l=np.array(r['l'])*u.deg, b=np.array(r['b'])*u.deg, frame='galactic')
        self.mag = np.array(r['phot_g_mean_mag'])
        self.nstars = len(self.l)
        print('finished loading coordinates of '+str(self.nstars)+' stars.')
