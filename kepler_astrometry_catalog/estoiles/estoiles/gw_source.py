'''
This class defines instances of supermassive blackhole binaries with the given
characteristics.
'''
from dataclasses import dataclass, field
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

# Use direct imports now that this file lives alongside gw_calc.py, calc_dn.py, interfaces.py
import gw_calc as gwc
import calc_dn as cdn
import interfaces

@dataclass
class GWSource(interfaces.DeflectionSourceInterface):
    '''
    Class that wraps GWcalc class, to make calculating deflections more convenient.
    See GWcalc class for explanation of parameters below.
    '''
    # Pylance disallows call expressions in type args, so use plain Quantity
    freq: u.Quantity
    Mc: float = 1.e8 * u.Msun
    q: float = 1.
    dl: float = 1. * u.Mpc
    inc: float = 0. * u.deg
    psi: float = 0 * u.deg
    # Orbital phase to use when calculating h-tensors. If None, calculate orbital phase
    # automatically based on the time member.
    phi: float | np.ndarray = None
    initial_phip: float = 0 * u.rad
    initial_phic: float = 0 * u.rad
    time: float | np.ndarray = 0. * u.s
    telcoord: SkyCoord = field(default_factory=lambda: SkyCoord(l=-90 * u.deg, b=90 * u.deg, frame='galactic'))
    sourcecoord: SkyCoord = field(default_factory=lambda: SkyCoord(l=0 * u.deg, b=90 * u.deg, frame='galactic'))
    post_newtonian_orderlist: np.ndarray = field(default_factory=lambda: np.array([True, True, True, True, True]))

    def __post_init__(self):
        self.logMs0 = np.log10((self.Mc / self.dl**.6).value)
        self.gw_calc = gwc.GWcalc(
            self.Mc,
            self.q,
            self.freq,
            self.dl,
            self.inc,
            self.psi,
            self.sourcecoord,
            self.telcoord,
            phip=self.initial_phip,
            phic=self.initial_phic,
        )
        self.srcindet = self.gw_calc.coordtransform(self.telcoord, self.sourcecoord)

    @property
    def h(self):
        return self.gw_calc.calc_h((self.time, self.post_newtonian_orderlist), phi_=self.phi)

    def dn(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
        '''Calculate coordinate deflections.

        Keyword arguments:
        n_ -- star's cartesian coordinates in telescope frame, numpy array
        t_ -- array of times at which to calculate the deflections.

        Returns:
        Array of deflection vectors of the star position, in cartesian coordinates in telescope frame, numpy array.
        Deflections are in units of radians, without an attached astropy unit.
        '''
        self.time = t_
        return cdn.dn(self.h, self.srcindet, n_)
