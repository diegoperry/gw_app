"""

"""

from dataclasses import dataclass, field
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import pickle

import kepler_astrometry_catalog.get_star_catalog as gsc

# from kepler_astrometry_catalog.plotting import make_fov_movie
from kepler_astrometry_catalog.paths import *

from raDec2Pix import raDec2PixModels as rm

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
from estoiles.gw_calc import GWcalc

from astropy.time import Time


@dataclass(kw_only=True)
class FakeSignal:
    texp: float = 90 * u.day
    freq: float = 1.71e-7 * u.Hz
    noise_std: float = (1.0 * u.mas).to(u.deg).value
    source_l: float = 76.3 * u.deg  # 0.0*u.deg
    source_b: float = 13.5 * u.deg  # 11.0*u.deg
    mc: float = 1.0e9 * u.Msun
    dl: float = 20 * u.Mpc
    sampleid: str = "brightestnonsat"
    plot_fov_movie: bool = False
    # ts: centroid time stamp, unit=day
    ts: list = field(default_factory=list)
    star_ra: list = field(default_factory=list)
    star_dec: list = field(default_factory=list)
    # starcoords: float = 0.0

    # FIXME: dataclass doesn't allow SkyCoord as keyword
    def __post_init__(self):

        self.sourcecoord = SkyCoord(l=self.source_l, b=self.source_b, frame="galactic")

        if len(self.ts) == 0:
            numexp = int((self.texp / (30 * u.min)).to(""))
            ## FIXME: original daty input is in days
            self.ts = (np.linspace(0, 90, numexp) * u.day).to(u.s)

        #if len(self.starcoords) == 0:
        if len(self.star_ra) == 0:
            self.getstarcoords()

        # BJD = BKJD + 2454833

        #time_bjd = self.ts + 2454833  # bjdrefi
        # FIXME: Why does adding 300 fix below line?
        time_bjd = self.ts.to(u.day).value + 2454833 + 300
        time = Time(time_bjd, format="jd", scale="tdb")
        raPointing, decPointing, rollPointing = rm.pointingModel().get_pointing(
            time.mjd
        )

        # FIXME: need to check that this is in the right coordinate system
        self.telcoord = SkyCoord(
            ra=raPointing, dec=decPointing, unit=u.deg, frame="icrs"
        )
        self.telcoord = self.telcoord.galactic

        # self.telcoord = SkyCoord(
        #     l=np.median(self.starcoords.galactic.l),
        #     b=np.median(self.starcoords.galactic.b),
        #     frame="galactic",
        # )
        self.starcoords = SkyCoord(
            ra=self.star_ra, dec=self.star_dec, unit=u.deg, frame="icrs"
        )
        self.starcoords = self.starcoords.galactic

        self.data = self.get_dn_v1()

        # self.h_m = self.get_h()  # get waveform
        # self.data = self.make_fake_data()
        if self.plot_fov_movie:
            self.make_movie()

    def getstarcoords(self):
        # should interface here with Clean
        self.cat = gsc.StarCatalog(
            #cat_type=self.sampleid,
            nmax=100,
            use_pqual=False,
            use_prot=True,
            get_lightcurves_shell=False,
            save_cat=False,
            lc_quarter=12,
            ext_cat=None,
        )
        kep_stars = self.cat.df
        self.starcoords = SkyCoord(
            ra=kep_stars["kic_degree_ra"], dec=kep_stars["kic_dec"], unit=u.deg
        )

    def get_dn_v1(self):
        #phis = (self.ts * u.d).to(1 * u.s) * self.freq * np.pi * u.rad
        phis = (self.ts).to(1 * u.s) * self.freq * np.pi * u.rad
        theta = (
            0 * u.s,
            np.array([True, False, False, False, False]),
        )  # 0s here just a dummy variable
        dra = np.empty([len(self.ts), len(self.starcoords)])
        ddec = np.empty([len(self.ts), len(self.starcoords)])
        for it, (p, telcoord) in enumerate(zip(phis, self.telcoord)):
            telcoord = telcoord[0]
            g = gwc.GWcalc(
                self.mc,
                1,
                self.freq,
                self.dl,
                0.0 * u.deg,
                0.0 * u.deg,
                self.sourcecoord,
                telcoord,
            )
            h = g.calc_h(theta, phi_=p)
            src = g.coordtransform(telcoord, self.sourcecoord)
            stc = g.coordtransform(telcoord, self.starcoords)

            # src = g.coordtransform(telcoord, self.sourcecoord)
            # stc = g.coordtransform(telcoord, self.starcoords)  #FIXME: check if this work
            dn = cdn.dn(h, src, stc).T[:, :, 0]  # in radians

            dra[it] = -dn[:, 0] / np.cos(
                self.starcoords.galactic.b
            )  # changes in radians
            ddec[it] = -dn[:, 1]  # changes in radians
        self.h = h  ## just save some strain so we have an idea what it is.

        # EKS added noise
        if self.noise_std > 0.0:
                noise = np.random.default_rng().normal(0, self.noise_std, len(self.ts))
        else:
                noise = np.zeros((len(self.ts)))
        #data[i] = self.evolve_star(l, b) + noise
        return np.add([dra.T * u.rad.to(u.deg), ddec.T * u.rad.to(u.deg)], noise) 

    def get_h(self):
        # phis = (self.ts * self.freq * u.rad * np.pi).to(u.rad)
        # phis = (self.ts * u.d).to(1 * u.s) * self.freq * np.pi
        phis = (self.ts).to(1 * u.s) * self.freq * np.pi
        self.g = gwc.GWcalc(
            self.mc,
            1,
            self.freq,
            self.dl,
            0.0 * u.deg,
            0.0 * u.deg,
            self.sourcecoord,
            self.telcoord,
        )
        self.src = self.g.coordtransform(self.telcoord, self.sourcecoord)
        theta = (0 * u.s, np.array([True, False, False, False, False]))
        h_m = [self.g.calc_h(theta, phi_=p) for p in phis]
        return h_m

    def evolve_star(self, l, b):
        stc = self.g.coordtransform(self.telcoord, SkyCoord(l=l, b=b, frame="galactic"))
        mock_dn = [
            cdn.dn(h, self.src, stc) for h in self.h_m
        ]  # do this for different times
        # some portion is vectorized, I think
        newpos_l = (np.array([mc[0] for mc in mock_dn]) * u.rad).to(u.mas)
        newpos_b = (np.array([mc[1] for mc in mock_dn]) * u.rad).to(u.mas)
        return (newpos_l).to(u.deg), (newpos_b).to(u.deg)

    def make_fake_data(self):
        data = np.zeros((len(self.starcoords), 2, len(self.ts)))
        # loop over this
        for i, (l, b) in enumerate(
            zip(self.starcoords.galactic.l, self.starcoords.galactic.b)
        ):
            if self.noise_std > 0.0:
                noise = np.random.default_rng().normal(0, self.noise_std, len(self.ts))
            else:
                noise = np.zeros((len(self.ts)))
            data[i] = self.evolve_star(l, b) + noise
        return data

    def data_to_lb(self):
        data = np.array(self.data).transpose(1, 0, 2) # switch coordinates axis with star # axis
        lbdata = np.zeros((np.shape(data)))
        #print(lbdata.shape)
        for i, d in enumerate(data): # iterating over stars
            #print(i, d)
            #print(d.shape)

            lbdata[i] = [
                d[0] * u.deg + self.starcoords.galactic.l[i, np.newaxis],
                d[1] * u.deg + self.starcoords.galactic.b[i, np.newaxis],
            ]
        return lbdata

    def make_movie(self, show=False, save=False):
        lbdata = self.data_to_lb()
        make_fov_movie(
            lbdata,
            os.path.join(
                MOVIEDIR,
                f"fakesig_{np.log10(self.mc.value):.0f}_{self.dl.value:.0f}.mp4",
            ),
            show=show,
            save=save,
        )
        return

    def save_model_to_pickle(self, outdir):
        true_model_params = {
            "freq": self.freq.to(u.Hz).value,
            "src_mass_msun": self.mc.to(u.M_sun).value,
            "src_distance_mpc": self.dl.to(u.Mpc).value,
            "position": self.sourcecoord,
            "source_ra_deg": self.sourcecoord.icrs.ra.deg,
            "source_dec_deg": self.sourcecoord.icrs.dec.deg,
            "source_psi_deg": 0.0,
            "estoiles_gw_source": True,
        }
        print("Model true parameters:")
        print(true_model_params)
        with open(f"{outdir}/model_true_params.pickle", "wb") as f:
            pickle.dump(true_model_params, f)
