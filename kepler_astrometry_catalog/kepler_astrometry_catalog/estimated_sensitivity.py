"""

Given an h5 file and systematics, will produce the transmission function and estimated sensitivity curve. Generalization of the formalism in Hazboun et. al (2019).

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.linalg import block_diag, cho_factor, cho_solve
import tables  # PyTables

from kepler_astrometry_catalog.paths import *
from kepler_astrometry_catalog.dva import DVA
from kepler_astrometry_catalog.constants import PIX_SCALE_MAS


MAS_TO_RAD = (1.0 * u.mas).to(u.rad).value


class Sensitivity:
    def __init__(self, h5file, model=None, f=None, plot=False):
        self.h5file = h5file
        if model is None:
            model = DVA(quarter=12, h5file=h5file)
        self.model = model
        group = h5file.root.Q12
        kicids = group.kicid
        pos = group.pos
        self.times = [row["time"] for row in pos]
        self.time = self.times[0]  ##FIXME.
        if f is None:
            self.f = np.logspace(
                np.log10(
                    (1 / (5 * (self.time[-1] - self.time[0]) * u.s)).to(u.Hz).value
                ),
                # np.log10((1./(10*u.year)).to(u.Hz).value),
                np.log10((1.0 / (60 * u.min)).to(u.Hz).value),
                1000,
            )
        else:
            self.f = f
        self.tf = []
        self.ni = []
        self.sens_curves = []
        self.sigmas = []
        for i, row in enumerate(kicids):
            try:
                tf = self.get_transmission_func(row["kicid"])
                self.tf.append(tf)
                sigma = (
                    np.sqrt(
                        (self.model.results[row["kicid"]]["std_x"]) ** 2
                        + (self.model.results[row["kicid"]]["std_y"]) ** 2
                    )
                    * MAS_TO_RAD
                    * PIX_SCALE_MAS
                )
                self.sigmas.append(sigma)
                ni = self.inverse_noise(
                    tf, sigma, deltat=self.time[1] - self.time[0]  ##FIXME.
                )
                self.ni.append(ni)
                self.sens_curves.append(self.hc_sens(self.f, ni, 1))
            except:
                print(f'No solution found for KIC{row["kicid"]}. Skipping.')
        self.tot_sens_curve = self.hc_sens(self.f, np.sum(self.ni, axis=0), 1)
        if plot == True:
            self.plot_tf_hc()

    def get_design_matrix(self, kicid):
        """
        produces the design matrices (Mx, My) for each systematic.
        """
        Mx, My = self.model.jacobian(self.time, kicid=kicid)  ## FIXME time
        return Mx, My

    def transmission_func(self, Gx, Gy):
        """
        calculates the transmission function given frequency, SVD of design matrices, Gx,Gy, and the times.
        """
        N = len(self.time)
        Gtildex = np.dot(
            np.exp(1j * 2 * np.pi * self.f[:, np.newaxis] * self.time), Gx
        ).astype(np.complex64)
        Tmatx = np.matmul(np.conjugate(Gtildex), Gtildex.T) / N

        Gtildey = np.dot(
            np.exp(1j * 2 * np.pi * self.f[:, np.newaxis] * self.time), Gy
        ).astype(np.complex64)
        Tmaty = np.matmul(np.conjugate(Gtildey), Gtildey.T) / N

        return np.real(np.diag(0.5 * (Tmatx + Tmaty)))

    def get_transmission_func(self, kicid):
        Mx, My = self.get_design_matrix(kicid)
        Ux, _, _ = np.linalg.svd(Mx, full_matrices=True)
        Gx = Ux[:, np.shape(Mx)[1] :]

        Uy, _, _ = np.linalg.svd(My, full_matrices=True)
        Gy = Uy[:, np.shape(My)[1] :]

        return self.transmission_func(Gx, Gy)

    def inverse_noise(self, tf, sigma, deltat):
        """
        calculates the inverse noise spectrum assuming white noise for the residuals.
        """
        pf = 2 * sigma**2 * deltat
        return tf / pf

    def seff(self, f, ni, nstar):
        """
        calculates the effective spectral noise density assuming all stars have the same inverse noise.
        """
        # rf = 1.0 / (12 * np.pi**2 * f**2)
        rf = 1.0  ##FIXME should be some other constant.
        return ((4.0 / 5.0) * nstar * ni * rf) ** (-1)

    def hc_sens(self, f, ni, nstar):
        """
        calculates the sensitivity in characteristic strain for a continuous source, averaged over sky position, inclination, and polarization angles
        """
        seff = self.seff(f, ni, nstar)
        return np.sqrt(f * seff)

    def h0_sens(self, f, seff, tobs):
        """
        calculates the sensitivity in strain for a continuous source, averaged over sky position, inclination, and polarization angles
        """
        # rf = 1.0 / (12 * np.pi**2 * f**2)
        return np.sqrt(seff / tobs)

    def plot_tf_hc(self):
        """
        plots T(f) and hc(f).
        """
        basename = os.path.splitext(os.path.basename(self.h5file.filename))[0]
        direc = os.path.join(
            RESULTSDIR, "systematics", "tests", basename, model.model_name
        )
        if not os.path.exists(direc):
            os.makedirs(direc)
        ## plot T(f)
        plt.plot(self.f, np.array(self.tf).T, color="gray", alpha=0.3)
        plt.axhline(1.0, color="black", ls="dashed")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("f [Hz]")
        plt.ylabel(r"$T(f)$")
        tffile = os.path.join(direc, "transmission_func.png")
        plt.savefig(tffile)
        plt.close()
        ## plot hc(f)
        plt.plot(self.f, np.array(self.sens_curves).T, color="gray", alpha=0.3)
        plt.plot(self.f, self.tot_sens_curve, color="black", alpha=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("f [Hz]")
        plt.ylabel(r"$h_c(f)$")
        hcfile = os.path.join(direc, "estimated_hc.png")
        plt.savefig(hcfile)
