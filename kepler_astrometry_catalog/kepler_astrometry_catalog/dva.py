#!/usr/bin/env python
"""

Script to fit the DVA model.

coded with help from chatgpt.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u
import time

from kepler_astrometry_catalog.systematic_model import SystematicModel


class DVA(SystematicModel):
    param_names = ["x_params", "y_params"]
    model_name = "dva"

    def __init__(
        self,
        fdva=(1.0 / (372.57 * u.day)).to(1.0 / u.s).value,
        order=4,
        quarter=12,
        h5file=None,
    ):
        self.fdva = fdva
        self.order = order
        self.quarter = quarter
        self.results = {}
        super().__init__(h5file)

    def _dva_model(self, t, phase, A, B, C, D, constant):
        terms = [
            A * np.cos(2 * np.pi * self.fdva * t + phase),
            B * (np.cos(2 * np.pi * self.fdva * t + phase)) ** 2,
            C * (np.cos(2 * np.pi * self.fdva * t + phase)) ** 3,
            D * (np.cos(2 * np.pi * self.fdva * t + phase)) ** 4,
        ]
        return np.sum(terms[: self.order], axis=0) + constant

    def fit(self, h5file):
        start = time.time()
        group = getattr(h5file.root, f"Q{self.quarter}")
        pos = group.pos
        kicids = group.kicid
        self.results = {}
        self.params["x_params"] = {}
        self.params["y_params"] = {}

        std_x_all = []
        std_y_all = []

        for i in range(len(pos)):
            t = pos[i]["time"]
            x = pos[i]["x"]
            y = pos[i]["y"]
            kicid = kicids[i]["kicid"]

            p0x = [0.0, np.std(x), 0.0, 0.0, 0.0, np.mean(x)]
            p0y = [0.0, np.std(y), 0.0, 0.0, 0.0, np.mean(y)]

            try:
                popt_x, _ = curve_fit(self._dva_model, t, x, p0=p0x)
                popt_y, _ = curve_fit(self._dva_model, t, y, p0=p0y)

                x_fit = self._dva_model(t, *popt_x)
                y_fit = self._dva_model(t, *popt_y)

                res_x = x - x_fit
                res_y = y - y_fit
                std_x = np.std(res_x)
                std_y = np.std(res_y)
                std_x_all.append(std_x)
                std_y_all.append(std_y)

                self.params["x_params"][kicid] = popt_x
                self.params["y_params"][kicid] = popt_y

                self.results[kicid] = {
                    "x_residuals": res_x,
                    "y_residuals": res_y,
                    "std_x": std_x,
                    "std_y": std_y,
                }
            except Exception as e:
                print(f"Fit failed for kicid {kicid}: {e}")

        self.validate_params()
        self.record_fit_metadata(
            num_stars=len(self.params["x_params"]),
            mean_std_x=np.mean(std_x_all),
            mean_std_y=np.mean(std_y_all),
            quarter=self.quarter,
            fdva=self.fdva,
            order=self.order,
            fit_duration=time.time() - start,
        )

    def predict(self, t, kicid=None):
        if kicid is None:
            raise ValueError("kicid must be provided for prediction.")

        x_fit = self._dva_model(t, *self.params["x_params"][kicid])
        y_fit = self._dva_model(t, *self.params["y_params"][kicid])
        return x_fit, y_fit

    def jacobian(self, time, kicid=None, fit_phase=True, fit_constant=True):
        if kicid is None:
            raise ValueError("kicid must be provided for prediction.")
        # get bestfit params
        phix, Ax, Bx, Cx, Dx, __ = self.params["x_params"][kicid]
        phiy, Ay, By, Cy, Dy, __ = self.params["y_params"][kicid]
        # define phase arg
        argx = 2 * np.pi * self.fdva * time + phix
        argy = 2 * np.pi * self.fdva * time + phiy
        # set up column containers
        terms_x = []
        terms_y = []
        # if fit_phase, add columns for phase
        if fit_phase:
            dndphix = (
                -Ax * np.sin(argx)
                - 2 * Bx * np.cos(argx) * np.sin(argx)
                - 3 * Cx * np.cos(argx) ** 2 * np.sin(argx)
                - 4 * Dx * np.cos(argx) ** 3 * np.sin(argx)
            )
            terms_x.append(dndphix)
            dndphiy = (
                -Ay * np.sin(argy)
                - 2 * By * np.cos(argy) * np.sin(argy)
                - 3 * Cy * np.cos(argy) ** 2 * np.sin(argy)
                - 4 * Dy * np.cos(argy) ** 3 * np.sin(argy)
            )
            terms_y.append(dndphiy)
        # add terms for cosine series
        terms_x += [np.cos(argx) ** n for n in range(1, self.order + 1)]
        terms_y += [np.cos(argy) ** n for n in range(1, self.order + 1)]
        # if fit_constant, add column for constant
        if fit_constant:
            terms_x.append(np.ones_like(argx))
            terms_y.append(np.ones_like(argy))
        # stack columns to make jacobians
        Mx = np.stack(terms_x, axis=-1)
        My = np.stack(terms_y, axis=-1)
        return Mx, My
