"""

Creates the inverse covariance for a given set of stars and systematics models.

** currently assumes only deterministic noise & diagonal inherent data noise.

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import tables  # PyTables

from kepler_astrometry_catalog.paths import *
from kepler_astrometry_catalog.dva import DVA
from kepler_astrometry_catalog.constants import PIX_SCALE_MAS


MAS_TO_RAD = (1.0 * u.mas).to(u.rad).value


def get_d_inv(M, N, Ninv=None):
    ## Eqn. 34 of NG 15 yr Background Methods paper. 2306.16223.
    if Ninv is None:
        Ninv = np.linalg.inv(N)
    Lam = M.T @ Ninv @ M
    Laminv = np.linalg.inv(Lam)
    return Ninv - Ninv @ M @ Laminv @ M.T @ Ninv


def get_inv_cov(h5file, model):
    group = h5file.root.Q12
    kicids = group.kicid
    pos = group.pos
    invcovs = []
    for krow, prow in zip(kicids, pos):
        jac = model.jacobian(prow["time"], kicid=krow["kicid"])
        noise = np.eye(2 * len(prow["time"]))
        xnoise = model.results[krow["kicid"]]["std_x"] ** 2
        ynoise = model.results[krow["kicid"]]["std_y"] ** 2
        noise[: len(prow["time"])] *= xnoise
        noise[len(prow["time"]) :] *= ynoise
        noiseinv = np.eye(2 * len(prow["time"]))
        noiseinv[: len(prow["time"])] *= 1.0 / xnoise
        noiseinv[len(prow["time"]) :] *= 1.0 / ynoise
        Dinv = get_d_inv(np.vstack([jac[0], jac[1]]), noise, noiseinv)
        invcovs.append(Dinv)
    return invcovs


def plot_inv_cov(Dinv):
    basename = os.path.splitext(os.path.basename(self.h5file.filename))[0]
    direc = os.path.join(RESULTSDIR, "systematics", "tests", basename, model.model_name)
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Dinv, origin="upper", cmap="viridis", norm=LogNorm(), interpolation="none"
    )
    plt.title(r"Inv Cov Matrix (D$^{-1})$")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.colorbar(label="Covariance Value")
    invcovplotfn = os.path.join(direc, "invcov.png")
    plt.savefig(invcovplotfn)
