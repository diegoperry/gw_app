import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.timeseries import LombScargle
import os
import sys
import argparse
from astropy.io import fits

from kepler_astrometry_catalog.constants import PIX_SCALE_MAS
import kepler_astrometry_catalog.clean as c
import kepler_astrometry_catalog.get_star_catalog as gsc
from kepler_astrometry_catalog.paths import RESULTSDIR


## first load the catalog.
QTR = 12
SAMPLEID = "brightestnonsat100"
fitsfn = "../results/cleaned_centroids/brightestnonsat100_rot.fits"


print(f"Starting analysis for quarter {QTR} w/ sampleid: {SAMPLEID}")
with fits.open(fitsfn) as hdu:
    time = hdu["TIME"].data["time"][0]  # in days, (Ntimes)
    kicids = hdu["PATHS"].data["kicid"]
    ra_array = hdu["FAKE_GLOB_MOM_POSCORR_RESIDUAL"].data[
        "CENTROID_X"
    ]  # in deg, (N_output, Ntimes)
    dec_array = hdu["FAKE_GLOB_MOM_POSCORR_RESIDUAL"].data[
        "CENTROID_Y"
    ]  # in deg, (N_output, Ntimes)
    rawx = hdu["MOM_RAW_CENTROIDS"].data["RAW_CENTROID_X"]
    rawy = hdu["MOM_RAW_CENTROIDS"].data["RAW_CENTROID_Y"]
    poscorrx = hdu["POSCORR"].data["CORR_X"]
    poscorry = hdu["POSCORR"].data["CORR_Y"]

## subtract poscorr from raw
X = rawx - poscorrx
# X = rawx
Y = rawy - poscorry
# Y = rawy

## get shapes
nstars = np.shape(rawx)[0]
ntimes = np.shape(rawx)[1]


## get xi eta from ra dec
def ra_dec_to_xi_eta(ra, dec, wcs):
    """
    Transforms from (RA, Dec) to (xi, eta) WCS coordinates.

    Parameters:
    ra (float or array-like): RA coordinate(s) in degrees
    dec (float or array-like): Dec coordinate(s) in degrees
    wcs (astropy.wcs.WCS): WCS object that includes the transformation information

    Returns:
    tuple: (xi, eta) coordinates
    """
    # Convert RA, Dec to pixel coordinates
    pixel_coords = wcs.wcs_world2pix(ra, dec, 0)

    # Convert pixel coordinates to (xi, eta)
    xi_eta = wcs.wcs_pix2world(pixel_coords[0], pixel_coords[1], 0)

    return xi_eta[0], xi_eta[1]


# Create a WCS object. ##FIXME!!
# w = WCS(naxis=2)
# w.wcs.crpix = [1, 1]
# w.wcs.cdelt = np.array([-0.066667, 0.066667])
# w.wcs.crval = [0, 0]
# w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

# Path to the FITS file ##FIXME this is a random one.
wcs_file = (
    "../data/lightcurves/Kepler/0008/000893233/kplr000893233-2012088054726_llc.fits"
)

# Load WCS data from the FITS file
with fits.open(wcs_file) as hdul:
    w = WCS(hdul[0].header)


# Transform (RA, Dec) to (xi, eta)
xi_array, eta_array = ra_dec_to_xi_eta(ra_array, dec_array, w)


# Initialize arrays to store the coefficients for each time step
A_coefficients = np.zeros(ntimes)
B_coefficients = np.zeros(ntimes)
C_coefficients = np.zeros(ntimes)
F_coefficients = np.zeros(ntimes)


def model(x, y, A, B, C, F):
    ## Eqns. 3 & 4 from Benedict et al. 2014.
    xi = A * x + B * y + C
    eta = -B * x + A * y + F
    return xi, eta


# Solve for the coefficients at each time step
for t in range(ntimes):
    x_t = X[:, t]
    y_t = Y[:, t]
    xi_t = xi_array[:, t]
    eta_t = eta_array[:, t]

    # Flatten the arrays for the current time step
    x_flat = x_t.flatten()
    y_flat = y_t.flatten()
    xi_flat = xi_t.flatten()
    eta_flat = eta_t.flatten()

    # Number of data points for the current time step
    n = x_flat.size

    # Constructing the matrix and the right-hand side vector
    A_matrix = np.zeros((2 * n, 4))
    b_vector = np.zeros(2 * n)

    for i in range(n):
        x = x_flat[i]
        y = y_flat[i]
        xi = xi_flat[i]
        eta = eta_flat[i]

        A_matrix[2 * i] = [x, y, 1, 0]
        A_matrix[2 * i + 1] = [-y, x, 0, 1]

        b_vector[2 * i] = xi
        b_vector[2 * i + 1] = eta

    # Solving the system using least squares
    coefficients, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

    A, B, C, F = coefficients

    # Store the coefficients for the current time step
    A_coefficients[t] = A
    B_coefficients[t] = B
    C_coefficients[t] = C
    F_coefficients[t] = F

print(np.nanmean(A_coefficients))
print(np.nanmean(B_coefficients))
print(np.nanmean(C_coefficients))
print(np.nanmean(F_coefficients))


plt.plot(time, A_coefficients, label="A")
plt.plot(time, B_coefficients, label="B")
plt.xlabel("kic time")
plt.ylabel("Coeff. value")
plt.legend()
plt.show()


plt.plot(time, C_coefficients, label="C")
plt.plot(time, F_coefficients, label="F")
plt.xlabel("kic time")
plt.ylabel("Coeff. value")
plt.legend()
plt.show()

modelxi, modeleta = model(
    X, Y, A_coefficients, B_coefficients, C_coefficients, F_coefficients
)
# diffx = np.nanmean(modelxi - xi_array, axis=0)
# diffy = np.nanmean(modeleta - eta_array, axis=0)

# plt.plot(time, diffx/np.nanmean(xi_array), label='xi')
# plt.plot(time, diffy/np.nanmean(eta_array), label='eta')
# plt.legend()
# plt.show()

plt.plot(time, modelxi[0])
plt.plot(time, xi_array[0, :], color="k")
plt.ylabel(r"$\xi$")
plt.xlabel("kic time")
plt.savefig("../results/plate_scale/xi_model.png")

plt.close()
plt.plot(time, modeleta[0])
plt.plot(time, eta_array[0, :], color="black")
plt.xlabel("kic time")
plt.ylabel(r"$\eta$")
plt.savefig("../results/plate_scale/eta_model.png")
