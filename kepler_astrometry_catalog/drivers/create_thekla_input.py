from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from importlib import reload
from astropy.table import Table
import pandas as pd
from astropy.io import fits, misc
import os
import h5py
from astropy.coordinates import SkyCoord
import sys
import argparse

from dotenv import find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")

from kepler_astrometry_catalog.paths import *
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict
from kepler_astrometry_catalog.clean import Clean
from kepler_astrometry_catalog.get_centroids import get_raw_centroids
from estoiles.gw_calc import GWcalc
import tables
from kepler_astrometry_catalog.constants import PIX_TO_DEG
from kepler_astrometry_catalog.covariance import get_inv_cov
from kepler_astrometry_catalog.dva import DVA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5file", help="Input hdf5 filename")
    parser.add_argument("--hdf5_path", help="Output hdf5 path")
    parser.add_argument("--quarter", type=int, default=12, help="The quarter")
    parser.add_argument(
        "--fake-signal", action="store_true", help="Set the add_fake_signal"
    )

    args = parser.parse_args()

    quarter = args.quarter
    fake_signal = args.fake_signal
    h5file = args.h5file
    hdf5_path = args.hdf5_path
    # mo_fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    # hdf5_path = RESULTSDIR + f"/for_thekla/{sampleid}_{quarter}.hdf5"
    if fake_signal:
        ## FIXME
        print("using fake signal")
        hdf5_path = hdf5_path.replace(".hdf5", "_fake.hdf5")

    with tables.open_file(h5file, mode="r") as f:
        group = f.root.Q12
        kicids = group.kicid
        pos = group.pos
        time = pos[0]["time"]  ## FIXME: thekla only takes 1 time right now?
        # x = pos[:]["x"]
        # y = pos[:]["y"]

        # Match kicids in group.stars["kicid"] and extract corresponding ra and dec
        star_kicids = f.root.stars[:]["kicid"]
        star_ra = f.root.stars[:]["ra"]
        star_dec = f.root.stars[:]["dec"]

        # Create a mapping from kicid to index in stars table
        kicid_to_idx = {k: i for i, k in enumerate(star_kicids)}
        indices = [kicid_to_idx[k["kicid"]] for k in kicids]

        ra0 = star_ra[indices]
        dec0 = star_dec[indices]

        # Get cov mat
        model = DVA(quarter=12, h5file=f, order=4)  ## FIXME: should be an input.
        # Get x and y residuals for the selected kicids
        x = np.array([model.results[k["kicid"]]["x_residuals"] for k in kicids])
        y = np.array([model.results[k["kicid"]]["y_residuals"] for k in kicids])
        invcovlist = get_inv_cov(f, model)
    # FIXME: need to actually convert x,y -- for now, just adding them to ra, dec
    ra = (x * PIX_TO_DEG) + ra0[:, np.newaxis]
    dec = (y * PIX_TO_DEG) + dec0[:, np.newaxis]

    ## copying from Ben's estoiles within thekla. see mock
    kcoords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    telcoord = SkyCoord(
        ra="19h22m40s", dec="+44:30:00", unit=(u.hourangle, u.deg), frame="icrs"
    )  ## his default Kepler coord.
    posxyz = [GWcalc.coordtransform(telcoord.galactic, k) for k in kcoords.galactic]
    posxyz = np.swapaxes(posxyz, 1, 2)

    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("t_", data=time)
        f.create_dataset("star_pos", data=posxyz)
        f.create_dataset("invcov_arr", data=np.array(invcovlist))

    print(f"Saved to {hdf5_path}.")
