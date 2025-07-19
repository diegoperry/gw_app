"""
This script defines the function `get_raw_centroids`, which extracts raw centroids 
from Kepler light curves for a given quarter and `sampleid`.

It calls the function `given_lcpath_get_time_ctd_dva` from the `helpers.py` script 
for each star in the `sampleid` catalog. The function follows this pipeline:

Parameters:
    lc_path (str): 
        Path to the Kepler light curve file 
        (e.g., `/data/lightcurves/Kepler/xxxx/xxxxxxx/kplrxxxxxxx-xxxx_llc.fits`).
    verbose (bool, optional): 
        Whether to print additional logs. Default is `False`.
    reload (bool, optional): 
        Whether to reload the data. Default is `False`, determined by `GET_CENTROIDS`
        in the `pre-clean.yaml` file.
    ppa (bool, optional): 
        Set to `True` if `"ppa"` is in `sampleid`. Default is `False`.
    local_only (bool, optional): 
        Whether to limit operations to local processing only. Default is `False`, 
        determined by `LOCAL_ONLY` in the `pre-clean.yaml` file.
    get_psf (bool, optional): 
        Whether to extract point spread function (PSF) centroids. Default is `False`, 
        determined by `GET_PSF` in the `pre-clean.yaml` file.
    get_poscorr (bool, optional): 
        Whether to extract position correction values. Default is `False`, 
        determined by `GET_POSCORR` in the `pre-clean.yaml` file.

Pipeline:
    1. Creates a `ctd` directory in the same parent folder as `lightcurve` 
       (e.g., `/data/ctd`).
    2. Saves centroid data in:
       `/data/ctd/Kepler/xxxx/xxxxxxx/kplrxxxxxxx-xxxx_llc_rawctds.csv`.
    3. Reads data from the Kepler light curve file, extracting:
        - `SAP_QUALITY`: Simple aperture photometry quality flag.
        - `TIME`: Properly formatted Kepler observation time.
        - `CADENCENO`: Cadence number.
        - `MOM_CENTR1`: Moment centroid (row, y).
        - `MOM_CENTR2`: Moment centroid (column, x).
        - If `get_psf` is `True`, also extracts:
            * `PSF_CENTR1`: PSF centroid (row, y).
            * `PSF_CENTR2`: PSF centroid (column, x).
        - If `get_poscorr` is `True`, also extracts:
            * `POS_CORR1`: Position correction (row, y).
            * `POS_CORR2`: Position correction (column, x).
    4. Uses `RaDec2Pix` to compute:
        - `dva_x`, `dva_y`: Differential velocity aberration (DVA).
        - Module and output values for moment centroids.
        - If `get_psf` is `True`, also computes DVA for PSF centroids.
    5. If `local_only` is `False`, uses `Pix2RaDec` to compute:
        - Right ascension (RA) and declination (Dec) for moment centroids.
        - If `get_psf` is `True`, also computes RA, Dec for PSF centroids.
    6. Saves processed data to:
       `/data/ctd/Kepler/xxxx/xxxxxxx/kplrxxxxxxx-xxxx_llc_rawctds.csv`.

Returns:
    None. The function processes the data and saves it to the specified location.
"""
#
# standard imports
#
from astropy.io import fits
import os
from glob import glob
import numpy as np
import pandas as pd

#
# non-standard imports
#
from kepler_astrometry_catalog.paths import RESULTSDIR, DATADIR
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict
from kepler_astrometry_catalog.helpers import (
    get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o,
    given_lcpath_get_time_ctd_dva,
)
from kepler_astrometry_catalog.clean import Clean
from kepler_astrometry_catalog.fake_signal import FakeSignal
import astropy.units as u


def get_raw_centroids(
    quarter,
    sampleid="brightestnonsat100_rot",
    batch_process=0,
    max_nstars=100,
    ibatch=-1,
    seed=42,
    verbose=1,
    reload=0,
    local_only=0,
    get_psf=0,
    get_poscorr=0,
):
    if "ppa" in sampleid:
        PPA = True
    else:
        PPA = False

    lcdir = os.path.join(DATADIR, "lightcurves", "Kepler")
    if verbose:
        print(f"lightcurve directory: {lcdir}", flush=1)
    if not os.path.exists(lcdir):
        print(f"{lcdir} does not exist", flush=1)
        return

    fn = os.path.join(RESULTSDIR, f"tables/{sampleid}.csv")
    df = pd.read_csv(fn)
    if verbose and batch_process:
        print(f"total number of stars in catalog: {len(df)}", flush=1)

    if batch_process:
        df = df.iloc[ibatch * max_nstars : (ibatch + 1) * max_nstars]
        if verbose:
            print(
                f"batch star indices: {ibatch*max_nstars} to {(ibatch+1)*max_nstars}",
                flush=1,
            )

    ## get kicids for catalog and randomize
    qrtr_mask = df[f"inQ{str(quarter)}"] == 1
    cat_kicids = df["kicid"][qrtr_mask].to_numpy()
    nstars = len(cat_kicids)
    if len(cat_kicids) > max_nstars:
        np.random.seed(seed)
        nstars = max_nstars
        cat_kicids = np.random.choice(cat_kicids, max_nstars, replace=False)

    datestr = quarter_datestr_dict[quarter]
    zfillkcids = [str(kc)[:-5].zfill(4) for kc in cat_kicids]
    zkcs_long = [str(kc).zfill(9) for kc in cat_kicids]
    lcpaths = [
        f"{lcdir}/{zkc}/{zkcl}/kplr{zkcl}-{datestr}_llc.fits"
        for (zkc, zkcl) in zip(zfillkcids, zkcs_long)
    ]

    if verbose:
        print("Number of stars in catalog =", len(df), flush=1)
        print("Number of stars chosen = ", str(nstars), flush=1)

    for ix, lcpath in enumerate(lcpaths):
        if verbose:
            if np.mod(ix + 1, 10) == 0:
                print(f"finished getting {ix+1}/{nstars} centroid data", flush=1)
        _ = given_lcpath_get_time_ctd_dva(
            lcpath,
            verbose=verbose,
            reload=reload,
            ppa=PPA,
            local_only=local_only,
            get_psf=get_psf,
            get_poscorr=get_poscorr,
        )
    return
