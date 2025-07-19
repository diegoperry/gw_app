"""
Contents:

Pipeline helpers:
    get_lcpaths_given_sampleid_seed_nsamples_quarter
    get_pca_basisvecs
    given_lcpath_get_time_ctd_dva
    interp_nans_mdva

Generally useful:
    get_astrometric_precision
"""

import os
import numpy as np, pandas as pd
from numpy import array as nparr
from os.path import join
from copy import deepcopy
from numpy import polyfit, poly1d
from astropy.io import fits
from astropy.time import Time
from scipy.interpolate import interp1d

from kepler_astrometry_catalog.paths import RESULTSDIR, DATADIR

# from kepler_astrometry_catalog.cotrending import get_raDec2Pix,get_Pix2RaDec
from pathlib import Path
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict

####################
### utility func ###
####################

def slurm_split(n_obj, *args):
    '''Take a variable number of same-length arrays, and split them according to
    environment variables set by SLURM.

    Args:
        n_obj (int): Length of all arrays in *args.

    Returns tuple:
        split_args: Tuple of *args, but split, indexed by the task index.
        offset: int describing array index that our split starts at.
    '''
    if 'SLURM_ARRAY_TASK_COUNT' in os.environ and 'SLURM_ARRAY_TASK_ID' in os.environ:
        use_job_array = True
        n_tasks_env_var = 'SLURM_ARRAY_TASK_COUNT'
        task_ind_env_var = 'SLURM_ARRAY_TASK_ID'
        # raise RuntimeWarning('SLURM job arrays are not supported by Maestro.')
    else:
        use_job_array = False
        n_tasks_env_var = 'SLURM_NTASKS'
        task_ind_env_var = 'SLURM_PROCID'
    if n_tasks_env_var in os.environ and int(os.environ[n_tasks_env_var]) > 1:
        n_tasks = int(os.environ[n_tasks_env_var])
        if use_job_array:
            # Job arrays can have any arbitrary lower index. Safeguard against accidentally submitting it with 1-N values.
            task_ind = int(os.environ[task_ind_env_var]) - int(os.environ['SLURM_ARRAY_TASK_MIN'])
        else:
            task_ind = int(os.environ[task_ind_env_var])
        assert task_ind < n_tasks and n_tasks <= n_obj
        assert np.all([len(a) == n_obj for a in args])
        
        split_args = [np.array_split(a, n_tasks)[task_ind] for a in args]
        split_first_arg = np.array_split(args[0], n_tasks)
        offset = np.sum([len(split_first_arg[i]) for i in range(task_ind)], dtype=int)

        return split_args, offset
    else:
        return args, 0

def get_df_keyword(csvpath, key):
    df = pd.read_csv(csvpath)
    return np.array(df[key])


from raDec2Pix import raDec2Pix

rdp = raDec2Pix.raDec2PixClass()


def get_raDec2Pix(ra, dec, time):
    """
    ra/dec: degrees
    time: Astropy.Time object
    """
    m, o, r, c = rdp.ra_dec_2_pix(ra, dec, time.mjd)
    return m, o, r, c


def get_Pix2RaDec(module, output, row, column, time, aberrate):
    ra, dec = rdp.pix_2_ra_dec(
        module, output, row, column, time.mjd, aberrateFlag=aberrate
    )
    return ra, dec


def _get_data_keyword(fits_file, keyword, ext=1):
    """
    (copied directly from astrobase; credit Waqas Bhatti if you use this
    function; pasted here to avoid import dependencies)

    This gets a single data array out of a FITS binary table.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword : str
        Keyword to access in the header.

    ext : int
        FITS extension number.

    Returns
    -------
    val : ndarray
        The value in the FITS data table header.

    """

    hdulist = fits.open(fits_file)

    if keyword in hdulist[ext].data.names:
        val = hdulist[ext].data[keyword]
    else:
        val = None

    hdulist.close()
    return val


def PCA_live(arr):
    from sklearn.decomposition import PCA

    mean = np.mean(arr, axis=0)
    pca = PCA()
    pca.fit(arr)
    return mean, pca.components_


deg2rad = np.pi / 180
rad2mas = 3600 * 180 / np.pi * 1000


def conv2cart(ra_, dec_):
    ## return the cartesian coordinates of displacement in mas
    mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
    mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
    dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
    dy = (dec_ - mean_dec) * deg2rad * rad2mas
    return dx, dy


def demean(arr):
    ## remove both the time-wise mean and per-time-slice mean
    arr_ = copy.deepcopy(arr)
    mean = np.mean(arr_, axis=1)[:, np.newaxis]
    arr_ -= mean
    mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
    arr_ -= mean_eigvec
    return arr_


def corr(x, y):
    return x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)


def centering(arr):
    ## remove the time-wise mean
    return arr - np.mean(arr, axis=1)[:, np.newaxis]


####################
### utility func ####
####################


def get_fake_local(glob_dx, glob_dy, module, output, output_axis_map):
    """This function take global signals e.g. injected GW and simulate what they would look locally. Notice that this is still the post-DVA residual in the local frame, so if you were to compare it to raw centroids, be sure to add DVA. output_axis_map is a dictionary with axis vectors for each module and output; call cache_module_output_axis to get this."""

    # generate the local signals
    loc_drow = np.empty([len(glob_dx), len(glob_dx[0])])
    loc_dcol = np.empty([len(glob_dy), len(glob_dx[0])])
    groups = [
        [18, 19, 22, 23, 24],
        [6, 11, 12, 13, 16, 17],
        [9, 10, 14, 15, 20],
        [2, 3, 4, 7, 8],
    ]
    group_key = {}
    for igroup, gp in enumerate(groups):
        for i in gp:
            group_key[i] = igroup

    for istar, (m, o) in enumerate(zip(module, output)):
        rowvec, colvec = output_axis_map[(int(group_key[m]), int(o))]
        # new axis
        x1, y1 = np.mean(rowvec, axis=1)
        norm = np.sqrt(x1**2 + y1**2)
        x1 /= norm
        y1 /= norm
        x2, y2 = np.mean(colvec, axis=1)
        n = np.sqrt(x2**2 + y2**2)
        x2 /= norm
        y2 /= norm

        x0, y0 = glob_dx[istar], glob_dy[istar]

        X1 = (y0 - y2 / x2 * x0) / (y1 / x1 - y2 / x2)
        Y1 = y2 / x2 * X1 + (y0 - y2 / x2 * x0)
        loc_drow[istar] = X1 * x1 + Y1 * y1
        X2 = (y0 - y1 / x1 * x0) / (y2 / x2 - y1 / x1)
        Y2 = y1 / x1 * X2 + (y0 - y1 / x1 * x0)
        loc_dcol[istar] = X2 * x2 + Y2 * y2
    return loc_drow, loc_dcol


def get_fake_global(loc_drow, loc_dcol, module, output, quarter, time):
    """This function is effectively the inverse transformation of get_fake_local"""

    fn = f"{RESULTSDIR}/temporary/Q{quarter}_output_axis_T{len(time)}.npy"
    output_axis_map = np.load(fn, allow_pickle=True).item()

    glob_dx = np.empty([len(loc_drow), len(loc_drow[0])])
    glob_dy = np.empty([len(loc_dcol), len(loc_dcol[0])])

    groups = [
        [18, 19, 22, 23, 24],
        [6, 11, 12, 13, 16, 17],
        [9, 10, 14, 15, 20],
        [2, 3, 4, 7, 8],
    ]
    group_key = {}
    for igroup, gp in enumerate(groups):
        for i in gp:
            group_key[i] = igroup

    for istar, (m, o) in enumerate(zip(module, output)):
        rowvec, colvec = output_axis_map[(int(group_key[m]), int(o))]
        row0, vec0 = loc_drow[istar], loc_dcol[istar]
        glob_dx[istar] = row0 * rowvec[0] + vec0 * colvec[0]
        glob_dy[istar] = row0 * rowvec[1] + vec0 * colvec[1]
        if np.mod(istar + 1, 500) == 0:
            print(f"finished {istar+1} stars", flush=1)
    return glob_dx, glob_dy


def cache_module_output_axis(quarter, time, force_refresh=False):
    """This function cache the axis vector for each module and output group (that are axis-aligned) at the requested time."""
    from raDec2Pix import raDec2Pix
    from raDec2Pix import raDec2PixModels as rm

    fn = f"{RESULTSDIR}/temporary/Q{quarter}_output_axis_T{len(time)}.npy"
    print("force refresh is:", force_refresh)
    if (os.path.exists(fn)) and (not force_refresh):
        print(f"{fn} already exists", flush=1)
        return
    if not os.path.exists(fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
    print("axis data not available, need to recompute", flush=1)
    time_bjd = time + 2454833  # bjdrefi
    time_obj = Time(time_bjd, format="jd", scale="tdb")
    print(f"number of time slices: {len(time_obj)}", flush=1)

    rdp = raDec2Pix.raDec2PixClass()
    NCOL, NROW = rm.get_parameters("nColsImaging"), rm.get_parameters("nRowsImaging")
    groups = [
        [18, 19, 22, 23, 24],
        [6, 11, 12, 13, 16, 17],
        [9, 10, 14, 15, 20],
        [2, 3, 4, 7, 8],
    ]
    all_module = [groups[i][0] for i in range(len(groups))]
    group_key = {}
    for igroup, gp in enumerate(groups):
        for i in gp:
            group_key[i] = igroup
    all_module, all_output = np.meshgrid(all_module, np.arange(1, 5))
    all_module = all_module.flatten()
    all_output = all_output.flatten()

    def get_unit_vec(start_ra, end_ra, start_dec, end_dec):
        dx = (end_ra - start_ra) * np.cos(np.deg2rad(start_dec))
        dy = end_dec - start_dec
        xvec = np.array([dx, dy])
        xvec /= np.linalg.norm(xvec, axis=0)
        return xvec

    hm = dict()

    for m, o in zip(all_module, all_output):
        origin_ra, origin_dec = rdp.pix_2_ra_dec(
            m, o, 0.0, 0.0, time_obj.mjd, aberrateFlag=1
        )
        xmax_ra, xmax_dec = rdp.pix_2_ra_dec(
            m, o, NROW, 0.0, time_obj.mjd, aberrateFlag=1
        )
        ymax_ra, ymax_dec = rdp.pix_2_ra_dec(
            m, o, 0.0, NCOL, time_obj.mjd, aberrateFlag=1
        )

        xvec = get_unit_vec(origin_ra, xmax_ra, origin_dec, xmax_dec)
        yvec = get_unit_vec(origin_ra, ymax_ra, origin_dec, ymax_dec)

        hm[(int(group_key[m]), o)] = np.array([xvec, yvec])
        print(f"finished module {m} output {o}", flush=1)

    np.save(fn, hm)
    print("finished caching output axis results", flush=1)
    return


def given_lcpath_get_time_ctd_dva(
    lcpath,
    verbose=0,
    process_raw=1,
    reload=0,
    ppa=0,
    local_only=0,
    get_psf=0,
    get_poscorr=0,
):
    """
    Args:

        lcpath (fits)

        verbose: prints stuff

        return_sel (bool): if true, returns the selected (quality==0) points.
        Otherwise, returns all points, which includes NaNs.

        process_raw (bool): if false, only checks if this csv exists.
        local_only (bool): if true, ignore the iterative process to find ra and dec
        get_psf (bool): if true, add PSF_CENTR (MOM_CENTR is also saved)
        get_poscorr (bool): if true, add POS_CORR

    Returns

        hdr: of the FITS light curve

        df: DataFrame containing keys

            time, mom_x, mom_y, dva_x, dva_y, module, output (optional: psf_x, psf_y, poscorr_x, poscorr_y, ra, dec)

            where

            mom_* = MOM_CENTR,
            dva_* = Bryson's DVA model prediction,
            module = detector module,
            output = detector output,
            psf_* = PSF_CENTR,
            poscorr_* = POS_CORR,

            and "_x" means CCD column, and "_y" means CCD row.

    NOTE: automatically caches the dataframe to a CSV file, since running
    Bryson's DVA code is a bit slow.
    """
    # if ppa:
    #     ctdfolder = "/".join(lcpath.split("/")[:-1])
    #     ctdfolder = ctdfolder.replace("lightcurves/Kepler", "ppa")
    #     Path(ctdfolder).mkdir(parents=True, exist_ok=True)

    #     selcsvpath = lcpath.replace(".fits", "_ppa.csv")
    #     selcsvpath = selcsvpath.replace("lightcurves/Kepler", "ppa")
    #     rawcsvpath = selcsvpath.replace("_ppa.csv", "_rawppa.csv")

    # else:
    ctdfolder = "/".join(lcpath.split("/")[:-1])
    ctdfolder = ctdfolder.replace("lightcurves/Kepler", "ctd")
    Path(ctdfolder).mkdir(parents=True, exist_ok=True)

    csvpath = lcpath.replace(".fits", "_rawctds.csv")
    rawcsvpath = csvpath.replace("lightcurves/Kepler", "ctd")

    hdulist = fits.open(lcpath)
    d = hdulist[1].data
    hdr = hdulist[0].header

    if not reload:
        if os.path.exists(rawcsvpath):
            return hdr, d, pd.read_csv(rawcsvpath)

    ## FIXME to be removed later! this is just to save time now (July 2024)
    if reload:
        if os.path.exists(rawcsvpath):
            table = pd.read_csv(rawcsvpath)
            try:
                __ = table["poscorr_x"]
                __ = table["ra"]
                print(f"CSV exists and has poscorr and global coords {rawcsvpath}.")
                return hdr, d, pd.read_csv(rawcsvpath)
            except:
                print(
                    f"CSV exists but does not have poscorr and/or global coords {rawcsvpath}."
                )
                # continue with the rest of the code
    # if not process_raw:
    #     print(
    #         f"csv not found for {selcsvpath}, need to process raw lightcurves...",
    #         flush=1,
    #     )
    #     return None, None, None

    quarter = hdulist[0].header["QUARTER"]
    keplerid = hdulist[0].header["KEPLERID"]
    ra, dec = hdulist[0].header["RA_OBJ"], hdulist[0].header["DEC_OBJ"]
    bjdrefi = hdulist[1].header["BJDREFI"]
    hdulist.close()

    # get data
    _qual = d["SAP_QUALITY"]
    _time = d["TIME"]
    _cadenceno = d["CADENCENO"]

    # by default get MOM_CENTR, if requested, get psf and poscorr
    if get_psf:
        try:
            _psf_x = d["PSF_CENTR2"]
            _psf_y = d["PSF_CENTR1"]
            if np.all(np.isnan(_psf_x)):
                print("this star doesn't have PSF_CENTR. Turning off the flag")
                get_psf = False
        except:
            print("this star doesn't have PSF_CENTR. Turning off the flag")
            get_psf = False
            # print(
            #     "requested star doens't have PSF_CENTR, need to set get_psf=0", flush=1
            # )
            # exit()
    if get_poscorr:  # FIXME: if no PSF, we automatically don't have POSCORR?
        _poscorr_x = d["POS_CORR2"]
        _poscorr_y = d["POS_CORR1"]

    _ctd_x = d["MOM_CENTR2"]  # column FIXME: why is this flipped?
    _ctd_y = d["MOM_CENTR1"]  # row

    # clean time slices with no dataframe
    sel = ~np.isnan(_ctd_x)

    if get_psf:
        psfsel = ~np.isnan(_psf_x) & sel
        pseli = np.argwhere(psfsel).flatten()
    time, ctd_x, ctd_y = _time[sel], _ctd_x[sel], _ctd_y[sel]
    if get_psf:
        psf_x, psf_y = _psf_x[psfsel], _psf_y[psfsel]
        ptime = _time[psfsel]
    cadenceno = _cadenceno[sel]

    if sum(sel) == 0:
        print(f"no valid time slice measurements for {lcpath}", flush=1)
        return

    def fill_nan(arr_val, i, size):
        res = np.ones(size) * np.nan
        res[i] = arr_val
        return res

    # BJD = BKJD + 2454833
    _time_bjd = _time + bjdrefi
    time_bjd = time + bjdrefi
    _temp_time = Time(time_bjd, format="jd", scale="tdb")
    m, o, dva_x, dva_y = get_raDec2Pix(ra, dec, _temp_time)
    if get_psf:
        psftime_bjd = ptime + bjdrefi
        psf_time = Time(psftime_bjd, format="jd", scale="tdb")
        psfm, psfo, psfdva_x, psfdva_y = get_raDec2Pix(ra, dec, psf_time)
        _psfdva_x = fill_nan(psfdva_x, pseli, len(_time))
        _psfdva_y = fill_nan(psfdva_y, pseli, len(_time))

    seli = np.argwhere(sel).flatten()

    _dva_x = fill_nan(dva_x, seli, len(_time))
    _dva_y = fill_nan(dva_y, seli, len(_time))
    _m = fill_nan(m, seli, len(_time))
    _o = fill_nan(o, seli, len(_time))

    if not local_only:
        # convert to global ra dec
        ab_ra, ab_dec = np.empty(len(m)), np.empty(len(m))
        if get_psf:
            print(rawcsvpath)
            pab_ra, pab_dec = np.empty(len(psfm)), np.empty(len(psfm))
            for i in range(len(psfm)):
                pab_ra[i], pab_dec[i] = get_Pix2RaDec(
                    psfm[i], psfo[i], psf_x[i], psf_y[i], psf_time[i], aberrate=True
                )
        for i in range(len(m)):
            ab_ra[i], ab_dec[i] = get_Pix2RaDec(
                m[i], o[i], ctd_x[i], ctd_y[i], _temp_time[i], aberrate=True
            )
        _ra = fill_nan(ab_ra, seli, len(_time))
        _dec = fill_nan(ab_dec, seli, len(_time))
        if get_psf:
            _pra = fill_nan(pab_ra, pseli, len(_time))
            _pdec = fill_nan(pab_dec, pseli, len(_time))
        if verbose:
            txt = (
                f"{os.path.basename(lcpath)}: quarter {quarter}, "
                f"module {m[0]}, output {o[0]}, row {dva_x[0]}, column {dva_y[0]}"
            )
            print(txt, flush=1)

    # Correct it here via least squares.  Note the least-squares solver needs
    # non-NaN data.
    # A = np.vstack([dva_x, np.ones(len(dva_x))]).T
    # m, c = np.linalg.lstsq(A, ctd_x, rcond=None)[0]
    # # mdva_x still the dva model
    # mdva_x = m * dva_x + c
    # _mdva_x = m * _dva_x + c

    # A = np.vstack([dva_y, np.ones(len(dva_y))]).T
    # m, c = np.linalg.lstsq(A, ctd_y, rcond=None)[0]
    # mdva_y = m * dva_y + c
    # _mdva_y = m * _dva_y + c

    # no underscore: these are "selected" (non-zero quality flags)
    # df = pd.DataFrame({
    #     'time':    time,
    #     'ctd_x':   ctd_x,
    #     'ctd_y':   ctd_y,
    #     'dva_x':   dva_x,
    #     'dva_y':   dva_y,
    #     'mdva_x':  mdva_x,
    #     'mdva_y':  mdva_y,
    #     'module': m,
    #     'output': o,
    #     'ra': ab_ra,
    #     'dec': ab_dec,
    #     'cadenceno':  cadenceno,
    # })
    # df.to_csv(selcsvpath, index=False)
    # print(f'Cached to {selcsvpath}',flush=1)

    # underscore: these are "raw" (no quality flag cut)
    df = pd.DataFrame(
        {
            "time": _time,
            "qual": _qual,
            "mom_x": _ctd_x,
            "mom_y": _ctd_y,
            "dva_x": _dva_x,
            "dva_y": _dva_y,
            "module": _m,
            "output": _o,
            "cadenceno": _cadenceno,
        }
    )
    if not local_only:
        df["ra"] = _ra
        df["dec"] = _dec
        if get_psf:
            df["psf_ra"] = _pra
            df["psf_dec"] = _pdec
            df["psf_dva_x"] = _psfdva_x
            df["psf_dva_y"] = _psfdva_y
    if get_psf:
        df["psf_x"] = _psf_x
        df["psf_y"] = _psf_y
    if get_poscorr:
        df["poscorr_x"] = _poscorr_x
        df["poscorr_y"] = _poscorr_y

    df.to_csv(rawcsvpath, index=False)
    print(f"Cached to {rawcsvpath}", flush=1)

    return hdr, d, pd.read_csv(rawcsvpath)


def get_astrometric_precision(kepmag):
    """
    Interpolate the Monet+2010 astrometric precision plot.
    If kepmag includes stars with magnitudes >16 or <11.4, their values will be
    filled as NaN.

    <11.4 is saturation
        (Gilliland+10 https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.160G/abstract)
    >16 is beyond the Monet+2010 limits.

    Args:
        kepmag (np.ndarray): of Kepler magnitudes
    Returns:
        astrom_precision_mas (np.ndarray): astrometric precision, in
        milliarcsecsonds, in a 30-minute stack.
    """

    csvpath = os.path.join(
        DATADIR, "processed", "monet2010_astrometric_precision_lowerenvelope.csv"
    )
    df = pd.read_csv(csvpath)

    eps = np.random.uniform(low=-1e-8, high=1e-8, size=len(df))
    df["kic_mag"] = df["kic_mag"] + eps

    df = df.sort_values(by="kic_mag", ascending=True)

    x = nparr(df.kic_mag)
    y = nparr(df.astrometric_error) * 1e3

    z = polyfit(x, y, 3)
    fn = poly1d(z)

    astrom_precision_mas = fn(kepmag)

    sel = (kepmag > 16) | (kepmag < 11.4)
    astrom_precision_mas[sel] = np.nan

    return astrom_precision_mas


def interp_nans_mdva(cadenceno, mdva_, nstars):
    """
    given cadence_no, model dva prediction "mdva_" (either_x or _y), and
    number of stars, fill in the nans!

    args can be (Nstars,Ntimes) np.ndarrays or (Ntimes) np.ndarrays for a
    single star.
    """

    c_mdva_ = []

    if len(cadenceno.shape) == 2 and len(mdva_.shape) == 2:
        for ix in range(nstars):
            fn = interp1d(
                cadenceno[ix, :][~np.isnan(mdva_[ix, :])],
                mdva_[ix, :][~np.isnan(mdva_[ix, :])],
                kind="quadratic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            this_mdva_ = fn(cadenceno[ix, :])
            c_mdva_.append(this_mdva_)
        c_mdva_ = np.vstack(c_mdva_)

    elif len(cadenceno.shape) == 1 and len(mdva_.shape) == 1:
        fn = interp1d(
            cadenceno[~np.isnan(mdva_)],
            mdva_[~np.isnan(mdva_)],
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        c_mdva_ = fn(cadenceno)

    else:
        raise ValueError(
            "Expected arguments to be (Nstars,Ntimes) or " "(Ntimes) np.ndarrays"
        )

    return c_mdva_


def get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o(
    sampleid, seed, n_samples, fix_quarter=None, verbose=1, target_m=[], target_o=[]
):
    fn = os.path.join(RESULTSDIR, f"tables/{sampleid}.csv")
    if not os.path.exists(fn):
        print(fn)
        print("sampleid does not exist", flush=1)
        return
    df = pd.read_csv(fn)
    lcdir = os.path.join(DATADIR, "lightcurves", "Kepler")
    datestr = quarter_datestr_dict[fix_quarter]
    qrtr_mask = df[f"inQ{str(fix_quarter)}"] == 1
    cat_kicids = df["kicid"][qrtr_mask].to_numpy()
    nstars = len(cat_kicids)
    if verbose:
        print(
            f"There are {nstars} stars in the catalog that are in this quarter.",
            flush=1,
        )
    zfillkcids = [str(kc)[:-5].zfill(4) for kc in cat_kicids]
    zkcs_long = [str(kc).zfill(9) for kc in cat_kicids]
    lcpaths = [
        f"{lcdir}/{zkc}/{zkcl}/kplr{zkcl}-{datestr}_llc.fits"
        for (zkc, zkcl) in zip(zfillkcids, zkcs_long)
    ]

    if len(target_m) > 0 or len(target_o) > 0:
        for lcpath in lcpaths:
            _ = given_lcpath_get_time_ctd_dva(lcpath, verbose=0, process_raw=0)

        selcsvpath = [lcpath.replace(".fits", "_ctds.csv") for lcpath in lcpaths]
        selcsvpath = [x.replace("lightcurves/Kepler", "ctd") for x in selcsvpath]
        csvpaths = [x.replace("_ctds.csv", "_rawctds.csv") for x in selcsvpath]

        sel = np.ones(len(csvpaths), dtype=bool)
        if len(target_m) > 0:
            module = nparr([get_df_keyword(f, "module")[0] for f in csvpaths])
            sel = sel & np.isin(module, target_m)
        if len(target_o) > 0:
            output = nparr([get_df_keyword(f, "output")[0] for f in csvpaths])
            sel = sel & np.isin(output, target_o)
        lcpaths = list(np.array(lcpaths)[sel])

    if (len(lcpaths) < n_samples) or (n_samples == -1):
        if verbose:
            print(f"returning all {nstars} lightcurves", flush=1)
        return lcpaths
    if verbose:
        print(f"Got {len(lcpaths)} light curves to draw from", flush=1)
        print(f"...picking {n_samples}", flush=1)
    np.random.seed(seed)
    return np.random.choice(lcpaths, n_samples, replace=False)


def get_lcpaths_given_kicids_quarter(quarter, kicids):
    ## kicids should be a list or nparr
    lcdir = os.path.join(DATADIR, "lightcurves", "Kepler")
    datestr = quarter_datestr_dict[quarter]
    zfillkcids = [str(kc)[:-5].zfill(4) for kc in kicids]
    zkcs_long = [str(kc).zfill(9) for kc in kicids]
    lcpaths = [
        f"{lcdir}/{zkc}/{zkcl}/kplr{zkcl}-{datestr}_llc.fits"
        for (zkc, zkcl) in zip(zfillkcids, zkcs_long)
    ]
    return lcpaths


def get_pca_basisvecs(
    sampleid, dvastr, qstr, n_components, svdstr="", globstr="", addsuffix=""
):
    """This function returns the cached text files for eigenvectors."""

    pcadir = os.path.join(RESULTSDIR, "pca_eig", sampleid)
    pcapath_x = join(pcadir, f"{qstr[1:]}{dvastr}{svdstr}{globstr}{addsuffix}_x.txt")
    pcapath_y = join(pcadir, f"{qstr[1:]}{dvastr}{svdstr}{globstr}{addsuffix}_y.txt")

    if not os.path.exists(pcapath_x):
        print(f"{pcapath_x} not found, run Clean first")
        exit()
        print("Calculating PCA...", flush=1)
        run_pca(
            quarter=int(qstr[2:]),
            sampleid=sampleid,
            dvacorr=(dvastr != ""),
            useTrunSVD=(svdstr != ""),
        )
    eigvecs_x = np.genfromtxt(pcapath_x)
    eigvecs_y = np.genfromtxt(pcapath_y)
    if n_components == -1:
        return eigvecs_x, eigvecs_y
    return eigvecs_x[: n_components + 1, :], eigvecs_y[: n_components + 1, :]

    # dtrvecs_x = eigvecs_x[: n_components + 1, :]
    # basisvecs_x = deepcopy(dtrvecs_x)
    # dtrvecs_y = eigvecs_y[: n_components + 1, :]
    # basisvecs_y = deepcopy(dtrvecs_y)
    # csvpath = os.path.join(pcadir, f'cad_log_{qstr[1:]}{svdstr}.csv')
    # sel_df = pd.read_csv(csvpath)

    # return basisvecs_x, basisvecs_y


def read_dva_res(sampleid, quarter, conv2glob, addsuffix="", mag=1.0, sig_only=False):
    """This function generates fake local coordinates based on true global coordinates. Assumes FITS already exists."""
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    if not os.path.exists(fitsfn):
        raise FileNotFoundError

    # assumes we are using the radec computed iteratively from MOM, can be replaced by REAL_PSF
    tag = "REAL_MOM"
    with fits.open(fitsfn) as hdu:
        ra = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["dec"]
        module = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["module"][:, 0]
        output = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["output"][:, 0]

        star_ra = hdu["PATHS"].data["survey_ra"]
        star_dec = hdu["PATHS"].data["survey_dec"]

    if addsuffix == "":
        dx, dy = conv2cart(ra, dec)
    else:
        CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}")
        dra = np.load(f"{CACHEDIR}/{addsuffix}_dra.npy") * 180 / np.pi
        ddec = np.load(f"{CACHEDIR}/{addsuffix}_ddec.npy") * 180 / np.pi
        if sig_only:
            inj_ra = star_ra + dra * mag
            inj_dec = star_dec + ddec * mag
        else:
            inj_ra = ra + dra * mag
            inj_dec = dec + ddec * mag
        dx, dy = conv2cart(inj_ra, inj_dec)  # in mas

    if not conv2glob:
        output_axis_map = np.load(
            f"{RESULTSDIR}/temporary/output_axis.npy", allow_pickle=True
        ).item()
        dx, dy = get_fake_local(dx, dy, module, output, output_axis_map)
    return dx, dy


def comp_pca_basisvecs(
    sampleid, quarter, n_components, conv2glob, sig_only=False, addsuffix="", mag=1.0
):
    """This function performs PCA live. The mean vector is stored as the first row."""
    dx, dy = read_dva_res(
        sampleid, quarter, conv2glob, addsuffix, mag, sig_only=sig_only
    )
    dx, dy = centering(dx), centering(dy)  # remove time-wise mean
    # is it legitimate to do this? what is the mean here?
    mean_x, eigenvecs_x = PCA_live(dx)  # in mas
    mean_y, eigenvecs_y = PCA_live(dy)  # in mas
    if n_components == -1:
        return np.vstack([mean_x, eigenvecs_x]), np.vstack([mean_y, eigenvecs_y])
    return np.vstack([mean_x, eigenvecs_x[:n_components]]), np.vstack(
        [mean_y, eigenvecs_y[:n_components]]
    )
