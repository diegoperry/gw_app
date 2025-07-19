"""
run_pca
get_raDec2Pix

_get_data_keyword
"""
#############
## LOGGING ##
#############

import logging

log_sub = '{'
log_fmt = '[{levelname:1.1} {asctime} {module}:{lineno}] {message}'
log_date_fmt = '%y%m%d %H:%M:%S'

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

#
# standard imports
#
import os
from glob import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from copy import deepcopy

from numpy import array as nparr

from sklearn.decomposition import PCA, TruncatedSVD #FactorAnalysis
from scipy.sparse import csr_matrix
# from sklearn.linear_model import LinearRegression, BayesianRidge, RidgeCV
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import MinMaxScaler
#
# non-standard imports
#
from aesthetic.plot import set_style, savefig
from kepler_astrometry_catalog.paths import LOCALDIR, RESULTSDIR, CTDCACHEDIR, DATADIR
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict


from raDec2Pix import raDec2Pix
rdp = raDec2Pix.raDec2PixClass()
from astropy.io import fits

def get_raDec2Pix(ra, dec, time):
    """
    ra/dec: degrees
    time: Astropy.Time object
    """
    m, o, r, c = rdp.ra_dec_2_pix(ra, dec, time.mjd)
    return m, o, r, c

def get_Pix2RaDec(module, output, row, column, time, aberrate):
    ra, dec = rdp.pix_2_ra_dec(module, output, row, column, time.mjd, aberrateFlag = aberrate)
    return ra, dec


def get_df_keyword(csvpath, key):
    # FIXME: should move to helpers...but then need to deal with circular imports.
    df = pd.read_csv(csvpath)
    return nparr(df[key])

def _get_data_keyword(fits_file,
                      keyword,
                      ext=1):
    '''
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

    '''

    hdulist = fits.open(fits_file)

    if keyword in hdulist[ext].data.names:
        val = hdulist[ext].data[keyword]
    else:
        val = None

    hdulist.close()
    return val

def run_pca(quarter=None, sampleid='brightestnonsat100', max_nstars=100, dvacorr=False, useTrunSVD=False):

    from kepler_astrometry_catalog.helpers import (
        get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o,
        given_lcpath_get_time_ctd_dva
    )
    from kepler_astrometry_catalog.paths import CTDCACHEDIR

    qstr = str(quarter).zfill(2)
    svdstr = '' if not useTrunSVD else '_TrunSVD'
    dvastr = '' if not dvacorr else '_dva'

    pcadir = os.path.join(RESULTSDIR,'pca_eig',sampleid)
    if not os.path.exists(pcadir):
        os.makedirs(pcadir)

    eigvecpath_x = os.path.join(pcadir,f'Q{qstr}{dvastr}{svdstr}_x.txt')
    eigvecpath_y = os.path.join(pcadir,f'Q{qstr}{dvastr}{svdstr}_y.txt')
    if os.path.isfile(eigvecpath_x) and os.path.isfile(eigvecpath_y):
        print('Found '+os.path.basename(eigvecpath_x))
        return

    lcdir = os.path.join(DATADIR, 'lightcurves', 'Kepler')
    print(lcdir)
    if not os.path.exists(lcdir): NotImplementedError

    datestr = quarter_datestr_dict[quarter]
    lcglob = os.path.join(lcdir, '*', '*', f'*{datestr}*_llc.fits')
    lcpaths = glob(lcglob)
    if len(lcpaths) > max_nstars:
        lcpaths = np.random.choice(lcpaths, max_nstars, replace=False)
    nstars = len(lcpaths)

    print('number of stars = ', str(nstars))

    for ix, lcpath in enumerate(lcpaths):
        if np.mod(ix+1, 10)==0:
            print(f'finished getting {ix+1}/{nstars} centroid data')
        _ = given_lcpath_get_time_ctd_dva(lcpath, verbose=0)

    print('finished getting centroid data on stars.')

    _ctd_x = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('MOM_CENTR2', nstars), np.repeat(1, nstars)
    )))
    # row
    _ctd_y = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('MOM_CENTR1', nstars), np.repeat(1, nstars)
    )))
    # variation of times of different stars is roughly 0.0003, negligible
    time = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('TIME', nstars), np.repeat(1, nstars)
    )))
    qual = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('SAP_QUALITY', nstars), np.repeat(1, nstars)
    )))

    csvnames = [os.path.basename(lcpath).replace('.fits','_rawctds.csv')
            for lcpath in lcpaths]
    csvpaths = [os.path.join(CTDCACHEDIR, csvname)
                for csvname in csvnames]

    mdva_x = nparr(list(map(
        get_df_keyword,
        csvpaths, np.repeat('mdva_x', nstars)
    )))
    mdva_y = nparr(list(map(
        get_df_keyword,
        csvpaths, np.repeat('mdva_y', nstars)
    )))

    ctd_x = _ctd_x - dvacorr*mdva_x
    ctd_y = _ctd_y - dvacorr*mdva_y
    qualsel = (qual == 0)

    if not useTrunSVD:
        # _sel = ((~np.isnan(ctd_x).any(axis=0)) & (~np.isnan(ctd_y).any(axis=0)))
        _sel = ((qualsel.any(axis=0)) & (~np.isnan(ctd_x).any(axis=0)) & (~np.isnan(ctd_y).any(axis=0)))
    else:
        _sel = ( (~np.isnan(ctd_x).all(axis=0)) & (~np.isnan(ctd_y).all(axis=0)))

    qctd_x = ctd_x[:,_sel]
    qctd_y = ctd_y[:,_sel]
    qtime = time[:,_sel]

    print('data shape', qctd_x.shape)
    print('finished loading data, start PCA...')
     
    mean_ctd_x = np.nanmean(qctd_x, axis=1)
    X_x = qctd_x - mean_ctd_x[:, None]
    if useTrunSVD:
        _ind = tuple(np.argwhere(np.isnan(qctd_x)).T)
        X_x[_ind] = 0
        X_x = csr_matrix(X_x)
        pca_x = TruncatedSVD(n_components=X_x.shape[0])
    else:
        pca_x = PCA()
    pca_x.fit(X_x)

    mean_ctd_y = np.nanmean(qctd_y, axis=1)
    X_y = qctd_y - mean_ctd_y[:, None]
    if useTrunSVD:
        _ind = tuple(np.argwhere(np.isnan(qctd_y)).T)
        X_y[_ind] = 0
        X_y = csr_matrix(X_y)
        pca_y = TruncatedSVD(n_components=X_y.shape[0])
    else:
        pca_y = PCA()
    pca_y.fit(X_y)

    eigenvecs_x = pca_x.components_
    eigenvecs_y = pca_y.components_

    outdf = pd.DataFrame({'time': time[0,:],'selected': _sel})
    outpath = os.path.join(pcadir, f'cad_log_Q{qstr}{svdstr}.csv')
    outdf.to_csv(outpath, index=False)

    np.savetxt(eigvecpath_x, eigenvecs_x)
    np.savetxt(eigvecpath_y, eigenvecs_y)

    return

def calc_pca_residual(quarter=None, sampleid='brightestnonsat100',dvacorr=True,useTrunSVD=False,n_components=3):

    from kepler_astrometry_catalog.helpers import (
    get_pca_basisvecs
)
    
    qstr = '' if quarter is None else f'_Q{str(quarter).zfill(2)}'
    dvastr = '' if not dvacorr else '_dva'
    svdstr = '' if not useTrunSVD else '_TrunSVD'

    lcdir = os.path.join(DATADIR, 'lightcurves', 'Kepler')
    datestr = quarter_datestr_dict[quarter]
    lcglob = os.path.join(lcdir, '*', '*', f'*{datestr}*_llc.fits')
    lcpaths = glob(lcglob)
    nstars = len(lcpaths)

    # get pca eigenvectors
    basisvecs_x, basisvecs_y, sel_df = get_pca_basisvecs(sampleid, dvastr, qstr, n_components, svdstr)
    sel = np.array(sel_df['selected'])

    # get centroids
    csvnames = [os.path.basename(lcpath).replace('.fits','_rawctds.csv')
            for lcpath in lcpaths]
    csvpaths = [os.path.join(CTDCACHEDIR, csvname)
                for csvname in csvnames]

    mdva_x = nparr(list(map(
        get_df_keyword,
        csvpaths, np.repeat('mdva_x', nstars)
    )))
    mdva_y = nparr(list(map(
        get_df_keyword,
        csvpaths, np.repeat('mdva_y', nstars)
    )))
    _ctd_x = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('MOM_CENTR2', nstars), np.repeat(1, nstars)
    )))
    _ctd_y = nparr(list(map(
        _get_data_keyword,
        lcpaths, np.repeat('MOM_CENTR1', nstars), np.repeat(1, nstars)
    )))

    ctd_x = (_ctd_x - dvacorr*mdva_x)[:,sel]
    ctd_y = (_ctd_y - dvacorr*mdva_y)[:,sel]
    norm_fn = lambda x: x-np.nanmean(x, axis=1)[:,np.newaxis]
    norm_ctd_x = norm_fn(ctd_x)
    norm_ctd_y = norm_fn(ctd_y)
    
    # get centroid residuals
    coef_x = np.einsum('ij,kj', np.nan_to_num(basisvecs_x), np.nan_to_num(norm_ctd_x))
    coef_y = np.einsum('ij,kj', np.nan_to_num(basisvecs_y), np.nan_to_num(norm_ctd_y))
    model_x = np.einsum('ik,ij->kj', coef_x, np.nan_to_num(basisvecs_x))
    model_y = np.einsum('ik,ij->kj', coef_y, np.nan_to_num(basisvecs_y))
    
    print(model_y.shape)
    res_x = np.nan_to_num(norm_ctd_x - model_x)
    res_y = np.nan_to_num(norm_ctd_y - model_y)
    res_quad = np.sqrt(res_x**2+res_y**2).flatten()*3.98e3

    # compute median and percentile
    med = np.median(res_quad)
    med_upper = np.nanpercentile(res_quad, 84) - med
    med_lower = med - np.nanpercentile(res_quad, 16)
    print(f'{sampleid}{qstr}{dvastr}{svdstr}, n_component={n_components} residual:')
    print('median deviation = '+'{:.3f}'.format(med)+ ' mas')
    print('+'+'{:.3f}'.format(med_upper)+' mas (to 84 percentile)')
    print('-'+'{:.3f}'.format(med_lower)+' mas (to 16 percentile)\n')
    
    # print('std = '+'{:.3f}'.format(np.mean(res_quad))+ ' mas')

    # var_bwt_stars = np.std(np.mean(res_quad, axis=1))
    # print('std variation = '+'{:.3f}'.format(var_bwt_stars)+ ' mas')

    # print('median = '+'{:.3f}'.format(np.median(res_quad))+ ' mas')

    # var_bwt_stars = np.std(np.median(res_quad, axis=1))
    # print('median variation = '+'{:.3f}'.format(var_bwt_stars)+ ' mas')
    
    return