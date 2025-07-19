import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import fits
import pandas as pd
import datetime
from astropy.time import Time
from scipy.interpolate import interp1d
from kepler_astrometry_catalog.paths import DATADIR
from astropy.timeseries import LombScargle as ls

ANC_PATH = os.path.join(DATADIR, "ancillary/anc-eng/parameter-bundled/")
outpath = os.path.join(DATADIR, "ancillary/anc_pca_ready.csv")

files = [
    "kplr_anc-eng_ReactionWheelSpeeds.csv.gz",
    # "kplr_anc-eng_OpticsTemperatures1of2.csv.gz",
]
files_param_dict={files[0]: [" ADRW1SPD_", " ADRW2SPD_", " ADRW3SPD_", " ADRW4SPD_"]}
files_param_list = [" ADRW1SPD_", " ADRW2SPD_", " ADRW3SPD_", " ADRW4SPD_"]
    # files[1]: [" PEDPMAT1", " PEDPMAT2", " PEDPMAT3", " PEDPMAT4"],


## get Q12 LC to get correct times. picked random LC.
# lc_fn = os.path.join(
#     DATADIR, "lightcurves/Kepler/0008/000893233/kplr000893233-2012088054726_llc.fits"
# )
# load from catalog 
from kepler_astrometry_catalog.clean import Clean
sampleid = 'brightestnonsat100_rot'
cl = Clean(sampleid=sampleid,verbose=1,conv2glob=0, dvacorr=1,pca=0,save_pca_eigvec=0,remove_earth_points=1,max_nstars=-1,write_cleaned_data=0) 
hdul = fits.open(np.random.choice(cl.lcpaths))
lc = hdul[1].data

def get_mjd_time(time):
    newtime = np.zeros(len(time))
    for i, t in enumerate(time):
        if np.isnan(t):
            newtime[i] = np.nan
        else:
            newtime[i] = Time(t + 2454833, format="jd").mjd
    return newtime


lc_mjd = get_mjd_time(lc["TIME"])
splines_df = pd.DataFrame()
splines_df["TIME"] = lc["TIME"]


def get_splines(fn, params):
    df = pd.read_csv(os.path.join(ANC_PATH, fn), header=9)
    df.rename(columns={"# MJD": "MJD"}, inplace=True)

    for param in params:
        spline = interp1d(df["MJD"], df[param])
        splines_df[param.lstrip()] = spline(lc_mjd)

    return splines_df

def anc_ls():
    sdf=get_splines(files[0],files_param_dict[files[0]])
    for i in range(len(files_param_list)):
        t=lc_mjd[~np.isnan(lc_mjd)]

        y=sdf.iloc[:,i+1]
        y=y.dropna()

        frequency, power = ls(t*86400,y).autopower()
        a=plt.figure()
        plt.plot(frequency,power)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency(HZ)')
        plt.ylabel('Power')
        plt.title(f"{files_param_list[i]} All")
        plt.grid()
        a.savefig("/home/howard/kepler/kepler_astrometry_catalog/data/maestro/" +str(i)+"GG.pdf")









# sdf = get_splines(files[0], files_param_dict[files[0]])
# sdf_tot = get_splines(files[1], files_param_dict[files[1]])


# sdf_tot.to_csv(outpath, index=False)