import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import Table
import pandas as pd
from astropy.io import fits

from kepler_astrometry_catalog.paths import *
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict

fix_quarter = 12

## OK so let's start by pulling the catalog for all Q12 stars.
# sampleid = "all_rot"
sampleid = 'brightestnonsat100_rot'
fn = os.path.join(RESULTSDIR, f"tables/{sampleid}.csv")
_df = pd.read_csv(fn)

df = _df[_df["inQ12"] == 1]

lcdir = os.path.join(DATADIR, "lightcurves", "Kepler")
datestr = quarter_datestr_dict[fix_quarter]
qrtr_mask = df[f"inQ{str(fix_quarter)}"] == 1
cat_kicids = df["kicid"][qrtr_mask].to_numpy()
nstars = len(cat_kicids)
print(f"There are {nstars} stars in the catalog that are in this quarter.", flush=1)
zfillkcids = [str(kc)[:-5].zfill(4) for kc in cat_kicids]
zkcs_long = [str(kc).zfill(9) for kc in cat_kicids]
lcpaths = [
    f"{lcdir}/{zkc}/{zkcl}/kplr{zkcl}-{datestr}_llc.fits"
    for (zkc, zkcl) in zip(zfillkcids, zkcs_long)
]


def check_ppa(lcpaths):
    kicid = []
    for lcpath in lcpaths:
        hdulist = fits.open(lcpath)
        d = hdulist[1].data
        if np.any(~np.isnan(d["PSF_CENTR1"][0])):
            keplerid = hdulist[0].header["KEPLERID"]
            kicid.append(keplerid)
        hdulist.close()
    return kicid


klist = check_ppa(lcpaths)
print(f"Found {len(klist)} PPA stars.")

kdf = pd.DataFrame({"kicid": klist})
sdf = df.merge(kdf, on="kicid", how="right")

fn = os.path.join(TABLEDIR, f"ppa_catalog_q{str(int(fix_quarter))}.csv")
sdf.to_csv(fn, index=False)
print(f"Saved catalog to {fn}")
