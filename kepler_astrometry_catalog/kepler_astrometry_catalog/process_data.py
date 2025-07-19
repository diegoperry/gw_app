'''process downloaded data and make tables

'''
import os
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from kepler_astrometry_catalog.paths import TABLEDIR, DATADIR,LOCALDIR
import urllib.request
from urllib.parse import urlparse

def get_kepler_star():
    # create stars_observed_by_kepler.csv, 9.8MB with quarters info
    print('Getting stars observed by kepler')
    outpath = os.path.join(TABLEDIR, 'stars_observed_by_kepler.csv')
    if os.path.isfile(outpath):
        print(os.path.basename(outpath)+' already exists.\n')
        return

    getterscripts = sorted(
        glob(os.path.join(LOCALDIR,'lightcurve_getters/kepler_lightcurves_Q*_long.sh'))
    )
    collectordict = {}
    all_kicids = []
    # There are 18 quarters, Q00 through Q17.  Get all the KIC IDs for stars that
    # were observed in each quarter.  NB: these changed a little between sectors.
    # So we need to be a bit more careful:
    for quarter_num, g in enumerate(getterscripts):
        with open(g, 'r') as f:
            lines = f.readlines()
        kicids = nparr([
            np.int64(l.split(' ')[-2].split('/')[3].lstrip('0'))
            for l in lines[1:]
        ])
        collectordict[quarter_num] = kicids
        all_kicids.append(kicids)

    # flatten the list of arrays; find unique KICIDs.
    u_kicids = np.unique(np.concatenate(all_kicids).ravel())

    df = pd.DataFrame({'kicid': u_kicids})
    quarter_cols = []
    for quarter_num, _ in enumerate(getterscripts):
        k = f'inQ{quarter_num}'
        df[k] = np.in1d(
            u_kicids, collectordict[quarter_num]
        ).astype(int)
        quarter_cols.append(k)

    df['total_quarters'] = df[quarter_cols].sum(axis=1)
    df.to_csv(outpath, index=False)
    print(f'Wrote {outpath}'+'\n')
    return

def get_xkic():
    # add v10 info to make stars_observed_by_kepler_X_KIC.csv, 27.7MB: very slow!
    print('Processing kepler v10 data...')
    outpath = os.path.join(TABLEDIR, 'stars_observed_by_kepler_X_KIC.csv')
    if os.path.isfile(outpath):
        print(os.path.basename(outpath)+' already exists.\n')
        return

    csvpath = os.path.join(TABLEDIR, 'stars_observed_by_kepler.csv')
    assert os.path.isfile(csvpath)
    # obs_df N entries: 207,617
    obs_df = pd.read_csv(csvpath)

    # kic_df N entries: 13,161,029
    # kic_df = pd.read_csv(os.path.join(LOCALDIR, "kepler_kic_v10.csv"))
    # see https://archive.stsci.edu/kepler/data_search/help/quickcol.content
    # col names see: https://archive.stsci.edu/search_fields.php?mission=kepler
    selcols = 'kic_degree_ra,kic_dec,kic_pmra,kic_pmdec,kic_kepmag,kic_kepler_id,kic_2mass_id,kic_pq'.split(',')
    
    kic_df = pd.read_csv(os.path.join(LOCALDIR, "kepler_kic_v10.csv"),usecols=selcols)
    kic_df = kic_df[selcols]

    # N(obs_df not in v10) = 1506
    mdf = obs_df.merge(
        kic_df, how='left', left_on='kicid', right_on='kic_kepler_id'
    )
    # mdf['kic_kepler_id'] = mdf['kic_kepler_id'].astype(int)
    mdf.to_csv(outpath, index=False)
    print(f'Wrote {outpath}'+'\n')
    return

def get_rperiod():
    # create rot_period_cat.csv, 2.1MB
    print('Processing rotational period data...')
    outpath = os.path.join(TABLEDIR, 'rot_period_cat.csv')
    if os.path.isfile(outpath):
        print('file exists.\n')
    cat_fn = {'MQ':['apjs492452t1_mrt.txt',10,11,24,32],
             'S_MK':['apjsab3b56t3_mrt.txt',10,22,24,54],
              'S_GF':['apjsac033ft1_mrt.txt',9,21,25,54]}
    full_df = pd.DataFrame({'kicids':[0]})

    for cat, fn in cat_fn.items():

        cat_name, hd_skip, hd_row, i_init, skiprow = fn
        txtpath = os.path.join(DATADIR, 'stellar_rotation', cat_name)

        hd_df = pd.read_table(txtpath,skiprows=range(hd_skip),header=None, nrows=hd_row)
        header = [cat+'_'+hd_df.iloc[i,0][i_init:i_init+8].strip() for i in range(hd_row)]

        header[0] = 'kicids'
        if cat == 'MQ':
            header[4] = cat+'_Prot'
            header[5] = cat+'_E_Prot'

        df = pd.read_table(txtpath,names=header, skiprows=range(skiprow),sep='\s+',engine='python')
        df['in_'+cat] = 1
        full_df = pd.merge(full_df,df[['kicids',cat+'_Prot',cat+'_E_Prot','in_'+cat]],how='outer',on='kicids')

    full_df.drop(index=full_df.index[0], axis=0, inplace=True)
    full_df.to_csv(outpath, index=False)
    return

if __name__ == "__main__":
   get_kepler_star()
   get_xkic()
   get_rperiod()
