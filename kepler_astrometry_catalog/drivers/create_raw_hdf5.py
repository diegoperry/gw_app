'''
Create HDF5 file according to spec in README.md, for the raw momentum centroids of all stars/all quarters.
'''
import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")

import argparse
import time
from pathlib import Path
import warnings
import multiprocessing

import pandas as pd
import numpy as np
import numpy.ma
import tables
import lightkurve
import astropy.time
import tqdm

import kepler_astrometry_catalog.hdf
from kepler_astrometry_catalog.paths import *
import kepler_astrometry_catalog.helpers

def main(output_hdf_path, centroid_base_path, quarter_list, kicids_per_quarter, progress_bar=True):
    '''Write the raw centroid data to an HDF5 file, given a list of quarters and KIC IDs for each quarter.
    This function assumes the HDF5 file already exists, and will append new data to it.
    This inherently assumes that only a single process is writing to the HDF5 file, total; file locking erases any gains from concurrency.

    Args:
        output_hdf_path (str): Output HDF5 file path where the centroid data will be stored.
        centroid_base_path (str): Base path for the centroid CSV files, typically something like 'ctd' (relative to DATADIR).
        quarter_list (list): List of quarters to process. Each quarter should be an integer.
        kicids_per_quarter (list): List of lists, where each sublist contains KIC IDs for the corresponding quarter in `quarter_list`.
        progress_bar (bool, optional): Whether to show a progress bar for processing each quarter. Defaults to False.
    '''
    assert len(quarter_list) == len(kicids_per_quarter), "quarter_list and kicids_per_quarter must have the same length."
    
    output_hdf_path = Path(output_hdf_path)
    
    for quarter, kicid_list in zip(quarter_list, kicids_per_quarter):
        if not isinstance(quarter, int):
            raise ValueError(f"Quarter {quarter} is not an integer.")
        lc_path_list = kepler_astrometry_catalog.helpers.get_lcpaths_given_kicids_quarter(quarter, kicid_list)
        
        if progress_bar:
            iterator = tqdm.tqdm(zip(lc_path_list, kicid_list), desc=f"Processing quarter {quarter}", total=len(kicid_list), mininterval=60, maxinterval=60, position=quarter, leave=True)
        else:
            iterator = zip(lc_path_list, kicid_list)
        with tables.open_file(output_hdf_path, mode='r+') as hdf_file:
            quarter_group = hdf_file.get_node(f"/Q{quarter}")
            kicid_arr = quarter_group.kicid
                
            for i, (lc_path, kicid) in enumerate(iterator):
                if not os.path.exists(lc_path):
                    raise FileNotFoundError(f"Light curve file for KICID {kicid} at quarter {quarter} does not exist.")
                csv_path = lc_path.replace(".fits", "_rawctds.csv")
                rawcsv_path = csv_path.replace("lightcurves/Kepler", centroid_base_path)
                try:
                    centroid_df = pd.read_csv(rawcsv_path)
                except FileNotFoundError:
                    print(f"Centroid CSV file {rawcsv_path} not found for KICID {kicid}. Skipping this entry.")
                    continue
                # Astropy time doesn't play nice with NaN values, so we need to mask first, compute deltas, then fill.
                lc_time = astropy.time.Time(numpy.ma.masked_invalid(centroid_df['time']), scale='tdb', format='bkjd')
                lc_time_delta = lc_time - astropy.time.Time([0], scale='tdb', format='bkjd')
                lc_time_delta_sec = numpy.ma.filled(lc_time_delta.sec, fill_value=np.nan)
                assert len(lc_time_delta_sec) == len(centroid_df)
                # print('{} NaNs in time, {} NaNs in x/y'.format(
                #     np.sum(np.isnan(lc_time_delta_sec)),
                #     np.sum(np.isnan(centroid_df['mom_x']))))
                # # Check that NaNs between time and x/y are aligned.
                # assert np.all(np.isnan(centroid_df['mom_x']) == np.isnan(centroid_df['mom_y'])), "NaN values in centroid DataFrame are not aligned between x and y columns."
                # assert np.all(np.isnan(lc_time_delta_sec) == np.isnan(centroid_df['mom_x'])), "NaN values in centroid DataFrame position columns are not aligned with time column."
                
                #assert kicid not in kicid_arr.read(), f"KICID {kicid} already exists in quarter {quarter}."
                kicid_arr.append(np.array([kicid]))#.astype(kicid_arr.dtype, casting='unsafe'))
                
                # This check may have performance implications if done every loop iteration.
                if i == 0:
                    if not hasattr(quarter_group, 'pos'):
                        pos_table = hdf_file.create_table(quarter_group, 'pos', kepler_astrometry_catalog.hdf.hdf_pos_tabledef(len(lc_time_delta)), "Centroid positions")
                    else:
                        pos_table = quarter_group.pos
                
                # A note for the future: uncommenting this line caused pytables to slow down unbearably as the table length increased.
                # assert len(lc_time_delta) == pos_table.col('time').shape[-1], f"Time length mismatch between centroid DataFrame and HDF5 table. Table with last KICID {kicid_arr[-2]} has {pos_table['time'].shape[-1]} entries, but DataFrame for KICID {kicid} has {len(lc_time_delta)} entries."
                
                with warnings.catch_warnings():
                    # pandas has an invalid cast warning, which doesn't appear to actually be true?
                    warnings.simplefilter("ignore")
                    
                    new_row = pos_table.row
                    new_row['time'] = lc_time_delta_sec
                    new_row['x'] = centroid_df['mom_x']#.astype(pos_table.description.x.dtype, casting='safe')
                    new_row['y'] = centroid_df['mom_y']#.astype(pos_table.description.y.dtype, casting='safe')
                    new_row['module'] = centroid_df['module']#.astype(pos_table.description.module.dtype, casting='safe')
                    new_row['output'] = centroid_df['output']#.astype(pos_table.description.output.dtype, casting='safe')
                    new_row['quality'] = centroid_df['qual']#.astype(pos_table.description.quality.dtype, casting='safe')
                    new_row.append()
                    
            hdf_file.flush()
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create raw HDF5 file for Kepler astrometry catalog centroids. The HDF5 file must not already exist; no overwriting support is provided for concurrency reasons.")
    parser.add_argument("sampleid", type=str, help="Name of sample CSV catalog to use in TABLEDIR for star KICIDs.")
    parser.add_argument("output_hdf_path", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("--centroid_base_path", type=str, help="Base path for centroid CSV files. Defaults to ctd-all-momonly/ .", default="ctd-all-momonly")
    parser.add_argument("--quarters", type=int, nargs='+', default=list(range(18)), help="List of quarters to process (default: all quarters 0-17).")

    args = parser.parse_args()
    
    if set(args.quarters) != set(range(18)):
        print(f"Warning: You are processing quarters {args.quarters}, which is not the full set of quarters 0-17. The `stars` table will still include stars from all quarters, but the `pos` table may not.")
        
    # Use all available threads for BLOSC/BLOSC2 compression.
    tables.set_blosc_max_threads(multiprocessing.cpu_count())
    tables.set_blosc2_max_threads(multiprocessing.cpu_count())
    
    kic_cat = pd.read_csv(os.path.join(TABLEDIR, f"{args.sampleid}.csv"))
    kicids_per_quarter = []
    for quarter in args.quarters:
        if not isinstance(quarter, int):
            raise ValueError(f"Quarter {quarter} is not an integer.")
        qrtr_mask = kic_cat[f"inQ{str(quarter)}"] == 1
        cat_kicids = kic_cat["kicid"][qrtr_mask].to_numpy()
        kicids_per_quarter.append(cat_kicids)
    
    assert not os.path.exists(args.output_hdf_path), f"HDF5 file {args.output_hdf_path} already exists. Please remove it before running this script."
    output_hdf = kepler_astrometry_catalog.hdf.create_template_hdf(args.output_hdf_path, len(kic_cat), quarters=args.quarters)
    # Populate the stars table.
    stars_table = output_hdf.root.stars
    
    for cat_row in kic_cat.itertuples():
        new_row = stars_table.row
        new_row['kicid'] = cat_row.kicid
        new_row['ra'] = cat_row.kic_degree_ra
        new_row['dec'] = cat_row.kic_dec
        new_row['kepmag'] = cat_row.kic_kepmag
        new_row.append()
        
    output_hdf.flush()
    output_hdf.close()
    
    main(args.output_hdf_path, args.centroid_base_path, args.quarters, kicids_per_quarter)