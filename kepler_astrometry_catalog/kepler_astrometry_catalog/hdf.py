'''
Table definitions and helper functions for the hdf5 object used in the new cleaning pipeline for input/output.
'''
import os
import pathlib 

import numpy as np
import tables

from kepler_astrometry_catalog.paths import DATADIR

class HdfStarsTable(tables.IsDescription):
    '''
    Table definition for the stars table in the hdf5 file.
    Auxillary info about stars. All columns are length (M total stars, which isn't necessarily equal to the N stars in each quarter).
    '''
    kicid = tables.UInt32Col()   # Integer KIC ID.
    ra = tables.Float64Col()     # Nominal RA from KIC, in deg. J2000 coordinate system.
    dec = tables.Float64Col()    # Nominal declination from KIC, in deg. J2000 coordinate system.
    kepmag = tables.Float64Col() # Kepler-band magnitude. ‘KEPMAG’ in the KIC: see column 15 in https://lweb.cfa.harvard.edu/kepler/kic/format/format.html

'''
Table generation functions for the hdf5 object. Since the columns have a dynamic shape (T = max # observations in a quarter)
in addition to the number of rows (N = number of stars), we can't define a single template class, but need factory
functions.
'''
def hdf_pos_tabledef(T_obs):
    return {
        'kicid': tables.UInt32Col(),                  # Integer KIC ID.
        'time': tables.Float64Col(shape=(T_obs,)),    # Time since first observation (globally, not from start of quarter) in seconds.
        'x': tables.Float64Col(shape=(T_obs,)),       # Centroid position in pixels.
        'y': tables.Float64Col(shape=(T_obs,)),       # Centroid position in pixels.
        'module': tables.UInt8Col(shape=(T_obs,)),    # Integer module index.
        'output': tables.UInt8Col(shape=(T_obs,)),    # Integer output/quadrant index with module.
        'quality': tables.UInt32Col(shape=(T_obs,)),  # Flag, which if nonzero, indicates the observation has some quality issue(s). See https://lightkurve.github.io/lightkurve/tutorials/2-creating-light-curves/2-2-kepler-noise-1-data-gaps-and-quality-flags.html for a reference on the values.
    }
    
def hdf_errors_tabledef(T_obs):
    return {
        'kicid': tables.UInt32Col(),                  # Integer KIC ID.
        'cov_xx': tables.Float64Col(shape=(T_obs,)),  # Per-observation estimated variance (sigma^2) on centroid, in x-direction. In units of pixel^2.
        'cov_yy': tables.Float64Col(shape=(T_obs,)),  # Per-observation estimated variance (sigma^2) on centroid, in y-direction. In units of pixel^2.
        'cov_xy': tables.Float64Col(shape=(T_obs,)),  # Per-observation estimated covariance between x/y axes. In units of pixel^2.
    }

'''
Helper functions
'''

def create_template_hdf(full_hdf_path, expectedrows, quarters=range(18), **kwargs):
    '''Create a template pytables object, with the structure for everything except the 'pos' and 'errors' tables for each quarters.

    Args:
        full_hdf_path (str): Full path to the hdf5 file to create.
        expectedrows (int): Expected number of rows for the 'pos' table. It is extremely important to set this to ~a good guess for the total # of observations x stars for each quarter,
                            otherwise PyTables writes will become extremely slow.
        quarters (list, optional): List of integer quarters to create the object for. Defaults to [0...17] (inclusive).
        **kwargs: Additional keyword arguments to pass to the PyTables creation function `tables.open_file`.
    '''
    quarters = list(quarters)
    assert all(isinstance(q, int) for q in quarters), "All quarters must be integers."
    hdffile = tables.open_file(full_hdf_path, mode='w', 
                               expected_rows_earray=expectedrows, expected_rows_table=expectedrows,
                               #filters=tables.Filters(complevel=9, complib='blosc2:lz4', shuffle=True, bitshuffle=False),
                               **kwargs)
    for q in quarters:
        quarter_group = hdffile.create_group('/', f'Q{q}', f'Data for quarter {q}')
        # 'pos' and 'errors' tables need to be created later, since they depend on the number of observations in each quarter.
        
    hdffile.create_table('/', 'stars', HdfStarsTable,
                         title='Auxillary info about stars. All columns are length '
                               +'(M total stars, which isn\'t necessarily equal to the N stars in each quarter).')
    aux_group = hdffile.create_group('/', 'cleaning', 'Auxillary info for the cleaning pipeline outputs.')
    
    hdffile.create_earray(aux_group, 'order', tables.StringAtom(itemsize=200), shape=(0,), 
                          title='Array of strings, each of which is a key for the groups below. ' + 
                                'Encodes the order of cleaning steps; for each step, you should append the step\'s name/whatever key you use below to the end of this array.')
    
    hdffile.flush()
    return hdffile