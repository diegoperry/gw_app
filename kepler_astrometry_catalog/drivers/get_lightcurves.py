import kepler_astrometry_catalog.get_data as gd
import os

# idstring = 'all'
i = int(os.getenv('SLURM_ARRAY_TASK_ID'))
idstring = f'brightestnonsat100_rot_mo{i}'
ibatch = 0
quarter = 12

# print(f'Starting download for quarter {quarter}')
gd.download_lightcurves(IDSTRING=idstring, qrt=quarter, ibatch=ibatch)
gd.reload_corrupt_FITS(idstring, quarter, ibatch)
