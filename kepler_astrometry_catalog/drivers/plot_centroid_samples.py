import os
import kepler_astrometry_catalog.plotting as gwp
from kepler_astrometry_catalog.paths import RESULTSDIR
from kepler_astrometry_catalog.plotting import plot_chip_sky_frame

PLOTDIR = os.path.join(RESULTSDIR, 'raw_detector')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

seed = 1
pf = 1  # plot fit 
quarter = 12

# for i_samp in range(1,5):
    
#     sampleid = f'brightestnonsat100_rot_mo{i_samp}'

#     gwp.plot_raw_dvafit(PLOTDIR, sampleid = sampleid,seed=seed, fix_quarter=quarter,plotfit=pf)

#     plot_chip_sky_frame(RESULTSDIR ,sampleid=sampleid,plot_chip=1,animate_chip=0,plot_sky=1,animate_sky=0,n_samples=-1, plot_arrow=0)

i_samp = 1
sampleid = f'brightestnonsat100_rot_mo{i_samp}'
# plot_chip_sky_frame(RESULTSDIR ,sampleid=sampleid,plot_chip=1,animate_chip=1,plot_sky=1,animate_sky=1,n_samples=-1, plot_arrow=0)

plot_chip_sky_frame(RESULTSDIR ,sampleid=sampleid,plot_chip=1,animate_chip=1,plot_sky=0,animate_sky=0,n_samples=-1, plot_arrow=0)
