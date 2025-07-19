import os
import kepler_astrometry_catalog.plotting as gwp
from kepler_astrometry_catalog.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, "pca_eig")
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

seed, quarter = 1, 12

# sampleid = f'brightestnonsat100_rot_mo{i_samp}'
# sampleid = "ppa_catalog_brightestnonsat100_q12"
sampleid = "brightestnonsat100_rot"
gwp.plot_pca_eig(PLOTDIR, sampleid=sampleid, quarter=quarter,n_comp=10, dvacorr=1,conv2glob=1, mag = 5e5,comp_eig=0)
# gwp.plot_pca_eig(
#     PLOTDIR, sampleid=sampleid, quarter=quarter, n_comp=3, dvacorr=0, conv2glob=1
# )
