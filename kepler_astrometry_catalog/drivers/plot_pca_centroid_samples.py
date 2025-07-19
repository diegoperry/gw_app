import os
import numpy as np
import kepler_astrometry_catalog.plotting as gwp
from kepler_astrometry_catalog.paths import RESULTSDIR


PLOTDIR = os.path.join(RESULTSDIR, "pcafit")
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

seed, quarter = 1, 12
n_components = 89
gwp.plot_pca_fit(PLOTDIR, sampleid='brightestnonsat100_rot',seed=seed, quarter=quarter, n_components=n_components,n_samples=10, dvacorr=True, plotfit=1,addsuffix='90',mag=5e5)

# for n_components in [3]:

# n_components = 3
# sampleid = 'brightestnonsat100_rot_mo1'
# sampleid = "ppa_catalog_brightestnonsat100_q12"
# sampleid = "brightestnonsat100_rot"
# ppalist = np.array(
#     [
#         10005147,
#         5390769,
#         9051558,
#         3532901,
#         6266498,
#         8410787,
#         10003270,
#         11911418,
#         12218888,
#         10272923,
#     ]
# )

# for i_samp in range(1,5):

#     sampleid = f'brightestnonsat100_rot_mo{i_samp}'

# no dva correction, dense matrix
# gwp.plot_pca_fit(PLOTDIR, seed=seed, quarter=quarter, n_components=n_components,n_samples=10)

# dva, dense matrix
# gwp.plot_pca_fit(PLOTDIR, seed=seed, sampleid=sampleid, quarter=quarter, n_components=n_components,n_samples=10, dvacorr=1, plotfit=1, conv2glob=0)


# gwp.plot_pca_fit(
#     PLOTDIR,
#     seed=seed,
#     sampleid=sampleid,
#     quarter=quarter,
#     n_components=n_components,
#     n_samples=10,
#     dvacorr=0,
#     plotfit=1,
#     conv2glob=1,
#     starlist=ppalist,
# )

# no dva, use all valid centroid data
# gwp.plot_pca_fit(PLOTDIR, seed=seed, quarter=quarter, n_components=n_components,n_samples=10, useTrunSVD=True)

# dva, use all valid centroid data
# gwp.plot_pca_fit(PLOTDIR, seed=seed, quarter=quarter, n_components=n_components,n_samples=10, dvacorr=True,useTrunSVD=True, plotfit=pf)
