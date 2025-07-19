import os
from kepler_astrometry_catalog import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), "data")
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), "results")
PAPERDIR = os.path.join(os.path.dirname(__path__[0]), "paper")
TABLEDIR = os.path.join(os.path.dirname(__path__[0]), "results", "tables")
MOVIEDIR = os.path.join(os.path.dirname(__path__[0]), "results", "movies")

LOCALDIR = os.path.join(os.path.expanduser("~"), ".kepler_astrometry_catalog")

CTDCACHEDIR = os.path.join(LOCALDIR, "ctd_cache")

listofdirs = [DATADIR, RESULTSDIR, PAPERDIR, LOCALDIR, TABLEDIR, CTDCACHEDIR, MOVIEDIR]
for l in listofdirs:
    if not os.path.exists(l):
        os.mkdir(l)
### in case needed later:
# import platform
# if platform.uname().node == 'discovery1.usc.edu':
#     pass
