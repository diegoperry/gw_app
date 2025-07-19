# kepler_astrometry_catalog
Code that creates the Kepler astrometry catalog

## Installation steps:
NOTE: This repo has already been installed on the USC discovery cluster. You can proceed to the run steps.
1. Clone this repo.
    Note: use `git clone --recurse-submodules <repository-url>` in order to clone the submodules `estoiles` and `Kepler-RaDex2Pix`.

2. Install the conda environment using: `conda env create -f environment.yml`

3. Activate the conda environment: `conda activate kac`

4. Install submodules `estoiles` and `Kepler-RaDex2Pix` by running `pip install .` in each of them.

## Systematic pipeline documentation

### HDF5 input/output file schema

The centroids for the entire dataset are stored in one HDF5 file, at location `TBD`.
Since HDF5 files are lazily-loaded into Python (i.e. if you try to access a slice of a dataset you don't try to load all the rest of it), this is OK.

In practice, sub-datasets should be made based on quality flags/magnitude/etc. using the auxillary `.csv` catalogs.
The stars are cross-matched by KIC ID.

The schema (not formally validated, sorry) of the file is as follows:
```
group: root
    # Data for the first quarter of observations. N here can be potentially different between quarters.
    - group: 'Q1'
        - PyTables table: 'pos'. Centroid observations. All columns except for 'kicid' have shape (N stars, T observed times). 'kicid' has shape (N stars,) i.e. each row entry is just one number. For stars with a total # of observations < T, unobserved entries for time/x/y/z are filled in with NaNs; module/output/quality values should be ignored. When constructing this table, the index of unobserved time entries should be aligned with the same exposure as observed stars. Columns as follows:
            kicid   (uint32):    Integer KIC ID.
            time    (float64):   Time since first observation (globally, not from start of quarter). In seconds. This is different for each star since due to CCD readout the observed time can be slightly different.
            x       (float64):   Centroid position, in pixels.
            y       (float64):   Centroid position, in pixels.
            module  (uint8):     Integer module index. Stars can move between modules with time.
            output  (uint8):     Integer 'quadrant' index within module. Stars can move between outputs with time.
            quality (uint32):    Flag, which if nonzero, indicates the observation has some quality issue(s). See https://lightkurve.github.io/lightkurve/tutorials/2-creating-light-curves/2-2-kepler-noise-1-data-gaps-and-quality-flags.html for a reference on the values.
        - OPTIONAL (may not be needed) PyTables table 'errors'. All columns except for 'kicid' have shape (N stars, T observed times). Times assumed to map to 'pos' table. For stars with a total # of observations < T, unobserved times for all columns are filled in with NaNs. Columns as follows:
            kicid  (uint32):    Integer KIC ID.
            cov_xx (float64):   Per-observation estimated variance (sigma^2) on centroid, in x-direction. In units of pixel^2.
            cov_yy (float64):   Per-observation estimated variance (sigma^2) on centroid, in y-direction. In units of pixel^2.
            cov_xy (float64):   Per-observation estimated covariance between x/y axes. In units of pixel^2.
    # Data for the second quarter of observations.
    - group: 'Q2'
        ...
    ...
    - PyTables table: 'stars'. Auxillary info about stars. All columns are length (M total stars). The ordering of stars by kicid is not guaranteed to be the same as any ordering in the quarter groups! Columns as follows:
        kicid  (uint32):    Integer KIC ID.
        ra     (float64):   Nominal RA from KIC, in deg. J2000 coordinate system.
        dec    (float64):   Nominal declination from KIC, in deg. J2000 coordinate system.
        kepmag (float64):   Kepler-band magnitude. ‘KEPMAG’ in the KIC: see column 15 in https://lweb.cfa.harvard.edu/kepler/kic/format/format.html.
    # Auxillary data for each cleaning step
    - group: 'cleaning'
        - PyTables array: 'order'. Array of strings, each of which is a key for the groups below. Encodes the order of cleaning steps; for each step, you should append the step's name/whatever key you use below to the end of this array.
        - group: 'step1' # Placeholder step name.
            Whatever you want goes here. Doesn't have to be a PyTables object, can be arbitrary data that's serializable into HDF5.
        - group: 'step2'
            ...
```

## Run steps:
1. Activate the conda environment: `conda activate kac`
    1. If on cluster, navigate to the project folder: `/project/kmpardo_1034/kepler_astrometry_catalog/`

2. PRE-CLEAN STEP: Adjust and run `config/pre-clean.yaml` or `config/cluster-pre-clean.yaml` if on cluster.
    1. This code creates a star catalog and downloads all of the files that you will need from the NASA Kepler archival data. This is the most time consuming step, but it only needs to be done once.
    2. Set the `SAMPLEID`: for now, best to use `brightestnonsatXX_rot`, where XX is replaced by the number of stars you want to use. For a personal computer, use 100. The first time you run this, it will take several hours.
    3. IF ON CLUSTER: **set `GET_CENTROIDS = FALSE`. They should have all been downloaded already for the main samples we are working with.
    4. Don't mess with the other setting for now.
    5. To run, navigate to the `config/` folder and run: `maestro run CONFIG_NAME`
    6. Check status by navigating to `data/maestro/CONFIGNAME_datetime/` and running `maestro status ./`
3. CLEAN-ANALYZE STEP: Adjust and run `config/clean-analyze.yaml` or `config/cluster-clean-analyze.yaml`.
    1. This code cleans the data and then produces various diagnostic plots. This will be the main code you'll work with. When working with large samples, this can take up to 10 hours to run, but usually it runs in just a few minutes.
    2. Set the `SAMPLEID` to one that matches what you've run with the pre-clean step.
    3. Set the various clean options. To use all of the stars in your sample, set `NSTARS = -1`.
    4. Follow steps 2.5 - 2.6 to run this.
    5. Check results of the plots in `data/maestro/CONFIGNAME_datetime/`.

## OLD:
1. Run `drivers/get_cat.py`. This will download all the files you need and set up a star catalog based on options you can set in that file. It will also download the necessary lightcurve files, if get_lightcurves = True.
2. Run `drivers/get_raw_centroids.py`. This will read from FITS files and save centroids in csv format. Default keywords save MOM_CENTR and iteratively find out the global ra and dec from MOM_CENTR. 
3. Call `Clean` object (e.g. `drivers/make_clean.py`). This cleans NaNs and Earth point cadences in centroids, computes centroid residuals and saves to FITS file. Keyword options allow for module-output filtering, choice between MOM and PSF, DVA and POSCORR. For centroids processing, we have four main options: MOM/PSF, DVA/POSCORR, LOC/GLOB, iteratively generated (IG)/linearly transformed(LT). They make up 16 sets of centroid residuals:
    1. MOM, DVA, GLOB, IG: rctd=MOM, ra, dec directly from reading csv
    2. MOM, DVA, GLOB, LT: __directly rotates MOM into global frame, so DVA is not corrected__
    3. MOM, DVA, LOC, IG: rctd=MOM, corr=DVA
    4. MOM, DVA, LOC, LT: __not implemented in `Clean`__. When we use fake local coordinates, this is usually for simulating injection signal in local coordinates. See `drivers/signal_injection_test.live_test_fake_local` for this.
    5. MOM, POSCORR, GLOB, IG: no effect
    6. MOM, POSCORR, GLOB, LT: no effect
    7. MOM, POSCORR, LOC, IG: rctd=MOM, corr=POSCORR
    8. MOM, POSCORR, LOC, LT: __avoid__, fake local coordinate generation only gives the post-DVA residual, so if desired we could set loc=POSCORR + fake local residual, but this is not physically motivated.
    9. PSF, DVA, GLOB, IG: rctd=PSF, directly read ra and dec. __assumes `drivers/get_raw_centroids.py` has been called with setting `use_psf_for_local=1`
    10. PSF, DVA, GLOB, LT: directly rotates PSF into global frame, no DVA correction
    11. PSF, DVA, LOC, IG: rctd=PSF, corr=DVA
    12. PSF, DVA, LOC, LT: __not implemented__
    13. PSF, POSCORR, GLOB, IG: __avoid__, IG assumes DVA correction, so it is inconsistent by demanding POSCORR correction. 
    14. PSF, POSCORR, GLOB, LT: same as (PSF, DVA, GLOB, LT)
    15. PSF, POSCORR, LOC, IG: rctd=PSF, corr=POSCORR
    16. PSF, POSCORR, LOC, LT: __avoid__, LT method assumes DVA correction, inconsistent. 
    Many of these cases can be not physically motivated, so we didn't test their implementation. The most frequently used combination are the following:
    1. MOM/PSF, DVA, GLOB, IG/LT: used for MCMC study of actual signal
    2. MOM/PSF, DVA, LOC, LT: used for local/global contrast with signal injection 
4. Run functions in `drivers/signal_injection_tests.py` to plot, make movies, run signal injection tests etc.

For plotting, 
1. plot_centroid_samples.py to plot raw centroids
2. plot_centroid_timex.py to plot raw centroids with time
3. plot_pca_centroid_samples.py to plot DVA residual and PCA fits, also plot the final residual 
4. plot_pca_eig.py to plot the leading eigenvectors with time