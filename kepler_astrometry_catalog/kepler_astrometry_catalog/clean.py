"""
Defines the Clean class, which should be used to clean the raw centroids and save the cleaned centroids to a fits file.

Currently, it does the following:
- If `read_fits` is True, and the file exists, it reads the centroids from the fits file.
- Otherwise:
    1. Reads the raw centroids from the csv files using get_raw_centroids.
    2. Cleans the raw centroids using clean_raw_ctd:
        - Applies qual flags.
        - Removes earth points.
    3. Sets the corrected centroids using set_ctd:
        - Converts to mas according to whether to convert to global coordinates and whether to correct for DVA.
    4. Writes the cleaned data to a fits file in fitsfn path (default: `RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"`).

Experimental features:
- PCA: Does PCA for x and y components fitted separately.
- Plots: Makes diagnostic plots.

"""

#############
## IMPORTS ##
#############

#
# standard imports
#
from astropy.io import fits
import astropy.units as u
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field
import datetime
from numpy import array as nparr
from scipy.optimize import curve_fit

from sklearn.decomposition import PCA, TruncatedSVD  # FactorAnalysis
from scipy.sparse import csr_matrix
from pathlib import Path

#
# non-standard imports
#
# from aesthetic.plot import set_style, savefig
from kepler_astrometry_catalog.paths import RESULTSDIR, DATADIR
from kepler_astrometry_catalog.quarters_to_datestrs import quarter_datestr_dict
from kepler_astrometry_catalog.helpers import (
    get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o,
    given_lcpath_get_time_ctd_dva,
    get_df_keyword,
    _get_data_keyword,
)
from raDec2Pix import raDec2Pix

rdp = raDec2Pix.raDec2PixClass()


@dataclass
class Clean:
    quarter: int = 12
    sampleid: str = "brightestnonsat100_rot"
    max_nstars: int = 100
    dvacorr: bool = False
    pca: bool = False
    pca_n_comps: int = 3
    useTrunSVD: bool = False
    remove_earth_points: bool = True
    n_earth_blocks: int = 150
    save_pca_eigvec: bool = False
    diagnostic_plots: bool = False
    write_cleaned_data: bool = False
    read_fits: bool = True
    conv2glob: bool = False
    seed: int = 0  ## random seed for selecting random stars from catalog.
    verbose: bool = False
    target_m: list = field(default_factory=list)
    target_o: list = field(default_factory=list)
    use_psf: bool = False
    use_poscorr: bool = False
    use_fake_global: bool = False
    fitsfn: str = None

    def __post_init__(self):
        if self.fitsfn is None:
            self.fitsfn = (
                RESULTSDIR + f"/cleaned_centroids/{self.sampleid}_{self.quarter}.fits"
            )
        if self.read_fits and os.path.exists(self.fitsfn):
            if self.verbose:
                print(f"Reading from {self.fitsfn}", flush=1)
            self.read_from_fits()
        else:
            self.clean_raw_ctd()
            self.set_ctd(dra=None, ddec=None)

        if self.pca:
            print("Doing pca...")
            self.do_pca(save_eigvec=self.save_pca_eigvec)
        if self.diagnostic_plots:
            self.make_diag_plots()
        if self.write_cleaned_data:
            self.write_data()
        return 0

    def clean_raw_ctd(self):
        """
        Clean the raw centroids by applying qual flags and removing earth points.
        """
        [self.rctd_x, self.rctd_y, self.time, self.qual] = self.get_raw_centroids()
        ## Apply qual flags
        qualsel = self.qual == 0
        if not self.useTrunSVD:
            self._sel = (
                (qualsel.any(axis=0))
                & (~np.isnan(self.rctd_x).any(axis=0))
                & (~np.isnan(self.rctd_y).any(axis=0))
            )
        else:
            self._sel = (~np.isnan(self.rctd_x).all(axis=0)) & (
                ~np.isnan(self.rctd_y).all(axis=0)
            )

        if self.remove_earth_points:
            self._sel = self.block_earth_point()
        self.rctd_x = self.rctd_x[:, self._sel]
        self.rctd_y = self.rctd_y[:, self._sel]
        self.time = self.time[:, self._sel]
        return

    def set_ctd(self, dra=None, ddec=None, local_signal_str=""):
        """
        Set the corrected centroids and convert to mas according to:
        1. Whether to convert to global coordinates.
        2. Whether to correct for DVA.

        Parameters:
        -----------
        dra: float
            RA offset in radians.
        ddec: float
            Dec offset in radians.
        local_signal_str: str
            Local signal string to be used for reading the cached data.

        Returns:
        --------
        None
        """
        if (self.conv2glob == 0) and (
            len(local_signal_str) > 0
        ):  ## read from cache if not converting to global and local_signal_str is provided.
            CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{self.sampleid}")
            self.rctd_x = np.load(f"{CACHEDIR}/{local_signal_str}_x.npy")
            self.rctd_y = np.load(f"{CACHEDIR}/{local_signal_str}_y.npy")

        # Conversion factors
        pix2mas = 3.98e3
        rad2mas = 3600 * 180 / np.pi * 1000
        multiplier = rad2mas if self.conv2glob else pix2mas

        ## if converting to global coordinates
        if self.conv2glob:
            if self.use_fake_global:
                from kepler_astrometry_catalog.helpers import cache_module_output_axis

                cache_module_output_axis(self.quarter, np.mean(self.time, axis=0))
                self.ctd_x, self.ctd_y = self.rotate_to_glob()
                multiplier = pix2mas
            else:
                self.ctd_x, self.ctd_y = self.use_global_coord(dra=dra, ddec=ddec)
            ## if using dva or poscorr, we need to correct the centroids.
            if (self.dvacorr) or (self.use_poscorr):
                ## still  calc dvacorr, but don't set as ctd. just save in fits.
                __ = self.correct_dva()
        else:
            if self.dvacorr:
                [self.ctd_x, self.ctd_y] = self.correct_dva()
            else:
                self.ctd_x = self.rctd_x
                self.ctd_y = self.rctd_y
        ## Convert to mas
        self.ctd_x *= multiplier
        self.ctd_y *= multiplier
        return

    def rotate_to_glob(self):

        from kepler_astrometry_catalog.helpers import get_fake_global

        self.get_module_output()
        try:
            ctd_x, ctd_y = get_fake_global(
                self.rctd_x,
                self.rctd_y,
                self.module[:, 0],
                self.output[:, 0],
                self.quarter,
                np.mean(self.time, axis=0),
            )
        except:
            print("Hm fake global didn't work. Need to regenerate fake axes.")
            from kepler_astrometry_catalog.helpers import cache_module_output_axis

            cache_module_output_axis(
                self.quarter, np.mean(self.time, axis=0), force_refresh=True
            )
            ctd_x, ctd_y = get_fake_global(
                self.rctd_x,
                self.rctd_y,
                self.module[:, 0],
                self.output[:, 0],
                self.quarter,
            )

        # center it
        ctd_x -= np.nanmean(ctd_x, axis=1)[:, np.newaxis]
        ctd_y -= np.nanmean(ctd_y, axis=1)[:, np.newaxis]

        # also compute ra, dec
        pix2deg = 3.98 / 3600
        ddec = ctd_y * pix2deg  # in degrees
        dra = ctd_x * pix2deg / np.cos(np.deg2rad(self.all_survey_dec))[:, np.newaxis]
        self.ra = np.array(self.all_survey_ra)[:, np.newaxis] + dra
        self.dec = np.array(self.all_survey_dec)[:, np.newaxis] + ddec
        return ctd_x, ctd_y

    def get_raw_centroids(self):
        """
        Get raw centroids from the lightcurves.

        Returns:
        --------
        ctd_x: np.ndarray
            x centroids
        ctd_y: np.ndarray
            y centroids
        time: np.ndarray
            time
        qual: np.ndarray
            quality flags
        """
        lcdir = os.path.join(DATADIR, "lightcurves", "Kepler")
        if self.verbose:
            print(lcdir, flush=1)
        if not os.path.exists(lcdir):
            NotImplementedError

        self.lcpaths = get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o(
            self.sampleid,
            self.seed,
            self.max_nstars,
            fix_quarter=self.quarter,
            verbose=self.verbose,
            target_m=self.target_m,
            target_o=self.target_o,
        )
        print("finished getting the lcpaths", flush=1)

        self.nstars = len(self.lcpaths)
        # if "ppa" in self.sampleid:
        #     selcsvpath = [
        #         lcpath.replace(".fits", "_ppa.csv") for lcpath in self.lcpaths
        #     ]
        #     selcsvpath = [x.replace("lightcurves/Kepler", "ppa") for x in selcsvpath]
        #     self.csvpaths = [x.replace("_ppa.csv", "_rawppa.csv") for x in selcsvpath]

        # else:
        selcsvpath = [lcpath.replace(".fits", "_ctds.csv") for lcpath in self.lcpaths]
        selcsvpath = [x.replace("lightcurves/Kepler", "ctd") for x in selcsvpath]
        self.csvpaths = [x.replace("_ctds.csv", "_rawctds.csv") for x in selcsvpath]

        if self.verbose:
            # print("Number of stars in catalog =", len(self.df), flush=1)
            print("Number of stars chosen = ", str(self.nstars), flush=1)
        self.kicid = np.zeros((len(self.csvpaths)))
        _ctd_x = []
        _ctd_y = []
        time = []
        qual = []
        self.all_survey_ra = []
        self.all_survey_dec = []

        ctd_x_lb = "psf_x" if self.use_psf else "mom_x"
        ctd_y_lb = "psf_y" if self.use_psf else "mom_y"

        for ix, csvpath in enumerate(self.csvpaths):
            _ = pd.read_csv(csvpath)
            try:
                _ctd_x.append(_[ctd_x_lb])
                _ctd_y.append(_[ctd_y_lb])
            except:
                print("Failed to read ctds from csv:\n", csvpath)
                continue
            time.append(_["time"])
            qual.append(_["qual"])

            lcpath = self.lcpaths[ix]
            self.kicid[ix] = np.int64(lcpath.split("/")[-2].lstrip("0"))

            hdulist = fits.open(lcpath)
            ra, dec = hdulist[0].header["RA_OBJ"], hdulist[0].header["DEC_OBJ"]
            hdulist.close()
            self.all_survey_ra.append(ra)
            self.all_survey_dec.append(dec)

        _ctd_x = nparr(_ctd_x)
        _ctd_y = nparr(_ctd_y)
        time = nparr(time)
        qual = nparr(qual)

        # ## Example of old way. Failed because too many open files.
        # # _ctd_x = nparr(list(map(
        # #     _get_data_keyword,
        # #     self.lcpaths, np.repeat('MOM_CENTR2', self.nstars), np.repeat(1, self.nstars)
        # # )))
        # _ctd_x = nparr([_get_data_keyword(f, "MOM_CENTR2") for f in self.lcpaths])
        # # row
        # _ctd_y = nparr([_get_data_keyword(f, "MOM_CENTR1") for f in self.lcpaths])

        # # variation of times of different stars is roughly 0.0003, negligible
        # time = nparr([_get_data_keyword(f, "TIME") for f in self.lcpaths])
        # qual = nparr([_get_data_keyword(f, "SAP_QUALITY") for f in self.lcpaths])
        # if self.verbose:
        #     print("finished getting centroid data on stars.", flush=1)

        ### FIXME: insert proper signal injection code here

        return _ctd_x, _ctd_y, time, qual

    def get_module_output(self):
        if hasattr(self, "module"):
            return self.module, self.output

        # check if module and output is different
        self.module = nparr([get_df_keyword(f, "module") for f in self.csvpaths])[
            :, self._sel
        ]
        self.output = nparr([get_df_keyword(f, "output") for f in self.csvpaths])[
            :, self._sel
        ]
        if self.verbose:
            ndrift = 0
            for i in range(len(self.module)):
                if (len(np.unique(self.module[i])) > 1) or (
                    len(np.unique(self.output[i])) > 1
                ):
                    ndrift += 1
            print(f"ndrift = {ndrift}", flush=1)
        return self.module, self.output

    def correct_dva(self, use_kep_corr=False):

        if (use_kep_corr) or (self.use_poscorr):
            if use_kep_corr:
                print("Using Kepler DVA corrections")
            else:
                print("Using Kepler poscorr")
            corr_x_lb = "poscorr_x" if self.use_poscorr else "dva_x"
            corr_y_lb = "poscorr_y" if self.use_poscorr else "dva_y"

            self.corr_x = nparr([get_df_keyword(f, corr_x_lb) for f in self.csvpaths])
            self.corr_y = nparr([get_df_keyword(f, corr_y_lb) for f in self.csvpaths])

            self.corr_x = self.corr_x[:, self._sel]
            self.corr_y = self.corr_y[:, self._sel]

            ctd_x = self.rctd_x - self.corr_x
            ctd_y = self.rctd_y - self.corr_y
        else:
            print("Fitting star DVA using sinusoids")
            ctd_x = np.zeros((len(self.kicid), len(self.rctd_x[0])))
            ctd_y = np.zeros((len(self.kicid), len(self.rctd_x[0])))
            self.corr_x = np.zeros((len(self.kicid), len(self.rctd_x[0])))
            self.corr_y = np.zeros((len(self.kicid), len(self.rctd_x[0])))
            self.dva_pars = np.zeros((len(self.kicid), 6, 2))
            for idx in range(0, len(self.kicid)):
                t = self.time[idx, :]
                momx = self.rctd_x[idx, :]
                momy = self.rctd_y[idx, :]
                for j, d in enumerate([momx, momy]):
                    # Loop over directions. Each fit separately.
                    # Choose an initial guess based on the data.
                    p0 = [
                        0.0,
                        np.std(d),
                        0.0,
                        0.0,
                        0.0,
                        np.mean(d),
                    ]
                    # Fit the DVA model.
                    popt, __ = curve_fit(dva_model, t, d, p0=p0)
                    self.dva_pars[idx, :, j] = popt

                    # Evaluate the model and compute residuals.
                    d_fit = dva_model(t, *popt)
                    if j == 0:
                        self.corr_x[idx] = d_fit
                        ctd_x[idx] = momx - d_fit
                    else:
                        self.corr_y[idx] = d_fit
                        ctd_y[idx] = momy - d_fit
        return ctd_x, ctd_y

    def use_global_coord(self, dra=None, ddec=None, scale=1.0):
        try:
            ra = nparr([get_df_keyword(f, "ra") for f in self.csvpaths])
            dec = nparr([get_df_keyword(f, "dec") for f in self.csvpaths])
        except:
            print(
                "ra dec was not iteratively generated, run get_raw_centroids with local_only=0 or set use_fake_global=1",
                flush=1,
            )
            exit()
        ra = ra[:, self._sel]
        dec = dec[:, self._sel]

        if dra is not None:
            ra += scale * dra * 180 / np.pi
        if ddec is not None:
            dec += scale * ddec * 180 / np.pi

        self.ra = ra
        self.dec = dec

        mean_ra = np.nanmean(ra, axis=1)[:, np.newaxis]
        mean_dec = np.nanmean(dec, axis=1)[:, np.newaxis]

        deg2rad = np.pi / 180

        dx = (ra - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad
        dy = (dec - mean_dec) * deg2rad

        self.module, self.output = self.get_module_output()

        return dx, dy

    def do_pca(self, save_eigvec=False, addsuffix=""):
        ## Does PCA for x and y components fitted separately.
        mean_ctd_x = np.nanmean(self.ctd_x, axis=1)
        X_x = self.ctd_x - mean_ctd_x[:, None]
        self.X_x_mean = np.nanmean(X_x, axis=0)[np.newaxis, :]
        X_x -= self.X_x_mean
        if self.useTrunSVD:
            _ind = tuple(np.argwhere(np.isnan(self.ctd_x)).T)
            X_x[_ind] = 0
            X_x = csr_matrix(X_x)
            pca_x = TruncatedSVD(n_components=X_x.shape[0])
        else:
            pca_x = PCA()
        pca_x.fit(X_x)

        mean_ctd_y = np.nanmean(self.ctd_y, axis=1)
        X_y = self.ctd_y - mean_ctd_y[:, None]
        self.X_y_mean = np.nanmean(X_y, axis=0)[np.newaxis, :]
        X_y -= self.X_y_mean
        if self.useTrunSVD:
            _ind = tuple(np.argwhere(np.isnan(self.ctd_y)).T)
            X_y[_ind] = 0
            X_y = csr_matrix(X_y)
            pca_y = TruncatedSVD(n_components=X_y.shape[0])
        else:
            pca_y = PCA()
        pca_y.fit(X_y)

        self.eigenvecs_x = pca_x.components_
        self.eigenvecs_y = pca_y.components_

        if save_eigvec:
            ## get ready to save data
            qstr = str(self.quarter).zfill(2)
            dvastr = "" if not self.dvacorr else "_dva"
            svdstr = "" if not self.useTrunSVD else "_TrunSVD"
            globstr = "" if not self.conv2glob else "_glob"

            pcadir = os.path.join(RESULTSDIR, "pca_eig", self.sampleid)
            if not os.path.exists(pcadir):
                os.makedirs(pcadir)

            asuf = "" if len(addsuffix) == 0 else f"_{addsuffix}"
            ## save eigenvectors
            eigvecpath_x = os.path.join(
                pcadir, f"Q{qstr}{dvastr}{svdstr}{globstr}{asuf}_x.txt"
            )
            eigvecpath_y = os.path.join(
                pcadir, f"Q{qstr}{dvastr}{svdstr}{globstr}{asuf}_y.txt"
            )

            np.savetxt(
                eigvecpath_x, np.concatenate([self.X_x_mean, self.eigenvecs_x], axis=0)
            )
            np.savetxt(
                eigvecpath_y, np.concatenate([self.X_y_mean, self.eigenvecs_y], axis=0)
            )

            if self.verbose:
                print(f"saved eigenvector_x to {eigvecpath_x}", flush=1)
                print(f"saved eigenvector_y to {eigvecpath_y}", flush=1)

        ## now get residuals
        basisvecs_x = self.eigenvecs_x[: self.pca_n_comps, :]
        basisvecs_y = self.eigenvecs_y[: self.pca_n_comps, :]

        coef_x = np.einsum("ij,kj", np.nan_to_num(basisvecs_x), np.nan_to_num(X_x))
        coef_y = np.einsum("ij,kj", np.nan_to_num(basisvecs_y), np.nan_to_num(X_y))
        model_x = np.einsum("ik,ij->kj", coef_x, np.nan_to_num(basisvecs_x))
        model_y = np.einsum("ik,ij->kj", coef_y, np.nan_to_num(basisvecs_y))

        ##FIXME maybe? these used to be ctd_x_pca_res, etc.
        self.ctd_x = X_x - model_x
        self.ctd_y = X_y - model_y

        if self.verbose:
            ctdx_res = (X_x - model_x).flatten()
            ctdy_res = (X_y - model_y).flatten()
            res = np.nanmean(np.sqrt(ctdx_res**2 + ctdy_res**2))
            logline = f"ncomp={self.pca_n_comps}, avg residuals, {res} mas"
            print(logline, flush=1)

        #### when we use the local coordinate version, the residual after subtracting all PCA components shows periodic patterns (dots on a regular grid). Not clear the exact reason but the global coord version doesn't have this.

        return

    def make_diag_plots(self):
        pass

    def write_data(self, fake_signal=None):
        """
        write data to fits file.

        - If fake_signal is provided, it writes the:
            - RA and DEC of the star.
            - The fake signal parameters.

        - Otherwise, it writes the following data:
            - kicid: KIC ID of the star.
            - time: Time of the observation.
            - sel_mask: Mask for the selected stars.
            - lcpaths: Path to the lightcurves.
            - csvpaths: Path to the csv files.
            - survey_ra: RA_OBJ of the star (local)
            - survey_dec: DEC_OBJ of the star. (local)
            - RAW_CENTROID_X: Raw x centroids.
            - RAW_CENTROID_Y: Raw y centroids.
            - CENTROID_X: Corrected x centroids.
            - CENTROID_Y: Corrected y centroids.

            If conv2glob is True, it also writes:
                - RA: RA of the star. (global)
                - DEC: DEC of the star. (global)
                - MODULE: Module of the star.
                - OUTPUT: Output of the star.

            If dvacorr or use_poscorr is True, it also writes:
                - POSCORR_X: Corrected x centroids.
                - POSCORR_Y: Corrected y centroids.
        """

        if fake_signal is not None:
            # fitsfn = fitsfn.replace(
            #     ".fits", f"_fake_{fake_signal.freq.value}_{fake_signal.mc.value}.fits"
            # )
            self.fitsfn = self.fitsfn.replace(".fits", "_fake.fits")
            new_hdul = fits.HDUList()  # Initialize new_hdul
            if os.path.exists(self.fitsfn):
                new_hdul = fits.open(
                    self.fitsfn, mode="update"
                )  # Update new_hdul if fitsfn exists
            col_ra = fits.Column(
                name="RA",
                array=self.ra,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            col_dec = fits.Column(
                name="DEC",
                array=self.dec,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            hdr2 = fits.Header({"EXTNAME": "FAKE_SIGNAL"})
            hdr2["FREQ"] = fake_signal.freq.value
            hdr2["MC"] = fake_signal.mc.value
            hdr2["DL"] = fake_signal.dl.value
            hdr2["SRC_L"] = fake_signal.source_l.value
            hdr2["SRC_B"] = fake_signal.source_b.value
            hdu2 = fits.BinTableHDU.from_columns([col_ra, col_dec], header=hdr2)
            try:
                new_hdul["FAKE_SIGNAL"] = hdu2  # reset the previous data
            except:
                new_hdul.append(hdu2)
            new_hdul.writeto(self.fitsfn, overwrite=True)
            print(f"Saved centroids to: {self.fitsfn}")
            return

        # if os.path.exists(self.fitsfn):
        #     new_hdul = fits.open(self.fitsfn, mode="update")
        # else:
        col_kicid = fits.Column(name="kicid", array=self.kicid, format="J")
        col_time = fits.Column(
            name="time",
            array=self.time,
            format=f"{len(self.time[0])}D",
            dim=f"({len(self.time[0])})",
        )
        col_sel = fits.Column(name="sel_mask", array=self._sel, format="L")

        col_lcpaths = fits.Column(
            name="lcpaths", format=f"{len(self.lcpaths[0])}A", array=self.lcpaths
        )
        col_csvpaths = fits.Column(
            name="csvpaths", format=f"{len(self.csvpaths[0])}A", array=self.csvpaths
        )
        col_as_ra = fits.Column(name="survey_ra", format="D", array=self.all_survey_ra)
        col_as_dec = fits.Column(
            name="survey_dec", format="D", array=self.all_survey_dec
        )

        col_rctd_x = fits.Column(
            name="RAW_CENTROID_X",
            array=self.rctd_x,
            format=f"{len(self.time[0])}D",
            dim=f"({len(self.time[0])})",
        )
        col_rctd_y = fits.Column(
            name="RAW_CENTROID_Y",
            array=self.rctd_y,
            format=f"{len(self.time[0])}D",
            dim=f"({len(self.time[0])})",
        )

        phdu = self.create_primary_hdu()
        hdr0 = fits.Header({"EXTNAME": "TIME"})
        hdu0 = fits.BinTableHDU.from_columns([col_time], header=hdr0)

        hdr = fits.Header({"EXTNAME": "SEL_MASK"})
        hdu_mask = fits.BinTableHDU.from_columns([col_sel], header=hdr)

        hdr = fits.Header({"EXTNAME": "PATHS"})
        hdu_path = fits.BinTableHDU.from_columns(
            [col_kicid, col_lcpaths, col_csvpaths, col_as_ra, col_as_dec],
            header=hdr,
        )

        hdr = fits.Header({"EXTNAME": "RAW_CENTROIDS"})
        hdu = fits.BinTableHDU.from_columns([col_rctd_x, col_rctd_y], header=hdr)

        new_hdul = fits.HDUList(
            [
                phdu,
                hdu_mask,
                hdu0,
                hdu_path,
                hdu,
            ]
        )
        # tag1 = "PSF" if self.use_psf else "MOM"
        # tag2 = "GLOB" if self.conv2glob else "LOC"
        # tag3 = "POSCORR" if self.use_poscorr else "DVA"
        # tag4 = "FAKE" if self.use_fake_global else "REAL"
        # ext_hdr = f"{tag4}_{tag2}_{tag1}_{tag3}_RESIDUAL"
        ext_hdr = "RESIDUALS"
        col_dvactd_x = fits.Column(
            name="CENTROID_X",
            array=self.ctd_x,
            format=f"{len(self.time[0])}D",
            dim=f"({len(self.time[0])})",
        )
        col_dvactd_y = fits.Column(
            name="CENTROID_Y",
            array=self.ctd_y,
            format=f"{len(self.time[0])}D",
            dim=f"({len(self.time[0])})",
        )
        hdr2 = fits.Header({"EXTNAME": ext_hdr})
        hdu2 = fits.BinTableHDU.from_columns([col_dvactd_x, col_dvactd_y], header=hdr2)
        # try:
        # new_hdul[ext_hdr] = hdu2  # reset the previous data
        # except:
        new_hdul.append(hdu2)

        if self.conv2glob:
            # tag1 = "PSF" if self.use_psf else "MOM"
            # tag2 = "FAKE" if self.use_fake_global else "REAL"
            # ext_hdr = f"{tag2}_{tag1}_RADEC_MODULE_OUTPUT"
            ext_hdr = "RADEC"
            # store the ra, dec, module and output
            col_ra = fits.Column(
                name="RA",
                array=self.ra,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            col_dec = fits.Column(
                name="DEC",
                array=self.dec,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            col_module = fits.Column(
                name="MODULE",
                array=self.module,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            col_output = fits.Column(
                name="OUTPUT",
                array=self.output,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            hdr2 = fits.Header({"EXTNAME": ext_hdr})
            hdu2 = fits.BinTableHDU.from_columns(
                [col_ra, col_dec, col_module, col_output], header=hdr2
            )
        if self.use_poscorr or self.dvacorr:
            ext_hdr = (
                "POSCORR" if self.use_poscorr else "DVA"
            )  # keeping the correction, which is
            col_dvax = fits.Column(
                name="CORR_X",
                array=self.corr_x,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            col_dvay = fits.Column(
                name="CORR_Y",
                array=self.corr_y,
                format=f"{len(self.time[0])}D",
                dim=f"({len(self.time[0])})",
            )
            hdr2 = fits.Header({"EXTNAME": ext_hdr})
            hdu2 = fits.BinTableHDU.from_columns([col_dvax, col_dvay], header=hdr2)
        try:
            new_hdul[ext_hdr] = hdu2  # reset the previous data
        except:
            new_hdul.append(hdu2)

        if not os.path.exists(self.fitsfn):
            Path(Path(self.fitsfn).parent).mkdir(parents=True, exist_ok=True)
            new_hdul.writeto(self.fitsfn, overwrite=True)

        new_hdul.close()
        print(f"Saved centroids to: {self.fitsfn}")
        return

    def create_primary_hdu(self):
        phdu = fits.PrimaryHDU()
        phdur = phdu.header
        phdur.set(
            "date",
            datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "UTC Date",
        )
        phdur.set("sampleid", self.sampleid, "ID for star sample")
        phdur.set("quarter", self.quarter, "Quarter")
        phdur.set("dvacorr", self.dvacorr, "DVA corrected?")
        phdur.set("poscorr", self.use_poscorr, "Poscorr corrected?")
        phdur.set("pca", self.pca, "PCA done?")
        phdur.set("pcacomps", self.pca_n_comps, "Number of PCA Components")
        phdur.set("global", self.conv2glob, "Converted to global?")
        phdur.set("fakeglob", self.use_fake_global, "Used fake global?")
        return phdu

    def read_from_fits(self):

        # to check the EXTNAME and column names, use:
        # for x in hdu[1:]:
        #     print(x.header['EXTNAME'])
        #     print(x.columns.names)

        hdu = fits.open(self.fitsfn)
        ## if you want to add checks of DVACORR or PCA, use header 0.
        ## e.g. dvacorr = hdu[0].header['DVACORR'] returns True or False.

        self.time = hdu["TIME"].data["time"]
        self._sel = hdu["SEL_MASK"].data["sel_mask"]
        self.csvpaths = hdu["PATHS"].data["csvpaths"]
        self.lcpaths = hdu["PATHS"].data["lcpaths"]
        self.all_survey_ra = hdu["PATHS"].data["survey_ra"]
        self.all_survey_dec = hdu["PATHS"].data["survey_dec"]
        self.kicid = hdu["PATHS"].data["kicid"]

        try:
            ext_hdr = ("PSF" if self.use_psf else "MOM") + "_RAW_CENTROIDS"
            self.rctd_x = hdu[ext_hdr].data["raw_centroid_x"]
            self.rctd_y = hdu[ext_hdr].data["raw_centroid_y"]
            print("have read raw centroids", flush=1)

            if self.conv2glob:
                tag1 = "PSF" if self.use_psf else "MOM"
                tag2 = "FAKE" if self.use_fake_global else "REAL"
                self.ra = hdu[f"{tag2}_{tag1}_RADEC_MODULE_OUTPUT"].data["ra"]
                self.dec = hdu[f"{tag2}_{tag1}_RADEC_MODULE_OUTPUT"].data["dec"]
                self.module = hdu[f"{tag2}_{tag1}_RADEC_MODULE_OUTPUT"].data["module"]
                self.output = hdu[f"{tag2}_{tag1}_RADEC_MODULE_OUTPUT"].data["output"]
            else:
                corr_lb = "POSCORR" if self.use_poscorr else "DVA"
                self.corr_x = hdu[corr_lb].data["corr_x"]
                self.corr_y = hdu[corr_lb].data["corr_y"]
            print("have read correction quantities and raDec", flush=1)

            tag1 = "PSF" if self.use_psf else "MOM"
            tag2 = "GLOB" if self.conv2glob else "LOC"
            tag3 = "POSCORR" if self.use_poscorr else "DVA"
            tag4 = "FAKE" if self.use_fake_global else "REAL"
            ext_hdr = f"{tag4}_{tag2}_{tag1}_{tag3}_RESIDUAL"
            self.ctd_x = hdu[ext_hdr].data["centroid_x"]
            self.ctd_y = hdu[ext_hdr].data["centroid_y"]
            print("have read the post-correction residuals", flush=1)

        except:
            if self.verbose:
                print(
                    "FITS exists, but the requested global/local data was not stored",
                    flush=1,
                )
            self.clean_raw_ctd()
            self.set_ctd()  # if doesn't exist, compute
        hdu.close()

        ## header 4 has pca corrected ones.
        ## to check for exact things in each header, print hdu[i].header.

        return

    def block_earth_point(self):
        ## take out ~3days(150) after earth points

        # method one: hardcode
        # earth_pt_flags = [8, 16392, 98312, 98328, 393224, 409608]
        # method 2: iterate through
        import lightkurve as lk

        earth_pt_flags = []
        for flag in np.unique(self.qual.flatten()):
            for item in lk.KeplerQualityFlags.decode(flag):
                if "Earth" in item:
                    earth_pt_flags.append(flag)

        ## create earth-point mask
        intervals = []
        for flag in earth_pt_flags:
            inds = np.argwhere(self.qual == flag)
            _, y = inds.T
            intervals.extend(list(np.unique(y)))

        sorted_intervals = []
        intervals.sort()
        for start in intervals:
            end = start + self.n_earth_blocks
            if len(sorted_intervals) == 0 or start > sorted_intervals[-1][1]:
                sorted_intervals.append([start, end])
                continue
            sorted_intervals[-1][1] = max(sorted_intervals[-1][1], end)

        earth_pt_mask = np.ones(self.qual.shape[1], dtype=bool)
        for start, end in sorted_intervals:
            earth_pt_mask[start:end] = 0

        return self._sel * earth_pt_mask


####### HELPER FUNCTIONS #######
# def get_raDec2Pix(ra, dec, time):
#     """
#     ra/dec: degrees
#     time: Astropy.Time object
#     """
#     m, o, r, c = rdp.ra_dec_2_pix(ra, dec, time.mjd)
#     return m, o, r, c


# def get_df_keyword(csvpath, key):
#     # FIXME: should move to helpers...but then need to deal with circular imports.
#     df = pd.read_csv(csvpath)
#     return nparr(df[key])


# def _get_data_keyword(fits_file, keyword, ext=1):
#     """
#     (copied directly from astrobase; credit Waqas Bhatti if you use this
#     function; pasted here to avoid import dependencies)

#     This gets a single data array out of a FITS binary table.

#     Parameters
#     ----------
#     fits_file : str
#         Path to fits file to open.

#     keyword : str
#         Keyword to access in the header.

#     ext : int
#         FITS extension number.

#     Returns
#     -------
#     val : ndarray
#         The value in the FITS data table header.

#     """

#     hdulist = fits.open(fits_file)

#     if keyword in hdulist[ext].data.names:
#         val = hdulist[ext].data[keyword]
#     else:
#         val = None

#     hdulist.close()
#     return val


def save_avg_to_fits(fn):
    # FIXME: this assumes certain headers exist. also should probably make this a class fn.
    with fits.open(fn) as hdu:
        time = hdu["TIME"].data["time"][0]
        ext_hdr = "FAKE_MOM_RADEC_MODULE_OUTPUT"
        module = hdu[ext_hdr].data["module"][:, 0]
        output = hdu[ext_hdr].data["output"][:, 0]
        ra = hdu[ext_hdr].data["ra"]
        dec = hdu[ext_hdr].data["dec"]

    mls = []
    ols = []
    avg_ra = np.empty([0, len(ra[0])])
    avg_dec = np.empty([0, len(dec[0])])

    all_module = np.array(
        [i for i in range(2, 5)]
        + [i for i in range(6, 21)]
        + [i for i in range(22, 25)]
    )
    all_module, all_output = np.meshgrid(all_module, np.arange(1, 5))
    all_module = all_module.flatten()
    all_output = all_output.flatten()

    for m, o in zip(all_module, all_output):
        sel = (module == m) & (output == o)
        if np.sum(sel) == 0:
            continue
        mls.append(m)
        ols.append(o)
        avg_ra = np.vstack([avg_ra, np.mean(ra[sel], axis=0)])
        avg_dec = np.vstack([avg_dec, np.mean(dec[sel], axis=0)])
    # save as a different file, save just the ra and dec

    def create_primary_hdu():
        phdu = fits.PrimaryHDU()
        phdur = phdu.header
        phdur.set("sampleid", "ppa_catalog_q12_12_avg", "ID for star sample")
        return phdu

    ## primary hdu
    phdu = create_primary_hdu()

    ## time
    hdr0 = fits.Header({"EXTNAME": "TIME"})
    col_time = fits.Column(name="time", format="D", array=time)
    hdu0 = fits.BinTableHDU.from_columns([col_time], header=hdr0)

    ## ra and dec
    ext_hdr = "RADEC"
    col_ra = fits.Column(
        name="RA",
        array=avg_ra,
        format=f"{len(avg_ra[0])}D",
        dim=f"({len(avg_ra[0])})",
    )
    col_dec = fits.Column(
        name="DEC",
        array=avg_dec,
        format=f"{len(avg_dec[0])}D",
        dim=f"({len(avg_dec[0])})",
    )
    hdr1 = fits.Header({"EXTNAME": ext_hdr})
    hdu1 = fits.BinTableHDU.from_columns([col_ra, col_dec], header=hdr1)

    ## module and output
    col_module = fits.Column(name="MODULE", array=mls, format="D")
    col_output = fits.Column(name="OUTPUT", array=ols, format="D")
    hdr2 = fits.Header({"EXTNAME": "MODULE_OUTPUT"})
    hdu2 = fits.BinTableHDU.from_columns([col_module, col_output], header=hdr2)

    new_hdul = fits.HDUList([phdu, hdu0, hdu1, hdu2])

    if not os.path.exists(self.fitsfn):
        Path(Path(self.fitsfn).parent).mkdir(parents=True, exist_ok=True)
    new_hdul.writeto(self.fitsfn, overwrite=True)
    new_hdul.close()
    print(f"saved to {self.fitsfn}", flush=1)
    return


def dva_model(t, phase, A, B, C, D, constant):
    fdva = (1.0 / (372.57 * u.day)).to(1.0 / u.day).value
    dva = A * np.cos(2 * np.pi * fdva * t + phase) + constant
    dva2 = B * (np.cos(2 * np.pi * fdva * t + phase)) ** 2
    dva3 = C * np.cos(2 * np.pi * fdva * t + phase) ** 3
    dva4 = D * np.cos(2 * np.pi * fdva * t + phase) ** 4
    return dva + dva2 + dva3 + dva4
