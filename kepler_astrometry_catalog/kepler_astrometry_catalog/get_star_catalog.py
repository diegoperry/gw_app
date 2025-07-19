"""
This script defines the StarCatalog class builds a star catalog based on the user's input parameters.
It's pipeline for getting the star catalog is as follows:
    - get_starlist()
        * get_data.get_kic_catalog():
            Get the full Kepler input catalog (KIC) if not already downloaded (at ~/.kepler_astrometry_catalog/kepler_kic_v10.csv)
        * get_data.get_lightcurve_scripts():
            Get all the lightcurve shell scripts (at ~/.kepler_astrometry_catalog/lightcurve_getters/kepler_lightcurves_Q*_long.sh)
        * process_data.get_kepler_star():
            Process the KIC data to get the star catalog as follows
                + Lists all KIC stars IDs and sorts them by quarter (from the getter scripts)
                + Creates a DataFrame with the following columns:
                    - kicid: Kepler Input Catalog ID
                    - inQ0, inQ1, ..., inQ17: Whether the star was observed in the given quarter
                    - total_quarters: Total number of quarters the star was observed
                + Writes the DataFrame to a CSV file (at results/tables/stars_observed_by_kepler.csv)
        * process_data.get_xkic():
            Merge the `stars_observed_by_kepler.csv` with selected columns from the Kepler v10 data
            and saves to a new file (at results/tables/stars_observed_by_kepler_X_KIC.csv).
            The selected columns are:
                [kic_degree_ra, kic_dec, kic_pmra, kic_pmdec, kic_kepmag, kic_kepler_id, kic_2mass_id, kic_pq]
                (See https://archive.stsci.edu/kepler/data_search/help/quickcol.html for more info)
        * reads the `stars_observed_by_kepler_X_KIC.csv` file as a DataFrame (df) and filters out bad stars:
            - screen_bad_stars():
                Filters out stars in the `data/bad_stars.txt` file
            - screen_rotation_stars() (if use_prot is True):
                Filters out rotating stars using the `results/tables/rot_period_cat.csv` file
            - screen_quality_flag() (if use_pqual is True):
                Filters out stars with quality flags (range: 0 to 8) less or equal to 6
            - Filter out stars according to the sampleid:
                > select_brightestnonsat() (if sampleid is "brightestnonsat"):
                    Filters out saturated stars and selects the brightest non-saturated stars as follows:
                        + Filters out stars with infinite magnitude measurements
                        + Filters out stars with magnitude < 11.3 + 0.1 (Gilliland+10 https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.160G/abstract)
                        + Filters out stars measured in less than 3 quarters
                > select_largeparallax() if sampleid is "large_parallax":
                    Sort stars with large parallax values first (from the Gaia-Kepler catalog)
                > select_from_user() if sampleid is "user_input":
                    Filter out stars using an external catalog that the user provides
            - (OPTIONAL) select_user_kicid_list() (if sampleid has "kicid_list" in it):
                Keep only the stars in the user list provided in the external catalog (ext_cat)
                This catalog is not filtered out by the other filters (except for bad stars)
        * select_module_output():
            Select specific modules and outputs from the DataFrame (df) based on the target_m and target_o values:
            Uses get_raDec2Pix() function, which uses ra_dec_2_pix from raDec2Pix code to convert the ra and dec values
            to module and output values, then filters the DataFrame based on the target_m and target_o values
    - save_catalog():
        Saves the DataFrame to a CSV file in the results/tables directory
    - get_lightcurves_shell():
        Downloads the lightcurve shell scripts if needed
"""

import os
import gzip
import shutil

import urllib.request
from urllib.parse import urlparse
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from dataclasses import dataclass, field
from astropy.table import Table

from kepler_astrometry_catalog.paths import TABLEDIR, DATADIR, LOCALDIR
import kepler_astrometry_catalog.get_data as gd
import kepler_astrometry_catalog.process_data as proc


@dataclass(kw_only=True)
class StarCatalog:
    sampleid: str = "brightestnonsat"  # type of star catalog
    use_pqual: bool = True  # screen out low quality stars
    use_prot: bool = True  # screen out rotating stars
    nmax: int = 100  # maximum number of stars
    save_cat: bool = False  # save catalog
    get_lightcurves_shell: bool = False  # get lightcurves shell script
    lc_quarter: None  # set only 1 quarter to download
    nbatch: int = 0  # for batch processing (0 for all)
    target_m: list = field(default_factory=list)  # target modules (chip number)
    target_o: list = field(
        default_factory=list
    )  # target outputs (chip's quadrant number)
    add_str: str = ""  # additional string to add to catalog name
    ext_cat: None

    def __post_init__(self):
        qualstr = "_qual" if self.use_pqual else ""
        protstr = "_rot" if self.use_prot else ""
        nmaxstr = str(self.nmax) if self.nmax > 0 else ""
        # set sampleid string if not already in correct format
        if ("rot" in self.sampleid) or any(char.isdigit() for char in self.sampleid):
            self.IDSTRING = self.sampleid
        # TalAdi: use external kicid list as filename
        elif "kicid_list" in self.sampleid:
            self.IDSTRING = self.sampleid
        else:
            self.IDSTRING = f"{self.sampleid}{nmaxstr}{qualstr}{protstr}{self.add_str}"
        # run the get starlist
        self.df = self.get_starlist()

        # save the catalog
        if self.save_cat:
            self.save_catalog(TABLEDIR)

        # get the lightcurve shell scripts, if needed.
        if self.get_lightcurves_shell:
            gd.get_catalog_quarter_shell(
                self.df, self.IDSTRING, self.lc_quarter, download_lc=True
            )
            if self.nbatch > 0:
                gd.get_catalog_batch_shell(
                    self.IDSTRING, qrts=[self.lc_quarter], nbatch=self.nbatch
                )

    def get_starlist(self):
        # read the full catalog
        tablepath = os.path.join(TABLEDIR, "stars_observed_by_kepler_X_KIC.csv")
        if os.path.isfile(tablepath):
            df = pd.read_csv(tablepath)
        else:
            print("Oops. Looks like you don't have the catalog. Let's create that.")
            print("This might take a while...")
            print("First, let's download the OG KIC catalog.")
            gd.get_kic_catalog()
            print(
                "Now let's get all the lightcurve getters. This will take a long time."
            )
            gd.get_lightcurve_scripts()
            print("Done.")
            print(
                "Now let's process all of the above to make the actual catalog we'll start with."
            )
            proc.get_kepler_star()
            proc.get_xkic()
            print("Finally done! Now loading the catalog.")
            df = pd.read_csv(tablepath)

        print(f"Original length: {len(df)}")
        ## screen bad stars
        df = self.screen_bad_stars(df)
        print(f"After screening bad stars: {len(df)}")
        # TalAdi: use external kicid list
        if "kicid_list" in self.sampleid:
            sdf = self.select_user_kicid_list(df)
            return sdf

        if self.use_prot:
            df = self.screen_rotation_stars(df)
            print(f"After screening rotating stars: {len(df)}")
        if self.use_pqual:
            df = self.screen_quality_flag(df)
            print(f"After screening quality flag stars: {len(df)}")

        if "brightestnonsat" in self.sampleid:
            sdf = self.select_brightestnonsat(df)
            print(f"After selecting brightestnonsat stars: {len(sdf)}")

        elif "large_parallax" in self.sampleid:
            sdf = self.select_largeparallax(df)

        elif "all" in self.sampleid:
            sdf = df

        elif "user_input" in self.sampleid:
            sdf = self.select_from_user(df)

        sdf = self.select_module_output(sdf)
        print(f"After selecting module output stars: {len(sdf)}")

        return sdf if self.nmax < 0 else sdf.head(n=self.nmax)

    def select_module_output(self, df):
        if len(self.target_m) == 0 and len(self.target_o) == 0:
            return df
        print("Now selecting specific modules and outputs.")
        ### FIXME: need a way to get quarter time
        # from kepler_astrometry_catalog.clean import Clean
        from astropy.time import Time
        from kepler_astrometry_catalog.helpers import get_raDec2Pix

        # cl = Clean(verbose=0,dvacorr=1,pca=0,save_pca_eigvec=0,max_nstars=1,seed=42,quarter=self.lc_quarter)
        # bjdrefi = 2454833
        # time_bjd = cl.time[0,0] + bjdrefi
        time_bjd = 2455932.4086584207
        init_mjds = Time(time_bjd, format="jd", scale="tdb")

        m, o, _, _ = get_raDec2Pix(df.kic_degree_ra, df.kic_dec, init_mjds)

        if len(self.target_m) == len(self.target_o):
            sel = np.zeros(len(df), dtype=bool)
            for m_, o_ in zip(self.target_m, self.target_o):
                sel = sel | ((m == m_) & (o == o_))
            return df[sel]

        sel = np.ones(len(df), dtype=bool)
        if len(self.target_m) > 0:
            sel = sel & np.isin(m, self.target_m)
        if len(self.target_o) > 0:
            sel = sel & np.isin(o, self.target_o)
        return df[sel]

    def save_catalog(self, outdir):
        fn = os.path.join(outdir, f"{self.IDSTRING}.csv")
        self.df.to_csv(fn, index=False)
        print(f"Saved catalog to {fn}")

    def select_brightestnonsat(self, df):
        # mag and N_quart basic screening
        # cite: Gilliland+10 https://ui.adsabs.harvard.edu/abs/2010ApJ...713L.160G/abstract
        saturation_kepmag = 11.3
        # avoid saturated stars by going a bit fainter than actual saturation
        cutoff_kepmag = saturation_kepmag + 0.1
        # select ones with finite magnitude measurement
        sel = ~pd.isnull(df.kic_kepmag)
        df = df[sel]
        # select unsaturated ones
        sel = df.kic_kepmag > cutoff_kepmag
        df = df[sel]
        # select ones measured over 3 quarters
        # FIXME: do we really need to implement this cut? should be freq. dependent, right?
        sel = df.total_quarters >= 3
        df = df[sel]
        sdf = df.sort_values(by="kic_kepmag", ascending=True)
        return sdf

    def select_largeparallax(self, df):
        # get gaia-kepler catalog, if it doesn't exist

        gaiacatpath = os.path.join(DATADIR, "astrometry", "kepler_dr3_good.fits")
        if not os.path.isfile(gaiacatpath):
            gd.get_gaia_kepler_catalog()
        ## load catalog and switch to pandas array
        _gdf = Table.read(gaiacatpath, format="fits").to_pandas()
        gdf = _gdf.rename({"kepid": "kicid"}, axis="columns")
        new = df.merge(gdf, on="kicid", how="left")
        sdf = new.sort_values(by="parallax", ascending=False)
        return sdf

    def select_user_kicid_list(self, df):
        udf = pd.read_csv(self.ext_cat)
        # keep only the stars in the user list
        sdf = df.merge(udf, on="kicid", how="inner")
        return sdf

    def select_from_user(self, df):
        ## filter out stars using an external catalog that the user provides.
        ## catalog should have a column "kicid".
        udf = pd.read_csv(self.ext_cat)
        sdf = df.merge(udf, on="kicid", how="left")
        return sdf

    def screen_bad_stars(self, df):
        print("First, let's screen any bad stars.")
        badstars = pd.read_csv(os.path.join(DATADIR, "bad_stars.txt"))
        sel = ~np.in1d(df.kicid, badstars.kicids)
        return df[sel]

    def screen_rotation_stars(self, df):
        print("Now let's screen rotating stars.")
        if os.path.isfile(os.path.join(TABLEDIR, "rot_period_cat.csv")):
            prot_df = pd.read_csv(os.path.join(TABLEDIR, "rot_period_cat.csv"))
        else:
            print("Oops -- you don't have the rotation table. Let's get that.")
            ##FIXME: url moved? this doesn't work.
            gd.get_rotation_catalogs()
            print("Catalogs downloaded. Now processing.")
            proc.get_rperiod()
            prot_df = pd.read_csv(os.path.join(TABLEDIR, "rot_period_cat.csv"))
        # print out the total number
        # print("total source", len(prot_df))
        # print("in McQuillan:", sum(prot_df["in_MQ"] == 1))
        # print(
        #     "in Santos", sum(((prot_df["in_S_GF"] == 1) | prot_df["in_S_MK"] == 1))
        # )
        sel = ~np.in1d(df.kicid, prot_df.kicids)
        return df[sel]

    def screen_quality_flag(self, df):
        # 0-8
        print("Let's screen for Kepler quality flags.")
        pq_min = 6
        sel = df.kic_pq >= pq_min
        return df[sel]
