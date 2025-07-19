"""
This script processes command line arguments for the Clean (kepler_astrometry_catalog.clean) class.  
It performs the following steps:  
    - Cleans astrometric data based on input parameters.  
    - Optionally injects a fake gravitational wave signal (FakeSignal) into the dataset.  
    - Applies PCA for systematic noise reduction if enabled.  
    - Saves the cleaned and processed data to a FITS file if specified.  

Usage:  
    python script.py <sampleid> [options]  

Arguments:  
    sampleid <str>                 Identifier for the sample to process.  

Options:  
    --fitsfn <str>                 Filename for output FITS file (default: None).  
    --fake-signal                  Flag to inject a fake gravitational wave signal (default: False).  
    --verbose <bool>               Flag to set verbosity level (default: True).  
    --quarter <int>                Kepler observing quarter to process (default: 12).  
    --dvacorr <bool>               Apply differential velocity aberration correction (default: True).  
    --use-poscorr <bool>           Use positional correction (default: True).  
    --pca <bool>                   Apply PCA for systematic noise reduction (default: False).  
    --pca-n-comps <int>            Number of PCA components to retain (default: 3).  
    --save_pca_eigvec <bool>       Save PCA eigenvectors (default: True).  
    --remove_earth_points <bool>   Remove Earth contamination points (default: True).  
    --conv2glob <bool>             Convert to global coordinates (default: True).  
    --read-fits <bool>             Read pre-existing FITS file instead of processing (default: True).  
    --write-cleaned-data <bool>    Write cleaned data to output file (default: False).  
    --max-nstars <int>             Maximum number of stars to process (default: -1, i.e., all).  
    --seed <int>                   Random seed for reproducibility (default: 42).  
    --use-psf <bool>               Apply PSF correction (default: False).  
    --use-fake-global <bool>       Use global fake signal model (default: True).  

Fake Signal Parameters (only if --fake-signal is enabled):  
    --frequency <float>            GW frequency in Hz (default: 1e-6).  
    --mc <float>                   Chirp mass in solar masses (default: 1.0e9).  
    --dl <float>                   Luminosity distance in Mpc (default: 20).  
    --source-l <float>             Source galactic longitude in degrees (default: 76.3).  
    --source-b <float>             Source galactic latitude in degrees (default: 13.5).  
    --thekla-dir <str>             Directory for saving fake signal model (default: ../../thekla/data/).  

The script cleans the data using the Clean class and injects a gravitational wave signal if requested.  
If PCA is enabled, the script applies principal component analysis for noise reduction.  
Processed data is optionally saved to a FITS file.  

See kepler_astrometry_catalog.clean and kepler_astrometry_catalog.fake_signal for implementation details.  
"""  

import numpy as np
import sys
import argparse
import astropy.units as u
import os

from dotenv import find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")


from kepler_astrometry_catalog.clean import Clean, save_avg_to_fits
from kepler_astrometry_catalog.fake_signal import FakeSignal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument("sampleid", type=str, help="Set the sampleid")
    parser.add_argument(
        "--fitsfn",
        type=str,
        help="Filename for output fits.",
    )
    parser.add_argument(
        "--fake-signal", action="store_true", help="Set the add_fake_signal"
    )
    parser.add_argument(
        "--verbose", type=str, default="1", help="Set the verbosity level"
    )
    parser.add_argument("--quarter", type=int, default=12, help="Set the quarter")
    parser.add_argument("--dvacorr", type=str, default='1', help="Set the dvacorr")
    parser.add_argument("--use-poscorr", type=str, default='1', help="Set the dvacorr")
    parser.add_argument("--pca", type=str, default="0", help="Set the pca")
    parser.add_argument("--pca-n-comps", type=int, default=3, help="Set the number of pca components")
    parser.add_argument(
        "--save_pca_eigvec", type=str, default="1", help="Set the save_pca_eigvec"
    )
    parser.add_argument(
        "--remove_earth_points", type=int, default=1, help="Set the remove_earth_points"
    )
    parser.add_argument("--conv2glob", type=int, default=1, help="Set the conv2glob")
    parser.add_argument("--read-fits", type=str, default='1', help="Set the read_fits")
    parser.add_argument(
        "--write-cleaned-data", type=str, default='0', help="Set the write-cleaned-data"
    )
    parser.add_argument(
        "--max-nstars", type=int, default=-1, help="Set the max_nstars"
    )
    parser.add_argument("--seed", type=int, default=42, help="Set the seed")
    parser.add_argument("--use_psf", type=int, default=0, help="Set the use_psf")
    parser.add_argument(
        "--use_fake_global", type=int, default=1, help="Set the use_fake_global"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1e-6,
        help="Set the frequency. Must be in Hz.",
    )
    parser.add_argument(
        "--mc",
        type=float,
        default=1.0e9,
        help="Set the chirp mass. Must be in solar masses.",
    )
    parser.add_argument(
        "--dl", type=float, default=20, help="Set the lum distance. Must be in Mpc"
    )
    parser.add_argument(
        "--source-l", type=float, default=76.3, help="Set the source_l. Must be in deg."
    )
    parser.add_argument(
        "--source-b", type=float, default=13.5, help="Set the source_b. Must be in deg."
    )
    parser.add_argument(
        "--thekla-dir",
        type=str,
        default="../../thekla/data/",
        help="Directory where thekla will read from.",
    )

    args = parser.parse_args()
    sampleid = args.sampleid
    ADD_FAKE_SIGNAL = args.fake_signal
    if ADD_FAKE_SIGNAL:
        frequency = args.frequency
        mc = args.mc
        dl = args.dl
        source_l = args.source_l
        source_b = args.source_b
        thekdir = args.thekla_dir
    verbose = args.verbose.lower() in ['true', '1', 't', 'y', 'yes']
    quarter = args.quarter
    dvacorr = args.dvacorr.lower() in ['true', '1', 't', 'y', 'yes']
    use_poscorr = args.use_poscorr.lower() in ['true', '1', 't', 'y', 'yes']
    pca = args.pca.lower() in ['true', '1', 't', 'y', 'yes']
    save_pca_eigvec = args.save_pca_eigvec.lower() in ['true', '1', 't', 'y', 'yes']
    remove_earth_points = args.remove_earth_points
    conv2glob = args.conv2glob
    read_fits = args.read_fits.lower() in ['true', '1', 't', 'y', 'yes']
    write_cleaned_data = args.write_cleaned_data.lower() in ['true', '1', 't', 'y', 'yes']
    max_nstars = args.max_nstars
    seed = args.seed
    use_psf = args.use_psf
    use_fake_global = args.use_fake_global
    fitsfn = args.fitsfn

    cl = Clean(
        verbose=verbose,
        quarter=quarter,
        sampleid=sampleid,
        dvacorr=dvacorr,
        pca=pca,
        pca_n_comps=args.pca_n_comps,
        save_pca_eigvec=save_pca_eigvec,
        remove_earth_points=remove_earth_points,
        conv2glob=conv2glob,
        read_fits=read_fits,
        write_cleaned_data=write_cleaned_data,
        max_nstars=max_nstars,
        seed=seed,
        use_poscorr=use_poscorr,
        use_psf=use_psf,
        use_fake_global=use_fake_global,
        fitsfn=fitsfn,
    )
    ##### add injection if needed
    if ADD_FAKE_SIGNAL:
        print(ADD_FAKE_SIGNAL)
        print("adding fake signal.")
        fn = fitsfn
        time = cl.time[0, :]
        star_ra, star_dec = cl.ra, cl.dec
        fs = FakeSignal(
            sampleid=sampleid,
            freq=frequency * u.Hz,
            mc=mc * u.Msun,
            dl=dl * u.Mpc,
            source_l=source_l * u.deg,
            source_b=source_b * u.deg,
            ts=time,
            star_ra=np.nanmean(star_ra, axis=1),
            star_dec=np.nanmean(star_dec, axis=1),
        )
        fs.save_model_to_pickle(thekdir)
        dra, ddec = fs.data
        print(f"Average fake dra (in rad): {np.average(np.abs(dra)*u.deg.to(u.rad))}")
        print(f"Average fake ddec (in rad): {np.average(np.abs(ddec)*u.deg.to(u.rad))}")
        print(f"Strain tensor: {fs.h}")
        cl.set_ctd(dra=dra, ddec=ddec)
        cl.do_pca(save_eigvec=1)
        cl.write_data(fake_signal=fs)
