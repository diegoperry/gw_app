"""
This script processes command line arguments for the StarCatalog (get_star_catalog) class.
It downloads:
    - The KIC (Kepler Input Catalog) (~/.kepler_astrometry_catalog/kepler_kic_v10.csv), if it doesn't already exist
    - The lightcurves shell script (~/.kepler_astrometry_catalog/lightcurve_getters/kepler_lightcurves_Q*_long.sh), if get_lightcurves_shell is set to True
    - The rotational period data (./data/stellar_rotation/*), if use_prot is set to True
It processes the KIC data to get the star catalog:
    - Filters out low quality stars (use_pqual)
It then saves the catalog to a CSV file in the results/tables directory.
See the /kepler_astrometry_catalog/get_star_catalog.py for more details.

Usage:
    python get_cat.py [options]

Options:
    --sampleid <str>                Type of star catalog (default: brightestnonsat)
    --use_pqual <bool>              Flag to use pqual - filter low quality stars (default: 1)
    --use_prot <bool>               Flag to use prot - filter rotating stars (default: 1)
    --get_lightcurves_shell <bool>  Flag to get lightcurves shell script (default: 0)
    --lc_quarter <int>              Quarter for lightcurves (default: None)
    --save-cat <bool>               Flag to save catalog (default: 1)
    --nmax <int>                    Maximum number of stars (default: -1, i.e. all stars in catalog)
    --ext_cat <str>                 External catalog (default: "")
    --target-m <str>                Target modules (chip number) to take stars from (default: "")
    --target-o <str>                Target outputs (chip's quadrant number) to take stars from (default: "")
    --add-str <str>                 Provide string to add to catalog name (default: "")
"""

import argparse
from dotenv import find_dotenv
import sys
import os

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")

from kepler_astrometry_catalog.get_star_catalog import StarCatalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process command line arguments for StarCatalog."
    )

    # Add command line arguments
    parser.add_argument(
        "--sampleid", type=str, default="brightestnonsat", help="Type of star catalog"
    )
    parser.add_argument("--use_pqual", type=str, default="1", help="Flag to use pqual")
    parser.add_argument("--use_prot", type=str, default="1", help="Flag to use prot")
    parser.add_argument(
        "--get_lightcurves_shell",
        type=str,
        default="0",
        help="Flag to get lightcurves shell",
    )
    parser.add_argument(
        "--lc_quarter", type=int, default=None, help="Quarter for lightcurves"
    )
    parser.add_argument(
        "--save-cat", type=str, default="1", help="Flag to save catalog"
    )
    parser.add_argument("--nmax", type=int, default=-1, help="Maximum number of stars")
    parser.add_argument("--ext_cat", type=str, default="", help="External catalog")

    parser.add_argument(
        "--target-m",
        default="",
        type=str,
        help="Target modules to take stars from. Provide a comma-separated list of ints",
    )
    parser.add_argument(
        "--target-o",
        default="",
        type=str,
        help="Target outputs to take stars from. Provide a comma-separated list of ints",
    )
    parser.add_argument(
        "--add-str", type=str, default="", help="Provide string to add to catalog name."
    )

    args = parser.parse_args()

    use_pqual = args.use_pqual.lower() in ["true", "1", "t", "y", "yes"]
    use_prot = args.use_prot.lower() in ["true", "1", "t", "y", "yes"]
    get_lightcurves_shell = args.get_lightcurves_shell.lower() in [
        "true",
        "1",
        "t",
        "y",
        "yes",
    ]
    save_cat = args.save_cat.lower() in ["true", "1", "t", "y", "yes"]

    try:
        if (args.target_m == " ") or (args.target_o == " "):
            target_m = []
            target_o = []
        else:
            target_m = (
                [int(item) for item in args.target_m.split(",")]
                if args.target_m
                else []
            )
            target_o = (
                [int(item) for item in args.target_o.split(",")]
                if args.target_o
                else []
            )

    except ValueError:
        # Handle the error, perhaps by logging an error message and exiting the script
        print("Error: All elements of target_m and target_o must be integers.")
        exit(1)

    if "rot" in args.sampleid:
        use_prot = True

    sc = StarCatalog(
        sampleid=args.sampleid,
        use_pqual=use_pqual,
        use_prot=use_prot,
        get_lightcurves_shell=get_lightcurves_shell,
        lc_quarter=args.lc_quarter,
        save_cat=save_cat,
        nmax=args.nmax,
        ext_cat=args.ext_cat,
        target_m=target_m,
        target_o=target_o,
        add_str=args.add_str,
    )

# sc = StarCatalog(
#     cat_type="all",
#     nmax=-1,
#     use_pqual=False,
#     use_prot=True,
#     get_lightcurves=True,
#     lc_quarter=12,
#     save_cat=True,
#     batch=True,
#     ibatch=ibatch,
#     batch_array_len=array_len,
# )

# nstar = 500

# for quarter in [11,12]:

#     # _mo1: target_m = [18]
#     target_m = [18]
#     sc = StarCatalog(cat_type='brightestnonsat',target_m = target_m,add_str='_mo1',  ext_cat='', use_pqual=0,use_prot=1,get_lightcurves_shell=0,lc_quarter=quarter, save_cat=1,nmax=nstar)

#     # _mo2: target_m = [18], target_o = [3]
#     target_m = [18]
#     target_o = [3]
#     sc = StarCatalog(cat_type='brightestnonsat',target_m = target_m,target_o=target_o, add_str='_mo2',  ext_cat='', use_pqual=0,use_prot=1,get_lightcurves_shell=0,lc_quarter=quarter, save_cat=1,nmax=nstar)

#     # _mo3: target_m = [6,11,12,13,16,17]
#     target_m = [6,11,12,13,16,17]
#     sc = StarCatalog(cat_type='brightestnonsat',target_m = target_m,add_str='_mo3',  ext_cat='', use_pqual=0,use_prot=1,get_lightcurves_shell=0,lc_quarter=quarter, save_cat=1,nmax=nstar)

#     # _mo4: target_m = [6,11,12,13,16,17,9,10,14,15,20], target_o=[2,2,2,2,2,2,4,4,4,4,4] (one-to-one)
#     target_m = [6,11,12,13,16,17,9,10,14,15,20]
#     target_o = [2,2,2,2,2,2,4,4,4,4,4]
#     sc = StarCatalog(cat_type='brightestnonsat',target_m = target_m,target_o=target_o,add_str='_mo4',  ext_cat='', use_pqual=0,use_prot=1,get_lightcurves_shell=0,lc_quarter=quarter, save_cat=1,nmax=nstar)

#     print(f'done with quarter {quarter}',flush=1)

# assuming all lightcurves have been downloaded
# sc = StarCatalog(cat_type='brightestnonsat',use_pqual=0,use_prot=1,get_lightcurves_shell=0,lc_quarter=12, save_cat=1,nmax=1000, ext_cat='')
