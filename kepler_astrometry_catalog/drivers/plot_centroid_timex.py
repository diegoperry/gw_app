import os
import argparse
import sys
from dotenv import find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")

import kepler_astrometry_catalog.plotting as gwp
from kepler_astrometry_catalog.paths import RESULTSDIR

def main():
    parser = argparse.ArgumentParser(description="Plot centroid samples timex")
    parser.add_argument("outpath", type=str, help="Output path")
    parser.add_argument("fitsfn", type=str, help="FITS file name")
    parser.add_argument("--seed", type=int, default=1, help="Seed value")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of samples")
    parser.add_argument("--extname", type=str, default="RADEC", help="Header string for FITS extension")
    args = parser.parse_args()

    gwp.plot_centroid_samples_timex(
        args.outpath,
        args.fitsfn,
        seed=args.seed,
        n_samples=args.n_samples,
        extname=args.extname
    )


if __name__ == "__main__":
    main()
