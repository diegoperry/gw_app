import os
import sys
from dotenv import find_dotenv
import argparse
import numpy as np

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")

from kepler_astrometry_catalog.get_centroids import get_raw_centroids
import datetime


# batch_size = 10
# i = int(os.getenv("SLURM_ARRAY_TASK_ID"))

# isample = i//4
# sampleid = f'brightestnonsat500_rot_mo{isample+1}'

# i = i%4
# iquarter = i//2
# quarter = 11+iquarter

# ibatch = i%2

# quarter = 12
# sampleid = "brightestnonsat100_rot"
# # batch_size = 5
# batch_process = 1
# ibatch = i


parser = argparse.ArgumentParser()
parser.add_argument("--quarter", type=int, default=12, help="Quarter value")
parser.add_argument(
    "--sampleid", type=str, default="brightestnonsat100_rot", help="Sample ID"
)

parser.add_argument(
    "--batch_process", type=str, default="False", help="Batch process value"
)
parser.add_argument("--reload", type=str, default="False", help="reload centroids?")
parser.add_argument("--local_only", default="False", type=str)
parser.add_argument("--get_poscorr", default="True", type=str)
parser.add_argument("--get_psf", default="True", type=str)
args = parser.parse_args()

batch_process = args.batch_process.lower() in ["true", "1", "t", "y", "yes"]
re = args.reload.lower() in ["true", "1", "t", "y", "yes"]
local_only = args.local_only.lower() in ["true", "1", "t", "y", "yes"]
get_poscorr = args.get_poscorr.lower() in ["true", "1", "t", "y", "yes"]
get_psf = args.get_psf.lower() in ["true", "1", "t", "y", "yes"]

if batch_process:
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    print("Number of SLURM arrays:", slurm_array_count)
    ibatch = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    nstar = int(''.join(filter(str.isdigit, args.sampleid)))
    batch_size = int(np.ceil(nstar/slurm_array_count))

else:
    try:
        batch_size = int(''.join(filter(str.isdigit, args.sampleid)))
    except:
        batch_size = 100
    ibatch = 0


get_raw_centroids(
    args.quarter,
    sampleid=args.sampleid,
    max_nstars=batch_size,
    ibatch=ibatch,
    batch_process=batch_process,
    reload=re,
    local_only=local_only,
    get_poscorr=get_poscorr,
    get_psf=get_psf,
)

print(f"{ibatch} done with getting centroids at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")