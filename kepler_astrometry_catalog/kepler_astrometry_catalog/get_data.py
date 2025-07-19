"""
This one-time driver script downloads all lightcurves_getter .sh files and the
kepler_kic_v10 csv.
"""

import os, gzip, shutil, sys
import subprocess
import urllib.request
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from numpy import array as nparr
from glob import glob
from kepler_astrometry_catalog.paths import LOCALDIR, DATADIR, TABLEDIR
from astropy.io import fits


def get_lightcurve_scripts():
    # download lightcurves .sh data: file size 45MB/file
    # link: https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads#lc
    print("Getting lightcurve download scripts", flush=1)
    fileString = "https://archive.stsci.edu/missions/kepler/download_scripts/lightcurves/kepler_lightcurves_Q"
    for quart in range(18):
        url = fileString + f"{quart:02d}" + "_long.sh"
        fname = os.path.basename(urlparse(url).path)
        folder = os.path.join(LOCALDIR, "lightcurve_getters")
        if not os.path.exists(folder):
            os.mkdir(folder)
        fpath = os.path.join(folder, fname)  # its current location
        if os.path.isfile(fpath):
            continue
        urllib.request.urlretrieve(url, fpath)
    return 0


def get_kic_catalog():
    # download kepler_v10 csv: file size 1.1G
    # link: https://archive.stsci.edu/pub/kepler/catalogs/, https://archive.stsci.edu/kepler/kic.html
    # see also: https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract
    url = "https://archive.stsci.edu/pub/kepler/catalogs/kepler_kic_v10.csv.gz"
    fname = os.path.basename(urlparse(url).path)
    gzpath = os.path.join(LOCALDIR, fname)
    csvpath = csvpath = gzpath.rsplit(".", 1)[0]
    if not os.path.isfile(csvpath):
        print("gz file does not exist, downloading...", flush=1)
        if not os.path.isfile(gzpath):
            urllib.request.urlretrieve(url, gzpath)
            print(".gz file downloaded", flush=1)
        with gzip.open(gzpath, "rb") as f_in, open(csvpath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("upzip finished", flush=1)
    return 0


def get_rotation_catalogs():
    # download rotational period data
    # McQuillan(2.4MB): https://iopscience.iop.org/article/10.1088/0067-0049/211/2/24#apjs492452t1
    # Santos MK(1.8MB): https://iopscience.iop.org/article/10.3847/1538-4365/ab3b56
    # Santos GF(4.8MB): https://iopscience.iop.org/article/10.3847/1538-4365/ac033f
    print("Getting rotation catalog", flush=1)
    urls = [
        "https://content.cld.iop.org/journals/0067-0049/211/2/24/revision1/apjs492452t1_mrt.txt",
        "https://content.cld.iop.org/journals/0067-0049/244/1/21/revision1/apjsab3b56t3_mrt.txt",
        "https://content.cld.iop.org/journals/0067-0049/255/1/17/revision1/apjsac033ft1_mrt.txt",
    ]
    for url in urls:
        fname = os.path.basename(urlparse(url).path)
        folder = os.path.join(DATADIR, "stellar_rotation")
        if not os.path.exists(folder):
            os.mkdir(folder)
        txtpath = os.path.join(folder, fname)
        if not os.path.isfile(txtpath):
            urllib.request.urlretrieve(url, txtpath)
    return 0


def get_gaia_kepler_catalog():
    # download gaia-kepler cross match
    # https://gaia-kepler.fun/
    # one-to-one match
    # made by: Megan Bedell
    # https: // www.dropbox.com/s/pk5cgwjxanczn6b/kepler_dr3_good.fits?dl = 0
    print("Getting gaia-kepler catalog", flush=1)
    # FIXME: this doesn't work.
    url = "https://www.dropbox.com/s/pk5cgwjxanczn6b/kepler_dr3_good.fits?dl=1"
    fname = os.path.basename(urlparse(url).path)
    folder = os.path.join(DATADIR, "astrometry")
    if not os.path.exists(folder):
        os.mkdir(folder)
    txtpath = os.path.join(folder, fname)
    if not os.path.isfile(txtpath):
        urllib.request.urlretrieve(url, txtpath)
    return 0


def get_catalog_quarter_shell(df, IDSTRING, get_quarter=None, download_lc=False):

    # creates shell scripts
    outpath = os.path.join(LOCALDIR, IDSTRING)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    desired_kicids = nparr(df.kicid).astype(np.int64)
    keplerquarters = range(18) if get_quarter is None else nparr([get_quarter])

    for q in keplerquarters:
        longgetter = os.path.join(
            LOCALDIR,
            "lightcurve_getters",
            f"kepler_lightcurves_Q{str(q).zfill(2)}_long.sh",
        )
        assert os.path.exists(longgetter)

        outgetter = os.path.join(outpath, f"kepler_lightcurves_Q{str(q).zfill(2)}.sh")
        if os.path.exists(outgetter):
            print(f"You already have the fixed shell script for quarter {q}", flush=1)
            continue

        print(f"Don't have shell scripts for Q{q} yet, starting now...", flush=1)

        with open(longgetter, "r") as f:
            lines = f.readlines()

        lines_to_write = []
        for ix, l in enumerate(lines):
            if ix == 0:
                lines_to_write.append(l)
                continue
            kicid = np.int64(l.split(" ")[-2].split("/")[3].lstrip("0"))
            if np.isin(kicid, desired_kicids):
                l = l.replace("--progress ", "")[:-1]
                localdir = l.split(" ")[6]
                line = f"if ! [ -f {localdir} ]; then \n\t{l}\nfi\n"
                lines_to_write.append(line)
            # if np.mod(ix+1, 10000) == 0:
            #     print(f"{ix+1}/{len(lines)}",flush=1)

        with open(outgetter, "w") as f:
            f.writelines(lines_to_write)

        print(
            f"Made {os.path.basename(outgetter)} with {len(lines_to_write)-1} entries...",
            flush=1,
        )
    if download_lc:
        print("Now downloading lightcurves...")
        download_lightcurves(fullpath=outgetter, IDSTRING=IDSTRING, qrt=get_quarter)
    if get_quarter is None:
        # also make a get-all-quarters script, very fast, always make on the fly
        print("Getting all quarters", flush=1)
        all_getter = os.path.join(outpath, "get_all.sh")
        shpaths = sorted(glob(os.path.join(outpath, "kepler_lightcurves_Q*.sh")))
        with open(all_getter, "w") as f:
            f.write("#!/bin/sh\n")
            for file in shpaths:
                f.write(f"( source {file}; wait ) & wait\n")
        print("Made get_all.sh", flush=1)

    return


def get_catalog_batch_shell(IDSTRING, qrts=range(18), nbatch=10):
    outpath = os.path.join(LOCALDIR, IDSTRING)
    for q in qrts:
        quarter_shellpath = os.path.join(
            outpath, f"kepler_lightcurves_Q{str(q).zfill(2)}.sh"
        )
        if not os.path.exists(quarter_shellpath):
            print(f"Quarter {q} shell script not found.", flush=1)
            continue

        with open(quarter_shellpath, "r") as f:
            lines = f.readlines()
        header = lines[0]

        ntotal = (len(lines) - 1) // 3  # if statement gives 3 lines
        nseg = ntotal // nbatch
        nrem = ntotal % nbatch
        nbatch_ls = [nseg + 1] * nrem + [nseg] * (nbatch - nrem)
        istart = 1

        for ibatch, nb in enumerate(nbatch_ls):
            batch_lines = lines[istart : istart + nb * 3]
            istart += nb * 3
            batch_lines.insert(0, header)
            batch_path = os.path.join(
                outpath, f"kepler_lightcurves_Q{str(q).zfill(2)}_{str(ibatch)}.sh"
            )
            with open(batch_path, "w") as f:
                f.writelines(batch_lines)
            print(f"Made {os.path.basename(batch_path)} with {nb} entries...", flush=1)
    return


def download_lightcurves(fullpath=None, IDSTRING=None, qrt=None, ibatch=None):
    if fullpath is None:
        outpath = os.path.join(LOCALDIR, IDSTRING)
        fullpath = os.path.join(
            outpath, f"kepler_lightcurves_Q{str(qrt).zfill(2)}_{str(ibatch)}.sh"
        )
        if not os.path.exists(fullpath):
            print(f"Quarter {qrt} batch {ibatch} shell script not found.", flush=1)
            return
    subprocess.check_call(["chmod", "+x", fullpath])
    datadir = os.path.join(DATADIR, "lightcurves")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    os.chdir(datadir)
    subprocess.call(["sh", fullpath])
    return


def reload_corrupt_FITS(IDSTRING, quarter, ibatch, niter=5):
    outpath = os.path.join(LOCALDIR, IDSTRING)
    datadir = os.path.join(DATADIR, "lightcurves")
    cor_path = os.path.join(
        outpath, f"kepler_lightcurves_Q{str(quarter).zfill(2)}_{str(ibatch)}_corrupt.sh"
    )
    header = "#!/bin/sh\n"

    if os.path.exists(cor_path):
        with open(cor_path, "r") as f:
            corrupt_fns = f.readlines()
        corrupt_fns = corrupt_fns[1:]
    else:
        #### initial scanning
        batch_path = os.path.join(
            outpath, f"kepler_lightcurves_Q{str(quarter).zfill(2)}_{str(ibatch)}.sh"
        )
        if not os.path.exists(batch_path):
            print(f"Quarter {quarter} batch {ibatch} shell script not found.", flush=1)
            return
        with open(batch_path, "r") as f:
            lines = f.readlines()
        lines = lines[1:]

        corrupt_fns = []
        nlines = len(lines) // 3
        for i in range(nlines):
            line = lines[3 * i + 1][1:]
            localdir = line.split(" ")[6].strip("'")
            path = os.path.join(datadir, localdir)
            try:
                hdulist = fits.open(path)
                d = hdulist[1].data
                hdr = hdulist[0].header
                hdulist.close()
            except:
                corrupt_fns.append(line)

    if len(corrupt_fns) == 0:
        if os.path.exists(cor_path):
            os.remove(cor_path)
        print("no corrupted downloads", flush=1)
        return
    else:
        print(
            f"found {len(corrupt_fns)} corrupted files, iteratively re-downloading...",
            flush=1,
        )

    maxiter = niter
    while niter > 0:

        corrupt_fns.insert(0, header)
        with open(cor_path, "w") as f:
            f.writelines(corrupt_fns)
        download_lightcurves(cor_path)

        nlines = len(corrupt_fns) - 1
        next_fns = []
        for line in corrupt_fns[1:]:
            localdir = line.split(" ")[6].strip("'")
            path = os.path.join(datadir, localdir)
            try:
                hdulist = fits.open(path)
                d = hdulist[1].data
                hdr = hdulist[0].header
                hdulist.close()
            except:
                next_fns.append(line)

        corrupt_fns = next_fns.copy()
        if len(corrupt_fns) == 0:
            os.remove(cor_path)
            print("fixed all corrupted downloads", flush=1)
            return

        print(
            f"iteration {maxiter-niter}, {len(corrupt_fns)} corrupted files left",
            flush=1,
        )
        niter -= 1

    print("max iteration reached!!!", flush=1)

    return
