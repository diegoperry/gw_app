import os
import sys
import numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.animation as animation
from astropy.time import Time
import datetime

# from aesthetic.plot import savefig, set_style

from kepler_astrometry_catalog.helpers import (
    get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o,
    given_lcpath_get_time_ctd_dva,
    get_pca_basisvecs,
    get_lcpaths_given_kicids_quarter,
    comp_pca_basisvecs,
    read_dva_res,
)
from raDec2Pix import raDec2Pix
from raDec2Pix import raDec2PixModels as rm
from kepler_astrometry_catalog.clean import Clean
from kepler_astrometry_catalog.paths import RESULTSDIR
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

##### convenience function

deg2rad = np.pi / 180
rad2mas = 3600 * 180 / np.pi * 1000


def conv2cart(ra_, dec_):
    ## return the cartesian coordinates of displacement in mas
    mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
    mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
    dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
    dy = (dec_ - mean_dec) * deg2rad * rad2mas
    return dx, dy


def demean(arr):
    ## remove both the time-wise mean and per-time-slice mean
    arr_ = copy.deepcopy(arr)
    mean = np.mean(arr_, axis=1)[:, np.newaxis]
    arr_ -= mean
    mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
    arr_ -= mean_eigvec
    return arr_


def corr(x, y):
    return x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)


def centering(arr):
    ## remove the time-wise mean
    return arr - np.mean(arr, axis=1)[:, np.newaxis]


##### end of helper functions #####


def plot_raw_dvafit(
    outdir,
    seed=42,
    n_samples=10,
    verbose=1,
    ykey="MOM_CENTR",
    fix_quarter=None,
    sampleid="brightestnonsat100",
    plotfit=True,
    injected_sig_str="",
):
    path = os.path.join(outdir, sampleid)
    if not os.path.exists(path):
        os.mkdir(path)
    fitstr = "" if not plotfit else "_wfit"
    qstr = "" if fix_quarter is None else f"_Q{str(fix_quarter).zfill(2)}"
    isstr = "" if len(injected_sig_str) == 0 else f"_{injected_sig_str}"
    outpath = os.path.join(
        path, f"seed{seed}_nsamp{n_samples}{qstr}{fitstr}{isstr}.png"
    )

    # if os.path.exists(outpath):
    #     print(f'Found {outpath}, continue.')
    #     return

    plt.close("all")
    set_style()
    if n_samples == 5:
        fig, axs = plt.subplots(ncols=n_samples, figsize=(10 + 0.5, 2))
        flat_axs = axs
    elif n_samples == 10:
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10 + 0.5, 4))
        flat_axs = axs.flatten()
    else:
        raise NotImplementedError

    cl = Clean(
        sampleid=sampleid,
        quarter=fix_quarter,
        verbose=verbose,
        conv2glob=0,
        dvacorr=1,
        pca=0,
        save_pca_eigvec=0,
        remove_earth_points=1,
        max_nstars=-1,
        seed=seed,
    )
    sel_lcpaths = get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o(
        sampleid, seed, n_samples, fix_quarter=fix_quarter
    )
    if len(injected_sig_str) > 0:
        print("resetting ctd")
        cl.set_ctd(local_signal_str=injected_sig_str)

    t0 = np.nanmin(cl.time[0])

    lcpaths = np.array(cl.lcpaths)
    sel_i = [np.argwhere(lcpaths == x)[0][0] for x in sel_lcpaths]

    pix2mas = 3.98e3
    rctd_x = cl.rctd_x[sel_i] * pix2mas
    rctd_y = cl.rctd_y[sel_i] * pix2mas
    rctd_x -= np.nanmean(rctd_x, axis=1)[:, np.newaxis]
    rctd_y -= np.nanmean(rctd_y, axis=1)[:, np.newaxis]
    lcpaths = lcpaths[sel_i]

    #### FIXME: using dva, there's a constant offset
    dva_x = cl.corr_x[sel_i] * pix2mas
    dva_y = cl.corr_y[sel_i] * pix2mas
    dva_x -= np.nanmean(dva_x, axis=1)[:, np.newaxis]
    dva_y -= np.nanmean(dva_y, axis=1)[:, np.newaxis]

    for ix, (ax, lcpath) in enumerate(zip(flat_axs, lcpaths)):
        hdr = given_lcpath_get_time_ctd_dva(lcpath)[0]

        cmap = mpl.cm.get_cmap("cividis_r")
        color = cl.time[ix] - t0
        _p = ax.scatter(
            rctd_x[ix],
            rctd_y[ix],
            c=color,
            alpha=1,
            zorder=3,
            s=2,
            rasterized=True,
            linewidths=0,
            marker=".",
            cmap=cmap,
        )

        if plotfit:
            _p1 = ax.scatter(
                dva_x[ix],
                dva_y[ix],
                c="tab:red",
                alpha=0.5,
                zorder=4,
                s=1,
                rasterized=True,
                linewidths=0,
                marker=".",
            )

        ax.tick_params(labelsize="small")

        if plotfit and ix == 0:
            ax.plot([], [], color="tab:red", linewidth=0.6, label="DVA")
            ax.legend(loc="upper left")

        txt = f'KIC{hdr["KEPLERID"]}, Q{hdr["QUARTER"]}'
        props = dict(
            boxstyle="square", facecolor="white", alpha=0.95, pad=0.15, linewidth=0
        )
        ax.text(
            0.05,
            0.05,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="k",
            fontsize="small",
            bbox=props,
        )

    fig.tight_layout(h_pad=0.1, w_pad=0.1)

    if n_samples > 5:
        cb = fig.colorbar(_p, ax=axs, shrink=0.6, location="right", pad=0.015)
    else:
        cb = fig.colorbar(_p, ax=axs[-1], shrink=0.6)
    cb.ax.tick_params(labelsize="small")
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.set_label("Days from start", fontsize="small")

    ystr = (
        (r"Raw Centroids, mDVA fits [mas] ") if plotfit else (r"Raw Centroids [mas] ")
    )
    fig.suptitle(ystr, y=1.02)
    savefig(fig, outpath, dpi=400, writepdf=False)
    return


def plot_pca_fit(
    outdir,
    seed=42,
    n_samples=10,
    verbose=1,
    quarter=None,
    sampleid="brightestnonsat100",
    n_components=2,
    dvacorr=False,
    useTrunSVD=False,
    plotfit=True,
    conv2glob=False,
    addsuffix="",
    starlist=None,
    mag=1.0,
    comp_eig=True,
):
    path = os.path.join(outdir, sampleid)
    if not os.path.exists(path):
        os.mkdir(path)

    fitstr = "" if not plotfit else "_wfit"
    qstr = "" if quarter is None else f"_Q{str(quarter).zfill(2)}"
    dvastr = "" if not dvacorr else "_dva"
    svdstr = "" if not useTrunSVD else "_TrunSVD"
    globstr = "" if not conv2glob else "_glob"
    asuf = "" if len(addsuffix) == 0 else f"_{addsuffix}"

    short_outpath = f"seed{seed}_nsamp{n_samples}{qstr}_Ncomp{n_components}{dvastr}{svdstr+globstr}{fitstr}{asuf}.png"
    outpath = os.path.join(path, short_outpath)
    short_outpath_r = f"res_seed{seed}_nsamp{n_samples}{qstr}_Ncomp{n_components}{dvastr}{svdstr+globstr}{asuf}.png"
    outpath_r = os.path.join(path, short_outpath_r)

    # if os.path.exists(outpath_r) and os.path.exists(outpath):
    #     print(f'Found {short_outpath} and {short_outpath_r}, continue.')
    #     return

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    if not os.path.exists(fitsfn):
        raise FileNotFoundError
    with fits.open(fitsfn) as hdu:
        lcpaths = hdu["PATHS"].data["lcpaths"]
        kicids = hdu["PATHS"].data["kicid"]
        times = hdu["TIME"].data["time"]

        tag = "REAL_MOM"
        star_m = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["module"][:, 0]
        star_o = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["output"][:, 0]

    # cl = Clean(
    #     sampleid=sampleid,
    #     verbose=verbose,
    #     conv2glob=conv2glob,
    #     dvacorr=dvacorr,
    #     pca=0,
    #     save_pca_eigvec=0,
    #     remove_earth_points=True,
    #     max_nstars=-1,
    #     seed=seed,
    #     read_fits=True,
    # )
    if starlist is None:
        sel_lcpaths = get_lcpaths_given_sampleid_seed_nsamples_quarter_m_o(
            sampleid, seed, n_samples, fix_quarter=quarter
        )
    else:
        if len(starlist) != n_samples:
            print(
                "Error! Number of panels is not equal to the number of stars in your list!"
            )
            exit()
        sel_lcpaths = get_lcpaths_given_kicids_quarter(quarter=quarter, kicids=starlist)

    ##### add injection if needed
    ctd_x, ctd_y = read_dva_res(
        sampleid, quarter, conv2glob, addsuffix=addsuffix, mag=mag
    )

    # add injection, think about what is a good strategy here
    # if len(addsuffix) > 0:
    #     if conv2glob:
    #         CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}")
    #         dra = np.load(f"{CACHEDIR}/{addsuffix}_dra.npy")
    #         ddec = np.load(f"{CACHEDIR}/{addsuffix}_ddec.npy")

    #         mag = 5e5 if addsuffix == "lg" else 5e6
    #         dra *= mag
    #         ddec *= mag
    #         cl.set_ctd(dra=dra, ddec=ddec)
    #     else:
    #         cl.set_ctd(local_signal_str=addsuffix)

    # maybe we should have directly read from the FITS files

    # cl.get_module_output()
    # star_m, star_o = cl.module[:, 0], cl.output[:, 0]

    # lcpaths = np.array(cl.lcpaths)
    sel_i = [np.argwhere(lcpaths == x)[0][0] for x in sel_lcpaths]
    # ctd_x = cl.ctd_x[sel_i]
    # ctd_y = cl.ctd_y[sel_i]
    ctd_x = ctd_x[sel_i]
    ctd_y = ctd_y[sel_i]

    lcpaths = lcpaths[sel_i]
    star_m = star_m[sel_i]
    star_o = star_o[sel_i]

    if comp_eig:
        basisvecs_x, basisvecs_y = comp_pca_basisvecs(
            sampleid,
            quarter,
            n_components,
            conv2glob,
            sig_only=False,
            addsuffix=addsuffix,
            mag=mag,
        )
    else:
        basisvecs_x, basisvecs_y = get_pca_basisvecs(
            sampleid, dvastr, qstr, n_components, svdstr, globstr, addsuffix=asuf
        )

    x_mean, y_mean = basisvecs_x[0], basisvecs_y[0]
    basisvecs_x = basisvecs_x[1 : 1 + n_components]
    basisvecs_y = basisvecs_y[1 : 1 + n_components]

    plt.close("all")
    set_style()

    # one figure to plot the fits, another to plot the residuals
    if n_samples == 5:
        fig, axs = plt.subplots(ncols=n_samples, figsize=(10 + 0.5, 2))
        flat_axs = axs
        fig_r, axs_r = plt.subplots(ncols=n_samples, figsize=(10 + 0.5, 2))
        flat_axs_r = axs_r

    elif n_samples == 10:
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10 + 0.5, 4))
        flat_axs = axs.flatten()
        fig_r, axs_r = plt.subplots(nrows=2, ncols=5, figsize=(10 + 0.5, 4))
        flat_axs_r = axs_r.flatten()

    else:
        raise NotImplementedError

    cmap = mpl.colormaps["cividis_r"]
    mean_dev = []
    norm_fn = lambda x: x - np.nanmean(x)

    for ix, ax in enumerate(flat_axs):
        # kicid = cl.kicid[sel_i[ix]]
        # time = cl.time[sel_i[ix]]
        kicid = kicids[sel_i[ix]]
        time = times[sel_i[ix]]
        color = time - np.nanmin(time)

        norm_ctd_x, norm_ctd_y = norm_fn(ctd_x[ix]), norm_fn(ctd_y[ix])
        norm_ctd_x_ctr, norm_ctd_y_ctr = norm_ctd_x - x_mean, norm_ctd_y - y_mean

        _p = ax.scatter(
            norm_ctd_x,
            norm_ctd_y,
            c=color,
            s=2,
            marker=".",
            linewidth=0,
            rasterized=True,
            cmap=cmap,
        )

        coef_x = np.einsum(
            "ij,j", np.nan_to_num(basisvecs_x), np.nan_to_num(norm_ctd_x_ctr)
        )
        coef_y = np.einsum(
            "ij,j", np.nan_to_num(basisvecs_y), np.nan_to_num(norm_ctd_y_ctr)
        )
        model_x = np.einsum("i,ij", coef_x, np.nan_to_num(basisvecs_x)) + x_mean
        model_y = np.einsum("i,ij", coef_y, np.nan_to_num(basisvecs_y)) + y_mean

        if plotfit:
            ax.scatter(
                model_x,
                model_y,
                c="tab:red",
                s=2,
                marker=".",
                linewidth=0,
                alpha=0.5,
                rasterized=True,
            )

        res_x = norm_ctd_x - model_x
        res_y = norm_ctd_y - model_y

        flat_axs_r[ix].scatter(
            res_x,
            res_y,
            c=color,
            s=2,
            marker=".",
            linewidth=0,
            alpha=1,
            rasterized=True,
            cmap=cmap,
        )

        mean_dev.append(np.mean(np.sqrt(res_x**2 + res_y**2)))

        props = dict(
            boxstyle="square", facecolor="white", alpha=0.95, pad=0.15, linewidth=0
        )
        for sub_ax in [ax, flat_axs_r[ix]]:
            sub_ax.tick_params(labelsize="small")
            txt = f"KIC{kicid}, Q{quarter}\nm{int(star_m[ix])}-o{int(star_o[ix])}"
            sub_ax.text(
                0.05,
                0.05,
                txt,
                transform=sub_ax.transAxes,
                ha="left",
                va="bottom",
                color="k",
                fontsize="small",
                bbox=props,
            )

        if plotfit and ix == 0:
            ax.scatter([], [], c="tab:red", s=0.6, label="PCA")
            ax.legend(loc="upper left")

    for f in [fig, fig_r]:
        f.tight_layout(h_pad=0.1, w_pad=0.1)

        if n_samples > 5:
            cb = f.colorbar(
                _p,
                ax=axs if f == fig else axs_r,
                shrink=0.6,
                location="right",
                pad=0.015,
            )
        else:
            cb = f.colorbar(_p, ax=(axs if f == fig else axs_r)[-1], shrink=0.6)
        cb.ax.tick_params(labelsize="small")
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.yaxis.set_label_position("right")
        cb.set_label("Days from start", fontsize="small")
    if plotfit:
        ystr = r"Centroids mDVA residue, PCA " + f"(nEig={n_components})" + r" [mas] "
    else:
        ystr = r"Centroids mDVA residue [mas] "
    fig.suptitle(ystr, y=1.02)
    fig.savefig(fig, outpath, dpi=400, writepdf=False)

    ystr = (
        r"Centroids residual"
        + f"(nEig={n_components})"
        + " [mas], median residual {:.3e}".format(np.median(mean_dev))
        + " mas"
    )
    fig_r.suptitle(ystr, y=1.02)

    fig.savefig(fig_r, outpath_r, dpi=400, writepdf=False)
    return


def plot_centroid_samples_timex(
    outpath,
    fitsfn,
    seed=42,
    n_samples=10,
    verbose=1,
    extname="FAKE_MOM_RADEC_MODULE_OUTPUT",
    plots_per_page=5,
):
    """
    Plot x and y centroid versus time for `n_samples` time-series.  These are
    currently written to pull from the "100 brightest non-saturated star"
    sample.

    kwargs:

        n_samples: number of stars to show. if -1, will plot all stars in fits file.

        verbose: whether to print words to STDOUT

        ykey: can be "MOM_CENTR" or "PSF_CENTR". PSF_CENTR are the better
        PSF-fitted centroids, but they are only available for PPA targets.
        MOM_CENTR are the flux-weighted (moment-derived) centroids, available
        for almost every target.

        fix_quarter: None or int.  Integer between 0 and 17 denotes quarter
        number.
    """

    with fits.open(fitsfn) as hdu:
        time = hdu["TIME"].data["time"][0]  # in days, (Ntimes)
        kicids = hdu["PATHS"].data["kicid"]
        ra = hdu[extname].data["CENTROID_X"]  # in deg, (N_output, Ntimes)
        dec = hdu[extname].data["CENTROID_Y"]  # in deg, (N_output, Ntimes)
        quarter = hdu[0].header["quarter"]

    # make plot
    plt.close("all")

    if n_samples == -1:
        n_samples = len(kicids)

    with PdfPages(outpath) as pdf:

        for i in range(n_samples):
            if i % plots_per_page == 0:
                fig, axs = plt.subplots(nrows=plots_per_page, figsize=(8.5, 11))

            ax = axs[i % plots_per_page]

            t0 = np.nanmin(time)
            ctd_x0 = np.nanmean(ra[i])
            ctd_y0 = np.nanmean(dec[i])
            ax.scatter(
                time - t0,
                ra[i] - ctd_x0,
                c="C0",
                s=1,
                zorder=2,
                linewidths=0,
                marker=".",
            )
            ax.scatter(
                time - t0,
                dec[i] - ctd_y0,
                c="orange",
                s=1,
                zorder=2,
                linewidths=0,
                marker=".",
            )

            if ((i + 1) % plots_per_page) == 0:
                ax.set_xticklabels([])

            if (i % plots_per_page) == 0:
                ax.scatter([], [], c="C0", label="raw x", s=1)
                ax.scatter([], [], c="orange", label="raw y", s=1)
                ax.legend(loc="upper right", fontsize=6)

            txt = f"KIC{kicids[i]}, Q{quarter}"
            props = dict(
                boxstyle="square",
                facecolor="white",
                alpha=0.95,
                pad=0.15,
                linewidth=0,
            )
            ax.text(
                0.05,
                0.05,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color="k",
                fontsize="small",
                bbox=props,
            )

            # Save the current figure to the PdfPages object if it's full
            if (((i + 1) % plots_per_page) == 0) or ((i + 1) == n_samples):
                ax.set_xlabel("Days from start")
                fig.text(-0.01, 0.5, "Centroid [px]", va="center", rotation=90)
                pdf.savefig(fig, bbox_inches="tight")

                # if (i + 1) == n_samples:
                #     # set file metadata
                #     d = pdf.infodict()
                #     d["Title"] = "Centroids vs Time"
                #     d["Author"] = "K. Pardo"
                #     d["CreationDate"] = datetime.datetime.today()


def plot_pca_eig(
    outdir,
    quarter=None,
    sampleid="brightestnonsat100",
    dvacorr=False,
    useTrunSVD=False,
    n_comp=3,
    conv2glob=False,
    addsuffix="",
    mag=1.0,
    comp_eig=True,  # compute eigenvectors on the fly, used for injection signals
    sig_only=False,
):
    qstr = str(quarter).zfill(2)
    svdstr = "" if not useTrunSVD else "_TrunSVD"
    dvastr = "" if not dvacorr else "_dva"
    globstr = "" if not conv2glob else "_glob"
    asuf = "" if len(addsuffix) == 0 else f"_{addsuffix}"

    pcadir = os.path.join(outdir, sampleid)
    if not os.path.exists(pcadir):
        os.makedirs(pcadir)
    outpath = os.path.join(
        pcadir, f"top{n_comp}_Q{qstr}{dvastr}{svdstr}{globstr}{asuf}.png"
    )

    # if os.path.isfile(outpath):
    #     return

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    try:
        hdu = fits.open(fitsfn)
        time = hdu["TIME"].data["time"][0]
        hdu.close()
    except:
        raise FileNotFoundError

    if comp_eig:
        eigenvecs_x, eigenvecs_y = comp_pca_basisvecs(
            sampleid,
            quarter,
            n_comp,
            conv2glob,
            sig_only=sig_only,
            addsuffix=addsuffix,
            mag=mag,
        )
    else:
        eigenvecs_x, eigenvecs_y = get_pca_basisvecs(
            sampleid, dvastr, "_Q" + qstr, n_comp, svdstr, globstr, addsuffix=asuf
        )

    plt.close("all")
    # set_style()

    ##### load signal only displacement ######
    ## FIXME: add an option to do this from above
    # if not addsuffix == "":
    #     CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}")
    #     if conv2glob:
    #         with fits.open(fitsfn) as hdu:
    #             star_ra = hdu["PATHS"].data["survey_ra"]
    #             star_dec = hdu["PATHS"].data["survey_dec"]

    #         if mag is None:
    #             mag = 5e5 if addsuffix == "lg" else 5e6

    #         asf = addsuffix.split("_")[0]

    #         dra = np.load(f"{CACHEDIR}/{asf}_dra.npy") * 180 / np.pi * mag
    #         ddec = np.load(f"{CACHEDIR}/{asf}_ddec.npy") * 180 / np.pi * mag

    #         ra = star_ra[:, np.newaxis] + dra
    #         dec = star_dec[:, np.newaxis] + ddec

    #         mean_ra = np.nanmean(ra, axis=1)[:, np.newaxis]
    #         mean_dec = np.nanmean(dec, axis=1)[:, np.newaxis]

    #         deg2rad = np.pi / 180

    #         sig_dx = (ra - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad
    #         sig_dy = (dec - mean_dec) * deg2rad
    #         mean_sig_dx = np.mean(sig_dx, axis=0) * multiplier
    #         mean_sig_dy = np.mean(sig_dy, axis=0) * multiplier

    #     else:
    #         sig_dx = np.load(f"{CACHEDIR}/{addsuffix}_x_sig.npy")
    #         sig_dy = np.load(f"{CACHEDIR}/{addsuffix}_y_sig.npy")

    #         with fits.open(fitsfn) as hdu:
    #             dvax = hdu["DVA"].data["dva_x"]
    #             dvay = hdu["DVA"].data["dva_y"]

    #         sig_dx -= dvax
    #         sig_dy -= dvay

    #         sig_dx -= np.mean(sig_dx, axis=1)[:, np.newaxis]
    #         sig_dy -= np.mean(sig_dy, axis=1)[:, np.newaxis]

    #         mean_sig_dx = np.mean(sig_dx, axis=0) * multiplier
    #         mean_sig_dy = np.mean(sig_dy, axis=0) * multiplier

    fig, axs = plt.subplots(nrows=n_comp + 1, figsize=(5, 1.5 * (n_comp + 1)))

    for ix, ax in enumerate(axs):
        eigvec_x = eigenvecs_x[ix, :]
        eigvec_y = eigenvecs_y[ix, :]

        assert len(time) == len(eigvec_x) == len(eigvec_y)

        norm_min = lambda x: (x - np.nanmin(x))

        ax.scatter(
            norm_min(time), eigvec_x, c="C0", s=1, zorder=2, linewidths=0, marker="."
        )
        ax.scatter(
            norm_min(time), eigvec_y, c="C1", s=1, zorder=2, linewidths=0, marker="."
        )

        # if not addsuffix == "":

        # if ix == 0:
        #     ax.scatter(
        #         norm_min(time),
        #         mean_sig_dx,
        #         c="tab:green",
        #         s=1,
        #         linewidths=0,
        #         marker=".",
        #     )
        #     ax.scatter(
        #     norm_min(time),
        #     mean_sig_dy,
        #     c="tab:red",
        #     s=1,
        #     linewidths=0,
        #     marker=".",
        # )

        if ix != n_comp - 1:
            ax.set_xticklabels([])

        if ix == 0:
            ax.scatter([], [], c="C0", s=1, label="x")
            ax.scatter([], [], c="C1", s=1, label="y")
            ax.legend(loc="upper right", fontsize=8)

        txt = f"EIGVEC{ix}" if ix != 0 else "MEAN"
        props = dict(
            boxstyle="square", facecolor="white", alpha=0.95, pad=0.15, linewidth=0
        )
        ax.text(
            0.05,
            0.05,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="k",
            fontsize="xx-small",
            bbox=props,
        )

    ax.set_xlabel("Days from start")
    fig.text(-0.01, 0.5, "Principal Component Value [px]", va="center", rotation=90)
    # savefig(fig, outpath, dpi=400, writepdf=False)
    fig.savefig(outpath, dpi=400)

    return


def plot_fov(data, outdir):
    """
    plots a single FOV
    """
    set_style()
    fig, ax = plt.subplots(nrows=1, figsize=(5, 5))
    ax.scatter(data, color="k")
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    savefig(fig, outpath, dpi=400, writepdf=False)
    return


def make_fov_movie(data, outpath, fps, show=False, save=False):
    """
    makes a movie of fovs

    data should have axes (stars, coord, time)
    """
    set_style()
    fig, ax = plt.subplots(nrows=1, figsize=(5, 5))
    ax.set_xlabel("l [deg]")
    ax.set_ylabel("b [deg]")
    sca = ax.scatter(
        data[:, 0, 0], data[:, 1, 0], color="k", s=1
    )  ## set initial frame to show

    def animate(i):
        sca.set_offsets(data[:, :, i])
        return (sca,)

    if save:
        ani = animation.FuncAnimation(fig, animate, blit=True)
        ani.save(outpath, dpi=400, fps=fps)
    if show:
        ani = animation.FuncAnimation(fig, animate, blit=True, repeat=True, fps=fps)
        plt.show()


def plot_chip_sky_frame(
    outpath,
    plot_chip=1,
    animate_chip=1,
    plot_sky=1,
    animate_sky=1,
    sampleid="brightestnonsat100_rot",
    quarter=12,
    seed=42,
    n_samples=100,
    iplot=100,
    target_m=[],
    target_o=[],
    plot_arrow=1,
    addsuffix="",
):

    path = os.path.join(outpath, sampleid)
    if not os.path.exists(path):
        os.mkdir(path)
    qstr = "" if quarter is None else f"_Q{str(quarter).zfill(2)}"
    asuf = "" if len(addsuffix) == 0 else f"_{addsuffix}"

    NCOL, NROW = rm.get_parameters("nColsImaging"), rm.get_parameters("nRowsImaging")
    rdp = raDec2Pix.raDec2PixClass()

    cl = Clean(
        sampleid=sampleid,
        verbose=0,
        conv2glob=0,
        dvacorr=1,
        pca=0,
        save_pca_eigvec=0,
        remove_earth_points=1,
        max_nstars=n_samples,
        seed=seed,
        quarter=quarter,
        target_m=target_m,
        target_o=target_o,
    )

    if len(addsuffix) > 0:
        cl.set_ctd(local_signal_str="ex_" + addsuffix)

    #######

    # BJD = BKJD + 2454833
    bjdrefi = 2454833
    time_bjd = cl.time[0, 0] + bjdrefi
    init_mjds = Time(time_bjd, format="jd", scale="tdb")

    # ctdx, ctdy = cl.ctd_x, cl.ctd_y
    mean_rctdx, mean_rctdy = np.nanmean(cl.rctd_x, axis=1), np.nanmean(
        cl.rctd_y, axis=1
    )
    nstars, ntimes = cl.rctd_x.shape
    nrow_per_frame = 40
    nframes = ntimes // nrow_per_frame
    print("nframes: ", nframes, flush=1)
    interval = 1

    cl.get_module_output()
    star_m, star_o = cl.module[:, 0], cl.output[:, 0]

    def plot_chip_frame(outpath):

        def create_grid(ax):
            ax.vlines(np.arange(0, 11, 2), ymin=0, ymax=10, color="k")
            ax.hlines(np.arange(0, 11, 2), xmin=0, xmax=10, color="k")

            ax.hlines(
                np.arange(1, 11, 2),
                xmin=0,
                xmax=10,
                color="lightgray",
                linestyle="dashed",
            )
            ax.vlines(
                np.arange(1, 11, 2),
                ymin=0,
                ymax=10,
                color="lightgray",
                linestyle="dashed",
            )

        def draw_single_cell(ax, module, output):

            if module in [18, 19, 22, 23, 24]:
                output_map = {1: (1, 1), 2: (0, 1), 3: (0, 0), 4: (1, 0)}
            elif module in [6, 11, 12, 13, 16, 17]:
                output_map = {1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (0, 0)}
            elif module in [9, 10, 14, 15, 20]:
                output_map = {1: (0, 1), 2: (0, 0), 3: (1.0, 0), 4: (1, 1)}
            else:
                output_map = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

            lower_y = (4 - (module - 1) // 5) * 2
            lower_x = 2 * ((module - 1) % 5)
            if len(output) == 1:
                dx, dy = output_map[output]
                lower_x += dx
                lower_y += dy
                d = 2
                ax.plot(
                    [lower_x, lower_x, lower_x + d, lower_x + d, lower_x],
                    [lower_y, lower_y + d, lower_y + d, lower_y, lower_y],
                    color="k",
                )
            else:
                d = 1
                ax.plot(
                    [lower_x, lower_x, lower_x + 2 * d, lower_x + 2 * d, lower_x],
                    [lower_y, lower_y + 2 * d, lower_y + 2 * d, lower_y, lower_y],
                    color="k",
                )
                ax.plot(
                    [lower_x, lower_x + 2 * d],
                    [lower_y + d, lower_y + d],
                    color="lightgray",
                    linestyle="dashed",
                )
                ax.plot(
                    [lower_x + d, lower_x + d],
                    [lower_y, lower_y + 2 * d],
                    color="lightgray",
                    linestyle="dashed",
                )
            return

        def highlight_cell(ax, module, output, facecolor="r", alpha=0.2):

            import matplotlib.patches as patches

            if module in [18, 19, 22, 23, 24]:
                output_map = {1: (1, 1), 2: (0, 1), 3: (0, 0), 4: (1, 0)}
            elif module in [6, 11, 12, 13, 16, 17]:
                output_map = {1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (0, 0)}
            elif module in [9, 10, 14, 15, 20]:
                output_map = {1: (0, 1), 2: (0, 0), 3: (1.0, 0), 4: (1, 1)}
            else:
                output_map = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

            lower_y = (4 - (module - 1) // 5) * 2
            lower_x = 2 * ((module - 1) % 5)
            dx, dy = output_map[output]
            ax.add_patch(
                patches.Rectangle(
                    (lower_x + dx, lower_y + dy), 1, 1, facecolor=facecolor, alpha=alpha
                )
            )
            return

        def draw_arrow(
            ax,
            module,
            output,
            startx,
            starty,
            endx=0.0,
            endy=0.0,
            use_pix_scale=1,
            arrow=1,
            scaled=None,
            col="tab:blue",
            oh=0.5,
            hw=0.2,
            hl=0.2,
        ):  # endx in coordinates

            if use_pix_scale:
                plate_ncols, plate_nrows = rm.get_parameters(
                    "nColsImaging"
                ), rm.get_parameters("nRowsImaging")
            else:
                plate_ncols = plate_nrows = 1

            startx /= plate_nrows
            starty /= plate_ncols

            endx /= plate_nrows
            endy /= plate_ncols

            if module in [18, 19, 22, 23, 24]:
                output_map = {1: (1, 1), 2: (0, 1), 3: (0, 0), 4: (1, 0)}
            elif module in [6, 11, 12, 13, 16, 17]:
                output_map = {1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (0, 0)}
            elif module in [9, 10, 14, 15, 20]:
                output_map = {1: (0, 1), 2: (0, 0), 3: (1.0, 0), 4: (1, 1)}
            else:
                output_map = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

            lower_y = (4 - (module - 1) // 5) * 2
            lower_x = 2 * ((module - 1) % 5)
            dx, dy = output_map[output]

            row_pair = {1: 4, 4: 1, 2: 3, 3: 2}
            col_pair = {1: 2, 2: 1, 3: 4, 4: 3}

            row_vec = [
                output_map[row_pair[output]][0] - output_map[output][0],
                output_map[row_pair[output]][1] - output_map[output][1],
            ]
            col_vec = [
                output_map[col_pair[output]][0] - output_map[output][0],
                output_map[col_pair[output]][1] - output_map[output][1],
            ]

            _rv = [x < 0 for x in row_vec]
            _cv = [x < 0 for x in col_vec]
            o_x = lower_x + dx + _rv[0] + _cv[0]  # coordinate of the origin
            o_y = lower_y + dy + _rv[1] + _cv[1]  # coordinate of the origin

            x_start, y_start = (
                o_x + row_vec[0] * startx + col_vec[0] * starty,
                o_y + row_vec[1] * startx + col_vec[1] * starty,
            )
            x_end, y_end = (
                o_x + row_vec[0] * endx + col_vec[0] * endy,
                o_y + row_vec[1] * endx + col_vec[1] * endy,
            )
            dx = x_end - x_start
            dy = y_end - y_start

            if not arrow:
                scatter_scale = 5000
                return ax.scatter(
                    x_start + dx * scatter_scale,
                    y_start + dy * scatter_scale,
                    color=col,
                    s=4,
                    linewidth=0,
                    zorder=4,
                )

            scale = scaled / np.sqrt(dx**2 + dy**2) if scaled is not None else 1
            arr = ax.arrow(
                x_start,
                y_start,
                dx * scale,
                dy * scale,
                head_width=hw,
                head_length=hl,
                overhang=oh,
                color=col,
            )
            return arr

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis("off")

        if len(target_m) == 1:
            draw_single_cell(ax, target_m, target_o)
        else:
            create_grid(ax)

        arr_ls = []

        def update_chip(i):
            nonlocal arr_ls

            iplot = np.min([i * nrow_per_frame, ntimes - 1])
            for arr in arr_ls:
                arr.remove()
            arr_ls.clear()

            for i, (m_, o_) in enumerate(zip(star_m, star_o)):

                if plot_arrow:
                    cols = [f"ctdx_{i}", f"ctdy_{i}", f"ctdx_{i}_fut", f"ctdy_{i}_fut"]
                    xstart, ystart, xend, yend = rctd_df.iloc[iplot][cols]
                    if (
                        np.isnan(xstart)
                        or np.isnan(xend)
                        or np.isnan(ystart)
                        or np.isnan(yend)
                    ):
                        continue
                else:
                    xstart, ystart = mean_rctdx[i], mean_rctdy[i]
                    xend, yend = (
                        rctd_df[f"ctdx_{i}"].iloc[iplot],
                        rctd_df[f"ctdy_{i}"].iloc[iplot],
                    )
                arr = draw_arrow(
                    ax,
                    m_,
                    o_,
                    xstart,
                    ystart,
                    endx=xend,
                    endy=yend,
                    arrow=plot_arrow,
                    scaled=0.5,
                    hw=0.1,
                    hl=0.1,
                    col="tab:blue",
                )
                arr_ls.append(arr)

            ndays = cl.time[0, iplot] - cl.time[0, 0]
            arr = ax.text(
                0.5,
                1.0,
                "{:.1f} ndays since start of quarter".format(ndays),
                va="bottom",
                fontsize=10,
                transform=ax.transAxes,
                ha="center",
            )
            arr_ls.append(arr)
            return arr_ls

        update_chip(10)
        arrstr = "_arrow" if plot_arrow else ""
        fig.savefig(f"{path}/{qstr}{arrstr}{asuf}_chip_frame.png", bbox_inches="tight")

        if animate_chip:
            ani = animation.FuncAnimation(
                fig, update_chip, frames=nframes, interval=150, blit=True
            )
            ani.save(f"{path}/{qstr}{arrstr}{asuf}_chip_frame.mp4")
        return

    if plot_chip:
        lbs = [f"ctdx_{i}" for i in range(nstars)] + [
            f"ctdy_{i}" for i in range(nstars)
        ]

        try:
            # use this if we read from FITS file, since FITS data is stored in big-endian, but pandas only accepts little-endian
            rctd_df = pd.DataFrame(
                np.concatenate(
                    [
                        cl.rctd_x.byteswap().newbyteorder(),
                        cl.rctd_y.byteswap().newbyteorder(),
                    ],
                    axis=0,
                ).T,
                columns=lbs,
            )
            rctd_df["time"] = cl.time[0].byteswap().newbyteorder()
            rctd_df["future_time"] = cl.time[0].byteswap().newbyteorder() - interval
        except:
            # this works if not read from FITS file
            rctd_df = pd.DataFrame(
                np.concatenate([cl.rctd_x, cl.rctd_y], axis=0).T, columns=lbs
            )
            rctd_df["time"] = cl.time[0]
            rctd_df["future_time"] = cl.time[0] - interval

        rctd_df = pd.merge_asof(
            rctd_df.drop(columns=["future_time"]),
            rctd_df.drop(columns=["time"]),
            left_on="time",
            right_on="future_time",
            direction="forward",
            suffixes=("", "_fut"),
            tolerance=0.5,
        )

        plot_chip_frame(outpath)

    def plot_sky_frame(outpath):

        all_module = np.array(
            [i for i in range(2, 5)]
            + [i for i in range(6, 21)]
            + [i for i in range(22, 25)]
        )
        all_output = np.array([i for i in range(1, 5)])
        all_module, all_output = np.meshgrid(all_module, all_output)
        all_module = all_module.flatten()
        all_output = all_output.flatten()
        all_rals, all_decls = rdp.pix_2_ra_dec(
            all_module,
            all_output,
            np.zeros_like(all_module),
            np.zeros_like(all_module),
            init_mjds.mjd,
            aberrateFlag=True,
        )

        all_xmax_rals, all_xmax_decls = rdp.pix_2_ra_dec(
            all_module,
            all_output,
            NROW * np.ones_like(all_module),
            np.zeros_like(all_module),
            init_mjds.mjd,
            aberrateFlag=True,
        )

        all_ymax_rals, all_ymax_decls = rdp.pix_2_ra_dec(
            all_module,
            all_output,
            np.zeros_like(all_module),
            NCOL * np.ones_like(all_module),
            init_mjds.mjd,
            aberrateFlag=True,
        )

        all_diag_rals, all_diag_decls = rdp.pix_2_ra_dec(
            all_module,
            all_output,
            NROW * np.ones_like(all_module),
            NCOL * np.ones_like(all_module),
            init_mjds.mjd,
            aberrateFlag=True,
        )

        def draw_chip_patch_onsky(ax):
            for i in range(len(all_rals)):
                ax.plot(
                    [all_rals[i], all_xmax_rals[i]],
                    [all_decls[i], all_xmax_decls[i]],
                    color="lightgray",
                )
                ax.plot(
                    [all_rals[i], all_ymax_rals[i]],
                    [all_decls[i], all_ymax_decls[i]],
                    color="lightgray",
                )
                ax.plot(
                    [all_xmax_rals[i], all_diag_rals[i]],
                    [all_xmax_decls[i], all_diag_decls[i]],
                    color="lightgray",
                )
                ax.plot(
                    [all_ymax_rals[i], all_diag_rals[i]],
                    [all_ymax_decls[i], all_diag_decls[i]],
                    color="lightgray",
                )
            return

        def draw_arrow_raDec(
            ax, ra_s, dec_s, ra_e=None, dec_e=None, scale=None, col="tab:blue", arrow=1
        ):  # endx in coordinates
            if not ra_e:
                ax.scatter(ra_s, dec_s, color=col, s=6, zorder=4)
                return

            dra = ra_e - ra_s
            ddec = dec_e - dec_s
            if not arrow:
                scale = 50000
                return ax.scatter(
                    ra_s + dra * scale,
                    dec_s + ddec * scale,
                    s=14,
                    linewidths=0,
                    marker=".",
                    color=col,
                    zorder=4,
                )

            scale = scale / np.sqrt(dra**2 + ddec**2) if scale is not None else 1
            return ax.arrow(
                ra_s,
                dec_s,
                dra * scale,
                ddec * scale,
                head_width=0.2,
                head_length=0.2,
                color=col,
            )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        draw_chip_patch_onsky(ax)
        arr_ls = []

        def update_sky(i):
            nonlocal arr_ls
            iplot = i * nrow_per_frame

            for arr in arr_ls:
                arr.remove()
            arr_ls.clear()

            for i in range(len(star_ra)):

                if plot_arrow:
                    cols = [f"ra_{i}", f"dec_{i}", f"ra_{i}_fut", f"dec_{i}_fut"]
                    ra_s, dec_s, ra_e, dec_e = radec_df.iloc[iplot][cols]
                    if (
                        np.isnan(ra_s)
                        or np.isnan(dec_s)
                        or np.isnan(ra_e)
                        or np.isnan(dec_e)
                    ):
                        continue
                else:
                    ra_s, dec_s = mean_ra[i], mean_dec[i]
                    ra_e, dec_e = (
                        radec_df[f"ra_{i}"].iloc[iplot],
                        radec_df[f"dec_{i}"].iloc[iplot],
                    )

                arr = draw_arrow_raDec(
                    ax,
                    ra_s,
                    dec_s,
                    ra_e,
                    dec_e,
                    scale=0.8,
                    col="tab:blue",
                    arrow=plot_arrow,
                )
                arr_ls.append(arr)

            ndays = cl.time[0, iplot] - cl.time[0, 0]
            arr = ax.text(
                0.5,
                1.0,
                "{:.1f} ndays since start of quarter".format(ndays),
                va="bottom",
                fontsize=10,
                transform=ax.transAxes,
                ha="center",
            )
            arr_ls.append(arr)

            return arr_ls

        update_sky(10)
        arrstr = "_arrow" if plot_arrow else ""
        fig.savefig(f"{path}/{qstr}{arrstr}{asuf}_sky_frame.png", bbox_inches="tight")

        if animate_sky:
            ani = animation.FuncAnimation(
                fig, update_sky, frames=nframes, interval=150, blit=True
            )
            ani.save(f"{path}/{qstr}{arrstr}{asuf}_sky_frame.mp4")
        return

    if plot_sky:
        cl.conv2glob = 1
        if len(addsuffix) > 0:
            CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}")
            dra = np.load(f"{CACHEDIR}/{addsuffix}_dra.npy")
            ddec = np.load(f"{CACHEDIR}/{addsuffix}_ddec.npy")
            mag = 5e5 if addsuffix == "lg" else 5e6
            dra *= mag
            ddec *= mag
            cl.set_ctd(dra=dra, ddec=ddec)
        else:
            cl.set_ctd()

        star_ra, star_dec = cl.ra, cl.dec
        mean_ra, mean_dec = np.nanmean(star_ra, axis=1), np.nanmean(star_dec, axis=1)

        lbs = [f"ra_{i}" for i in range(nstars)] + [f"dec_{i}" for i in range(nstars)]
        try:
            # use this if we read from FITS file, since FITS data is stored in big-endian, but pandas only accepts little-endian

            radec_df = pd.DataFrame(
                np.concatenate(
                    [
                        star_ra.byteswap().newbyteorder(),
                        star_dec.byteswap().newbyteorder(),
                    ],
                    axis=0,
                ).T,
                columns=lbs,
            )
            radec_df["time"] = cl.time[0].byteswap().newbyteorder()
            radec_df["future_time"] = cl.time[0].byteswap().newbyteorder() - interval
        except:
            # this works if not read from FITS file
            radec_df = pd.DataFrame(
                np.concatenate([star_ra, star_dec], axis=0).T, columns=lbs
            )
            radec_df["time"] = cl.time[0]
            radec_df["future_time"] = cl.time[0] - interval

        radec_df = pd.merge_asof(
            radec_df.drop(columns=["future_time"]),
            radec_df.drop(columns=["time"]),
            left_on="time",
            right_on="future_time",
            direction="forward",
            suffixes=("", "_fut"),
            tolerance=0.5,
        )
        plot_sky_frame(outpath)
    return
