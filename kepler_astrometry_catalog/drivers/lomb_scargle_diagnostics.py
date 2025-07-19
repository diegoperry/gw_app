"""
Does various tests using the Lomb Scargle periodogram

call by specifiying quarter in command line. e.g.,

python lomb_scargle_diagnostics.py 12

for quarter 12
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.timeseries import LombScargle
import os
import sys
import argparse
from astropy.io import fits
from astropy.coordinates import SkyCoord
from dotenv import find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
sys.path.append(os.path.dirname(find_dotenv()) + "/estoiles/")
sys.path.append(os.path.dirname(find_dotenv()) + "/Kepler-RaDex2Pix/")


from kepler_astrometry_catalog.constants import PIX_SCALE_MAS
import kepler_astrometry_catalog.clean as c
import kepler_astrometry_catalog.get_star_catalog as gsc
from kepler_astrometry_catalog.paths import RESULTSDIR
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description="Lomb Scargle diagnostics")
parser.add_argument("sampleid", type=str, help="sample id")
parser.add_argument("outpath", type=str, help="outdir for plots")
parser.add_argument("fitsfn", type=str, help="fits path")
parser.add_argument("--n_stars", type=int, default=-1, help="Number of samples")
parser.add_argument(
    "--extname", type=str, default="RADEC", help="Header string for FITS extension"
)
parser.add_argument("--quarter", type=int, default=12, help="quarter number")
parser.add_argument("--distanceave", type=str, default="0", help="Average by distance")
parser.add_argument(
    "--localposcorr", type=str, default="0", help="Use local minus poscorr"
)

args = parser.parse_args()

SAMPLEID = args.sampleid
QTR = args.quarter
fitsfn = args.fitsfn
PLOTDIR = args.outpath
NSTARS = args.n_stars
extname = args.extname
localposcorr = args.localposcorr.lower() in ["true", "1", "t", "y", "yes"]
distance_ave = args.distanceave.lower() in ["true", "1", "t", "y", "yes"]

pix_to_deg = PIX_SCALE_MAS * (1.0 * u.mas).to(u.deg)


def main():
    print(f"Starting analysis for quarter {QTR} w/ sampleid: {SAMPLEID}")
    with fits.open(fitsfn) as hdu:
        time = hdu["TIME"].data["time"][0]  # in days, (Ntimes)
        kicids = hdu["PATHS"].data["kicid"]
        if localposcorr:
            x = hdu["RAW_CENTROIDS"].data["RAW_CENTROID_X"]
            y = hdu["RAW_CENTROIDS"].data["RAW_CENTROID_Y"]
            posx = hdu["POSCORR"].data["CORR_X"]
            posy = hdu["POSCORR"].data["CORR_Y"]
            median_time_index = np.abs(time - np.median(time)).argmin()
            ra = ((x - x[:, median_time_index][:, np.newaxis]) - posx) * PIX_SCALE_MAS
            dec = ((y - y[:, median_time_index][:, np.newaxis]) - posy) * PIX_SCALE_MAS
        if distance_ave:
            x = hdu["RAW_CENTROIDS"].data["RAW_CENTROID_X"]
            y = hdu["RAW_CENTROIDS"].data["RAW_CENTROID_Y"]
            posx = hdu["POSCORR"].data["CORR_X"]
            posy = hdu["POSCORR"].data["CORR_Y"]
            median_time_index = np.abs(time - np.median(time)).argmin()
            deltax = (
                (x - x[:, median_time_index][:, np.newaxis]) - posx
            ) * PIX_SCALE_MAS
            deltay = (
                (y - y[:, median_time_index][:, np.newaxis]) - posy
            ) * PIX_SCALE_MAS
            ra = hdu["PATHS"].data["survey_ra"]
            dec = hdu["PATHS"].data["survey_dec"]
        elif "RAW" in extname:
            x = hdu[extname].data["RAW_CENTROID_X"]
            y = hdu[extname].data["RAW_CENTROID_Y"]
            median_time_index = np.abs(time - np.median(time)).argmin()
            ra = ((x - x[:, median_time_index][:, np.newaxis])) * PIX_SCALE_MAS
            dec = ((y - y[:, median_time_index][:, np.newaxis])) * PIX_SCALE_MAS
        else:
            ra = hdu[extname].data["CENTROID_X"]  # in deg, (N_output, Ntimes)
            dec = hdu[extname].data["CENTROID_Y"]  # in deg, (N_output, Ntimes)
    global NSTARS
    if NSTARS == -1:
        NSTARS = len(kicids)
    if distance_ave:
        average_by_distance(time, deltax, deltay, ra, dec, kicids)
    else:
        plot_average_centroids(time, ra, dec)
        plot_ls(time, ra, dec)
        get_best_fit(time, ra, dec, plot=True)


def average_each_time(centroids):
    ## axis0 = star, axis1 = time
    ## this assumes axes are aligned.
    ## which is true for global coords.
    return np.average(centroids, axis=0)


def std_each_time(centroids):
    ## axis0 = star, axis1 = time
    ## this assumes axes are aligned.
    ## which is true for global coords.
    return np.std(centroids, axis=0)


def plot_average_centroids(t, x, y):
    f = plt.figure()
    plt.scatter(t - t[0], average_each_time(x), label="x")
    plt.scatter(t - t[0], average_each_time(y), label="y")
    plt.legend()
    plt.xlabel("Time [days]")
    plt.ylabel("Averaged Centroid Shifts [mpix]")
    f.tight_layout()
    f.savefig(
        PLOTDIR + f"average_centroids_{SAMPLEID}_Q{str(QTR).zfill(2)}_NS{NSTARS}.pdf"
    )
    # Save t, x, y to a CSV file
    data = np.column_stack(
        (
            t,
            average_each_time(x),
            average_each_time(y),
            std_each_time(x),
            std_each_time(y),
        )
    )
    header = "Time [days], Average X Centroid Shifts [mas], Average Y Centroid Shifts [mas], Std X Centroid Shifts [mas], Std Y Centroid Shifts [mas]"
    np.savetxt(
        PLOTDIR
        + f"data_average_centroids_{SAMPLEID}_Q{str(QTR).zfill(2)}_NS{NSTARS}.csv",
        data,
        delimiter=",",
        header=header,
    )


def plot_ls(t, x, y, label=None):
    f_x, power_x = LombScargle(
        (t - t[0]) * u.day, average_each_time(x), std_each_time(x)
    ).autopower()
    f_y, power_y = LombScargle(
        (t - t[0]) * u.day, average_each_time(y), std_each_time(y)
    ).autopower()
    f = plt.figure()
    plt.loglog(f_x.to(u.Hz), power_x, label="x")

    plt.loglog(f_y.to(u.Hz), power_y, label="y")
    plt.legend(loc="upper right")
    plt.axvline(1.7e-7, color="k", ls="dashed")
    plt.fill_between([1.59e-7, 1.81e-7], [100, 100], [0, 0], alpha=0.3, color="k")
    plt.ylim([1.0e-10, 10])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    if label is None:
        plt.text(1.0e-8, 1.0e-9, f"Q{QTR}")
    else:
        plt.text(1.0e-8, 1.0e-9, f"{label}")
    f.savefig(
        PLOTDIR + f"power_spectrum_{SAMPLEID}_Q{str(QTR).zfill(2)}_NS{NSTARS}.pdf"
    )


def get_best_fit(t, x, y, freq=1.71e-7 * u.Hz, fap=True, chisq=True, plot=False):
    lsx = LombScargle((t - t[0]) * u.day, average_each_time(x), std_each_time(x))
    x_fit = lsx.model((t - t[0]) * u.day, freq)

    lsy = LombScargle((t - t[0]) * u.day, average_each_time(y), std_each_time(y))
    y_fit = lsy.model((t - t[0]) * u.day, freq)

    if fap:
        xfap = lsx.false_alarm_probability(lsx.power(freq))
        yfap = lsy.false_alarm_probability(lsy.power(freq))
        print(f"False Alarm Prob. X: {xfap.value}")
        print(f"False Alarm Prob. Y: {yfap.value}")

    if chisq:
        xterms = (x - x_fit) ** 2 / std_each_time(x) ** 2
        yterms = (y - y_fit) ** 2 / std_each_time(y) ** 2
        chisq = -0.5 * (np.sum(xterms) + np.sum(yterms))
        print(f"Chisq: {chisq.value}")

    if plot:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].errorbar(
            t - t[0],
            average_each_time(x),
            yerr=std_each_time(x),
            fmt="o",
            markersize=1,
            elinewidth=0.5,
            zorder=3,
        )
        ax[0].plot(t - t[0], x_fit, zorder=10)
        ax[0].set_xlabel(r"Time [days]")
        ax[0].set_ylabel(r"Averaged X Centroid Shifts [mpix]")
        ax[0].text(
            0.1,
            0.1,
            f"$\chi^2 =$ {(-0.5*np.sum(xterms)).value:.0f}",
            transform=ax[0].transAxes,
        )

        ax[1].errorbar(
            t - t[0],
            average_each_time(y),
            yerr=std_each_time(y),
            fmt="o",
            markersize=1,
            elinewidth=0.5,
            zorder=3,
        )
        ax[1].plot(t - t[0], y_fit, zorder=10)
        ax[1].set_xlabel(r"Time [days]")
        ax[1].set_ylabel(r"Averaged Y Centroid Shifts [mpix]")
        ax[1].text(
            0.1,
            0.1,
            f"$\chi^2 =$ {(-0.5*np.sum(yterms)).value:.0f}",
            transform=ax[1].transAxes,
        )
        f.tight_layout()
        f.savefig(
            PLOTDIR + f"ls_best_fit_{SAMPLEID}_Q{str(QTR).zfill(2)}_NS{NSTARS}.pdf"
        )

    return x_fit, y_fit


def average_by_distance(time, deltax, deltay, ra, dec, kicids, ncluster=10):
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # Combine coordinates into a feature matrix
    features = np.column_stack((ra, dec))

    # Perform clustering using K-means algorithm
    kmeans = KMeans(n_clusters=ncluster)  # Specify the number of clusters
    kmeans.fit(features)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Plot the clusters
    for i in range(len(np.unique(labels))):
        cluster_points = coords[labels == i]
        plt.scatter(cluster_points.ra, cluster_points.dec)

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.savefig(PLOTDIR + "kmeans_distancegrouping.png")
    plt.close()

    # Average over each cluster
    aves = np.zeros((len(np.unique(labels)), len(time), 4))
    for i in range(len(np.unique(labels))):
        deltaxs = deltax[labels == i]
        deltays = deltay[labels == i]
        aves[i, :, 0] = np.average(deltaxs, axis=0)
        aves[i, :, 1] = np.average(deltays, axis=0)
        aves[i, :, 2] = np.std(deltaxs, axis=0)
        aves[i, :, 3] = np.std(deltays, axis=0)
    # Plot average centroids
    for i in range(len(np.unique(labels))):
        plt.plot(time, aves[i, :, 0], label=f"x+Label_{i}")
        plt.plot(time, aves[i, :, 1], label=f"y+Label_{i}")
        plt.ylabel(r"Averaged Centroid Shifts [mas]")
        plt.xlabel(r"Time")
        plt.legend()
    plt.savefig(PLOTDIR + "distance_grouping_ave.png")
    # Plot std centroids
    plt.figure()
    for i in range(len(np.unique(labels))):
        plt.plot(time, aves[i, :, 2], label=f"x+Label_{i}")
        plt.plot(time, aves[i, :, 3], label=f"y+Label_{i}")
        plt.ylabel(r"Std. Dev. Centroid Shifts [mas]")
        plt.xlabel(r"Time")
        plt.legend()
    plt.savefig(PLOTDIR + "distance_grouping_std.png")
    # Save aves to separate CSV files for each label
    for i in range(len(np.unique(labels))):
        label_aves = aves[i].reshape(-1, aves.shape[-1])
        label_header = "Time [days], Average X Centroid Shifts [mas], Average Y Centroid Shifts [mas], Std X Centroid Shifts [mas], Std Y Centroid Shifts [mas]"
        np.savetxt(
            PLOTDIR
            + f"data_average_centroids_{SAMPLEID}_Q{str(QTR).zfill(2)}_NS{NSTARS}_Label_{i}.csv",
            label_aves,
            delimiter=",",
            header=label_header,
        )


if __name__ == "__main__":
    main()
