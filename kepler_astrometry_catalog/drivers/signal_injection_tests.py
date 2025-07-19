from kepler_astrometry_catalog.clean import Clean
import astropy.units as u
import os
import numpy as np
from kepler_astrometry_catalog.paths import RESULTSDIR
from kepler_astrometry_catalog.fake_signal import FakeSignal
from astropy.time import Time
from kepler_astrometry_catalog.helpers import get_raDec2Pix
from kepler_astrometry_catalog.clean import Clean
from astropy.io import fits
from pathlib import Path
from sklearn.decomposition import PCA
import copy
from matplotlib import pyplot as plt
from kepler_astrometry_catalog.helpers import get_pca_basisvecs

deg2rad = np.pi / 180
rad2mas = 3600 * 180 / np.pi * 1000

seed, quarter = 1, 12
n_components = 3
i_samp = 1
# sampleid = f'brightestnonsat100_rot_mo{i_samp}'
sampleid = "brightestnonsat100_rot"
# sampleid = 'brightestnonsat1000_rot'

CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}/Q{quarter}")
Path(CACHEDIR).mkdir(parents=True, exist_ok=True)

print(sampleid, flush=1)


####### utility ########
def conv2cart(ra_, dec_):
    mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
    mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
    dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
    dy = (dec_ - mean_dec) * deg2rad * rad2mas
    return dx, dy


def demean(arr):
    arr_ = copy.deepcopy(arr)
    mean = np.mean(arr_, axis=1)[:, np.newaxis]
    arr_ -= mean
    mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
    arr_ -= mean_eigvec
    return arr_


##############################
##### cache most basic data ####
##############################
cache_Clean_fits = 0
if cache_Clean_fits:
    for cv in range(2):
        cl = Clean(
            sampleid=sampleid,
            verbose=1,
            conv2glob=cv,
            dvacorr=1,
            pca=0,
            save_pca_eigvec=0,
            remove_earth_points=1,
            max_nstars=-1,
            seed=seed,
            write_cleaned_data=1,
        )

##############################
##### cache fake signal data ####
##### use only when generating true local signal ####
##############################
cache_fake_signal = 0
if cache_fake_signal:

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    tag = "REAL_MOM"
    with fits.open(fitsfn) as hdu:

        time = hdu["TIME"].data["time"][0]
        time_bjd = time + 2454833  # bjdrefi
        time_obj = Time(time_bjd, format="jd", scale="tdb")

        star_ra = hdu["PATHS"].data["survey_ra"]
        star_dec = hdu["PATHS"].data["survey_dec"]
        rals = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["ra"]
        decls = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["dec"]

    rad2deg = 180 / np.pi

    ####### small angle #######

    l, b = 76.3 * u.deg, 13.5 * u.deg
    fs = FakeSignal(
        sampleid=sampleid,
        source_l=l,
        source_b=b,
        freq=1e-6 * u.Hz,
        ts=time,
        star_ra=star_ra,
        star_dec=star_dec,
    )
    dra, ddec = fs.data
    np.save(f"{CACHEDIR}/sm_dra.npy", dra)
    np.save(f"{CACHEDIR}/sm_ddec.npy", ddec)

    # local
    mag = 5e5 if l == 0.0 * u.deg else 5e6
    for im in [1, 10]:  # one extra for visualization
        mag *= im
        dra *= mag
        ddec *= mag
        xls = np.empty_like(dra)
        yls = np.empty_like(dra)

        if im == 1:
            xls_sig = np.empty_like(dra)
            yls_sig = np.empty_like(dra)

        for i, t in enumerate(time_obj):
            if np.mod(i + 1, 300) == 0:
                print(f"finished getting {i+1}/{len(time_obj)}")

            m, o, x, y = get_raDec2Pix(
                rals[:, i] + dra[:, i] * rad2deg, decls[:, i] + ddec[:, i] * rad2deg, t
            )
            xls[:, i] = x
            yls[:, i] = y
            if im == 1:
                m, o, x, y = get_raDec2Pix(
                    star_ra + dra[:, i] * rad2deg, star_dec + ddec[:, i] * rad2deg, t
                )
                xls_sig[:, i] = x
                yls_sig[:, i] = y

        magstr = "" if im == 1 else "ex_"
        np.save(f"{CACHEDIR}/{magstr}sm_x.npy", xls)
        np.save(f"{CACHEDIR}/{magstr}sm_y.npy", yls)
        if im == 1:
            np.save(f"{CACHEDIR}/{magstr}sm_x_sig.npy", xls_sig)
            np.save(f"{CACHEDIR}/{magstr}sm_y_sig.npy", yls_sig)

    ####### large angle #######
    l, b = 0.0 * u.deg, 11.0 * u.deg
    fs = FakeSignal(
        sampleid=sampleid,
        source_l=l,
        source_b=b,
        freq=1e-6 * u.Hz,
        ts=time,
        star_ra=star_ra,
        star_dec=star_dec,
    )
    dra, ddec = fs.data
    np.save(f"{CACHEDIR}/lg_dra.npy", dra)
    np.save(f"{CACHEDIR}/lg_ddec.npy", ddec)

    # local
    mag = 5e5 if l == 0.0 * u.deg else 5e6
    for im in [1, 10]:  # one extra for visualization
        mag *= im
        dra *= mag
        ddec *= mag
        xls = np.empty_like(dra)
        yls = np.empty_like(dra)

        if im == 1:
            xls_sig = np.empty_like(dra)
            yls_sig = np.empty_like(dra)

        for i, t in enumerate(time_obj):
            if np.mod(i + 1, 300) == 0:
                print(f"finished getting {i+1}/{len(time_obj)}")

            m, o, x, y = get_raDec2Pix(
                rals[:, i] + dra[:, i] * rad2deg, decls[:, i] + ddec[:, i] * rad2deg, t
            )
            xls[:, i] = x
            yls[:, i] = y
            if im == 1:
                m, o, x, y = get_raDec2Pix(
                    star_ra + dra[:, i] * rad2deg, star_dec + ddec[:, i] * rad2deg, t
                )
                xls_sig[:, i] = x
                yls_sig[:, i] = y

        magstr = "" if im == 1 else "ex_"
        np.save(f"{CACHEDIR}/{magstr}lg_x.npy", xls)
        np.save(f"{CACHEDIR}/{magstr}lg_y.npy", yls)
        if im == 1:
            np.save(f"{CACHEDIR}/{magstr}lg_x_sig.npy", xls_sig)
            np.save(f"{CACHEDIR}/{magstr}lg_y_sig.npy", yls_sig)

##############################
##### cache different angle fake signal ####
##############################
cache_angle = 0
if cache_angle:
    from astropy.coordinates import SkyCoord
    from scipy.optimize import minimize
    from raDec2Pix import raDec2PixModels as rm

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    tag = "REAL_MOM"
    with fits.open(fitsfn) as hdu:
        time = hdu["TIME"].data["time"][0]
        time_bjd = time + 2454833  # bjdrefi
        time_obj = Time(time_bjd, format="jd", scale="tdb")
        star_ra = hdu["PATHS"].data["survey_ra"]
        star_dec = hdu["PATHS"].data["survey_dec"]
        rals = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["ra"]
        decls = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["dec"]

    # getting 3 sources at 0, 45, 90 deg from the fov
    pointing = rm.pointingModel().get_pointing(time_obj.mjd)
    raPointing = np.mean(pointing[0])
    decPointing = np.mean(pointing[1])
    fov_skycoord = SkyCoord(ra=raPointing, dec=decPointing, unit="deg")
    raPointing_lon, decPointing_lat = (
        fov_skycoord.galactic.l.deg,
        fov_skycoord.galactic.b.deg,
    )

    def get_angle_w_fov(l):
        sc = SkyCoord(l=l * u.deg, b=decPointing_lat * u.deg, frame="galactic")
        test_ra = sc.l.deg * u.deg
        test_dec = sc.b.deg * u.deg
        return (
            calc_angle_on_sphere(
                test_ra, raPointing_lon * u.deg, test_dec, decPointing_lat * u.deg
            )
            .to(1 * u.deg)
            .value
        )

    def calc_angle_on_sphere(ra1, ra2, dec1, dec2):
        """
        Calculate the angle between two points on the sphere
        """
        ra1 = ra1.to(u.rad)
        ra2 = ra2.to(u.rad)
        dec1 = dec1.to(u.rad)
        dec2 = dec2.to(u.rad)
        return np.arccos(
            np.sin(dec1) * np.sin(dec2)
            + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        )

    def fun(l, target_angle):
        if l < raPointing_lon or l > raPointing_lon + 180:
            return 180
        return np.abs(get_angle_w_fov(l) - target_angle)

    res_long_45 = minimize(fun, raPointing_lon + 1, args=(45)).x[0]
    res_long_90 = minimize(fun, raPointing_lon + 1, args=(90)).x[0]
    lls = [raPointing_lon * u.deg, res_long_45 * u.deg, res_long_90 * u.deg]
    labels = ["0", "45", "90"]

    for l, angstr in zip(lls, labels):
        fs = FakeSignal(
            sampleid=sampleid,
            source_l=l,
            source_b=decPointing_lat * u.deg,
            freq=1e-6 * u.Hz,
            ts=time,
            star_ra=star_ra,
            star_dec=star_dec,
        )
        dra, ddec = fs.data
        np.save(f"{CACHEDIR}/{angstr}_dra.npy", dra)
        np.save(f"{CACHEDIR}/{angstr}_ddec.npy", ddec)
        print(f"finished with {angstr} deg injection source", flush=1)


##############################
##### plot local and dva fit ####
##############################
plot_local_dva = 0
if plot_local_dva:
    from kepler_astrometry_catalog.plotting import plot_raw_dvafit

    PLOTDIR = os.path.join(RESULTSDIR, "raw_detector")
    # directly get rctd from Clean, fake local not available
    for iss in [""]:
        plot_raw_dvafit(
            PLOTDIR,
            sampleid=sampleid,
            seed=seed,
            fix_quarter=quarter,
            plotfit=1,
            injected_sig_str=iss,
        )
    print(
        "finished plotting the raw centroid and dva (only available if using local coord)"
    )

##############################
##### OBSOLETE: clean sample cache pca results ####
##############################
clean_sample_do_pca = 0
if clean_sample_do_pca:

    ##### global solution #####
    cl = Clean(
        sampleid=sampleid,
        verbose=1,
        conv2glob=1,
        dvacorr=1,
        pca=1,
        save_pca_eigvec=1,
        remove_earth_points=1,
        max_nstars=-1,
        seed=seed,
    )

    # for angstr in ['sm','lg']:
    angstr = "lg"
    dra = np.load(f"{CACHEDIR}/{angstr}_dra.npy")
    ddec = np.load(f"{CACHEDIR}/{angstr}_ddec.npy")

    # magls = [5e5, 3e3, 1e7] # would be easier to compute what the magnification is here

    # need to understand the average size of the signal and the average size of noise factors
    # for icase, mag in enumerate(magls):

    mag = 5e5 if angstr == "lg" else 5e6
    dra *= mag
    ddec *= mag

    cl.set_ctd(dra=dra, ddec=ddec)
    cl.do_pca(save_eigvec=1, addsuffix=angstr)  # +f'_{icase}'

    print("done with global solution")

    # ##### local solution #####
    cl = Clean(
        sampleid=sampleid,
        verbose=1,
        conv2glob=0,
        dvacorr=1,
        pca=1,
        save_pca_eigvec=1,
        remove_earth_points=1,
        max_nstars=-1,
        seed=seed,
    )

    for angstr in ["lg_fake", "lg"]:
        cl.set_ctd(local_signal_str=angstr)
        cl.do_pca(save_eigvec=1, addsuffix=angstr)
    print("done with local solution")

##############################
##### plot pca eigenvectors ####
##############################

plot_pca_eigvec = 0
if plot_pca_eigvec:
    from kepler_astrometry_catalog.plotting import plot_pca_eig

    PLOTDIR = os.path.join(RESULTSDIR, "pca_eig")
    for angstr in ["", "0", "45", "90"]:
        for cg in range(2):
            mag = 5e5 if len(angstr) > 0 else 1.0
            plot_pca_eig(
                PLOTDIR,
                sampleid=sampleid,
                quarter=quarter,
                n_comp=3,
                dvacorr=1,
                conv2glob=cg,
                addsuffix=angstr,
                mag=mag,
                comp_eig=1,
            )
    print("done with plotting pca eigenvectors")

##############################
##### plot pca residuals #####
##############################
plot_pca_residuals = 0
if plot_pca_residuals:
    from kepler_astrometry_catalog.plotting import plot_pca_fit

    PLOTDIR = os.path.join(RESULTSDIR, "pcafit")
    for angstr in ["", "0", "45", "90"]:
        for cg in range(2):
            mag = 5e5 if len(angstr) > 0 else 1.0
            plot_pca_fit(
                PLOTDIR,
                seed=seed,
                sampleid=sampleid,
                quarter=quarter,
                n_components=n_components,
                n_samples=10,
                dvacorr=1,
                plotfit=1,
                conv2glob=cg,
                addsuffix=angstr,
                mag=mag,
                comp_eig=1,
            )

    print("done with plotting pca residuals")

##############################
##### make videos #####
# will be very slow for larger samples!!!
##############################
make_videos = 0
if make_videos:
    PLOTDIR = os.path.join(RESULTSDIR, "movies")
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    from kepler_astrometry_catalog.plotting import plot_chip_sky_frame

    for angstr in [""]:
        plot_chip_sky_frame(
            PLOTDIR,
            sampleid=sampleid,
            plot_chip=1,
            animate_chip=1,
            plot_sky=1,
            animate_sky=1,
            n_samples=-1,
            plot_arrow=0,
            addsuffix=angstr,
        )

##############################
#### scale signal from eigenvec ###
##############################
plot_scaled_signal = 0
if plot_scaled_signal:
    """This method injects signals with the same std as individual eigenvectors and checks the response of new eigendecomposition. All in global coordinates. The target repsonse is that, for large-scale signals, changing the size of the injection signal changes the mean, but not a lot in the other eigenvectors."""

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    with fits.open(fitsfn) as hdu:
        ra = hdu["RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu["RADEC_MODULE_OUTPUT"].data["dec"]

    mean_ra = np.nanmean(ra, axis=1)[:, np.newaxis]
    mean_dec = np.nanmean(dec, axis=1)[:, np.newaxis]

    deg2rad = np.pi / 180
    rad2mas = 3600 * 180 / np.pi * 1000

    dx = (ra - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
    dy = (dec - mean_dec) * deg2rad * rad2mas

    mean_dx = np.mean(dx, axis=1)[:, np.newaxis]
    dx -= mean_dx
    mean_dy = np.mean(dy, axis=1)[:, np.newaxis]
    dy -= mean_dy

    mean_eigvec_x = np.mean(dx, axis=0)[np.newaxis, :]
    dx_demean = dx - mean_eigvec_x
    mean_eigvec_y = np.mean(dy, axis=0)[np.newaxis, :]
    dy_demean = dy - mean_eigvec_y

    pca_x = PCA()
    pca_y = PCA()
    pca_x.fit(dx_demean)
    x_eigvec = pca_x.components_
    pca_y.fit(dy_demean)
    y_eigvec = pca_y.components_

    coef_x = np.einsum("ij,kj", x_eigvec, dx_demean)
    model_x = np.einsum("ki,kj->kij", coef_x, x_eigvec)
    norm2_x = np.einsum("ij, ij -> i", dx_demean, dx_demean)
    ip_x = np.einsum("kij, ij -> ki", model_x, dx_demean)
    norm1_x = np.einsum("kij, kij -> ki", model_x, model_x)
    corr_x = ip_x / np.sqrt(np.einsum("ij, j->ij", norm1_x, norm2_x))

    coef_y = np.einsum("ij,kj", y_eigvec, dy_demean)
    model_y = np.einsum("ki,kj->kij", coef_y, y_eigvec)
    norm2_y = np.einsum("ij, ij -> i", dy_demean, dy_demean)
    ip_y = np.einsum("kij, ij -> ki", model_y, dy_demean)
    norm1_y = np.einsum("kij, kij -> ki", model_y, model_y)
    corr_y = ip_y / np.sqrt(np.einsum("ij, j->ij", norm1_y, norm2_y))

    dra = np.load(f"{CACHEDIR}/lg_dra.npy")
    ddec = np.load(f"{CACHEDIR}/lg_ddec.npy")
    ra = dra * 180 / np.pi
    dec = ddec * 180 / np.pi
    mean_ra = np.nanmean(ra, axis=1)[:, np.newaxis]
    mean_dec = np.nanmean(dec, axis=1)[:, np.newaxis]
    dx = (ra - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
    dy = (dec - mean_dec) * deg2rad * rad2mas

    true_scale = np.median(np.sqrt(dx**2 + dy**2))
    magls = []
    model_dis = np.sqrt(model_x**2 + model_y**2)

    for i_eigvec in range(3):
        mul = np.median(model_dis[i_eigvec]) / true_scale
        magls.append(mul)
    # magls

    cl = Clean(
        sampleid=sampleid,
        verbose=1,
        conv2glob=1,
        dvacorr=1,
        pca=0,
        save_pca_eigvec=0,
        remove_earth_points=1,
        max_nstars=-1,
        seed=seed,
    )
    angstr = "lg"
    for icase, mag in enumerate(magls):
        cl.set_ctd(dra=dra * mag, ddec=ddec * mag)
        cl.do_pca(save_eigvec=1, addsuffix=angstr + f"_{icase}")

    from kepler_astrometry_catalog.plotting import plot_pca_eig

    PLOTDIR = os.path.join(RESULTSDIR, "pca_eig")
    for icase, mag in enumerate(magls):
        plot_pca_eig(
            PLOTDIR,
            sampleid=sampleid,
            quarter=quarter,
            n_comp=3,
            dvacorr=1,
            conv2glob=1,
            addsuffix=angstr + f"_{icase}",
            mag=mag,
        )

##############################
#### shuffled axis plot ###
##############################
plot_shuffle_axis = 0
if plot_shuffle_axis:
    """This is an earlier test where we shuffle x, y coordinates randomly to mimic the effect of using local coordinates. This also computes the correlation between the global eigenvectors and the fake local ones. The conclusion for Q12 is that, when we shuffle randomly, the correlation is high in the no injection case. With injected signal, the correlation is low."""

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    with fits.open(fitsfn) as hdu:
        ra = hdu["RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu["RADEC_MODULE_OUTPUT"].data["dec"]
    dra = np.load(f"{CACHEDIR}/lg_dra.npy") * 180 / np.pi
    ddec = np.load(f"{CACHEDIR}/lg_ddec.npy") * 180 / np.pi
    nstar = len(ra)

    def conv2cart(ra_, dec_):
        mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
        mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
        dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
        dy = (dec_ - mean_dec) * deg2rad * rad2mas
        return dx, dy

    def demean(arr):
        arr_ = copy.deepcopy(arr)
        mean = np.mean(arr_, axis=1)[:, np.newaxis]
        arr_ -= mean
        mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
        arr_ -= mean_eigvec
        return arr_

    def shuffle(dx, dy, xy_ind, sign_ind1, sign_ind2):
        dx_copy = copy.deepcopy(dx)
        dy_copy = copy.deepcopy(dy)
        dx_copy[xy_ind], dy_copy[xy_ind] = dy_copy[xy_ind], dx_copy[xy_ind]
        dx_copy[sign_ind1] *= -1
        dy_copy[sign_ind2] *= -1
        return dx_copy, dy_copy

    def comp_eig_corr(arr1, arr2, ncomp):
        pca1, pca2 = PCA(), PCA()
        pca1.fit(arr1)
        pca2.fit(arr2)
        eig1, eig2 = pca1.components_, pca2.components_
        ans = []
        for i in range(ncomp):
            ans.append(
                eig1[i].dot(eig2[i]) / np.linalg.norm(eig1[i]) / np.linalg.norm(eig2[i])
            )
        return ans

    np.random.seed(0)
    indxy = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]
    np.random.seed(1)
    ind1 = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]
    np.random.seed(2)
    ind2 = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]

    ncomp = 3
    magls = np.geomspace(2e4, 2e6, 3)
    n_real = 3
    res = np.empty([n_real, len(magls), 2, ncomp])

    for imag, mag in enumerate(magls):
        dx, dy = conv2cart(ra + dra * mag, dec + ddec * mag)
        dx_dm, dy_dm = demean(dx), demean(dy)

        for i in range(n_real):
            np.random.seed(i)
            indxy = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]
            np.random.seed(i + 45)
            ind1 = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]
            np.random.seed(i + 8)
            ind2 = np.argwhere(np.random.choice([0, 1], size=nstar))[:, 0]

            dx_shf, dy_shf = shuffle(dx, dy, indxy, ind1, ind2)
            dx_shf_dm, dy_shf_dm = demean(dx_shf), demean(dy_shf)

            res[i, imag, 0] = comp_eig_corr(dx_dm, dx_shf_dm, 3)
            res[i, imag, 1] = comp_eig_corr(dy_dm, dy_shf_dm, 3)

    res_avg = np.mean(res, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(ncomp):
        axs[0].plot(magls, abs(res_avg[:, 0, i]), label=f"comp {i+1}", marker=".")
        axs[1].plot(magls, abs(res_avg[:, 1, i]), label=f"comp {i+1}", marker=".")

    for i in range(2):
        axs[i].set_xlabel("signal magnification")
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")

    axs[0].set_ylabel("corr between global and local")
    axs[0].set_title("x")
    axs[1].set_title("y")
    fig.tight_layout()
    plt.savefig(f"{RESULTSDIR}/{sampleid}_{quarter}_eig_corr.png", bbox_inches="tight")

##############################
#### global local correlation ###
##############################
print_glob_loc_corr = 0
if print_glob_loc_corr:
    """This method computes the pairwise correlation between global and local eigenvectors"""
    # getting the global coordinates

    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    tag = "REA_MOM"
    with fits.open(fitsfn) as hdu:
        ra = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["dec"]
    dra = np.load(f"{CACHEDIR}/lg_dra.npy") * 180 / np.pi
    ddec = np.load(f"{CACHEDIR}/lg_ddec.npy") * 180 / np.pi

    nstar = len(ra)

    def conv2cart(ra_, dec_):
        mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
        mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
        dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
        dy = (dec_ - mean_dec) * deg2rad * rad2mas
        return dx, dy

    def demean(arr):
        arr_ = copy.deepcopy(arr)
        mean = np.mean(arr_, axis=1)[:, np.newaxis]
        arr_ -= mean
        mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
        arr_ -= mean_eigvec
        return arr_

    def comp_eig_corr(arr1, arr2, ncomp):
        pca1, pca2 = PCA(), PCA()
        pca1.fit(arr1)
        pca2.fit(arr2)
        eig1, eig2 = pca1.components_, pca2.components_
        ans = []
        for i in range(ncomp):
            ans.append(
                eig1[i].dot(eig2[i]) / np.linalg.norm(eig1[i]) / np.linalg.norm(eig2[i])
            )
        return ans

    # global coordinate
    ncomp = 3
    mag = 5e5
    angstr = ""
    if angstr == "lg":
        ra += dra * mag
        dec += ddec * mag
    glob_dx, glob_dy = conv2cart(ra, dec)
    glob_dx_dm, glob_dy_dm = demean(glob_dx), demean(glob_dy)

    # local coordinates
    if angstr == "lg":
        dx = np.load(f"{CACHEDIR}/{angstr}_x.npy")
        dy = np.load(f"{CACHEDIR}/{angstr}_y.npy")
    else:
        fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
        ext = "REAL_LOC_MOM_DVA_RESIDUAL"
        with fits.open(fitsfn) as hdu:
            dx = hdu[ext].data["centroid_x"]
            dy = hdu[ext].data["centroid_y"]
    dx_dm, dy_dm = demean(dx), demean(dy)

    print(comp_eig_corr(dx_dm, glob_dx_dm, 3))
    print(comp_eig_corr(dy_dm, glob_dy_dm, 3))


##############################
#### plot the global the local eigenvectors ###
##############################
plot_glob_loc_eigvec = 0
if plot_glob_loc_eigvec:
    """This method directly pulls from the get_pca_basisvecs function in helpers (which assumes both the local and global pca vectors have been cached) and plot the eigenvectors on the same figure"""
    globstr = "_glob"
    angstr = "lg"
    ncomp = 3
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    with fits.open(fitsfn) as hdu:
        time = hdu["TIME"].data["time"][0]

    basisvecs_x_glob, basisvecs_y_glob = get_pca_basisvecs(
        sampleid, "_dva", "_Q12", ncomp, "", globstr, addsuffix=f"_{angstr}"
    )
    globstr = ""
    basisvecs_x, basisvecs_y = get_pca_basisvecs(
        sampleid, "_dva", "_Q12", ncomp, "", globstr, addsuffix=f"_{angstr}"
    )

    def comp_corr(arr1, arr2):
        ans = []
        for ix in range(ncomp + 1):
            ans.append(
                arr1[ix].dot(arr2[ix])
                / np.linalg.norm(arr1[ix])
                / np.linalg.norm(arr2[ix])
            )
        return ans

    n_comp = 3
    fig, axs = plt.subplots(nrows=n_comp + 1, figsize=(5, 1.5 * (n_comp + 1)))
    corr_x = comp_corr(
        basisvecs_x, basisvecs_x_glob
    )  # already assumed first one is the mean vector
    corr_y = comp_corr(basisvecs_y, basisvecs_y_glob)

    for ix, ax in enumerate(axs):
        ax.scatter(time, basisvecs_x[ix], c="C0", s=1, linewidth=0, marker=".")
        ax.scatter(time, basisvecs_y[ix], c="C1", s=1, linewidth=0, marker=".")
        ax.scatter(time, basisvecs_x_glob[ix], c="C2", s=1, linewidths=0, marker=".")
        ax.scatter(time, basisvecs_y_glob[ix], c="C3", s=1, linewidths=0, marker=".")

        if ix != n_comp:
            ax.set_xticklabels([])

        if ix == 0:
            ax.scatter([], [], c="C0", s=1, label="x")
            ax.scatter([], [], c="C1", s=1, label="y")
            ax.scatter([], [], c="C2", s=1, label="x glob")
            ax.scatter([], [], c="C3", s=1, label="y glob")
            ax.legend(loc="upper right", fontsize=8)

        txt = f"EIGVEC{ix}" if ix != 0 else "MEAN"
        corr_txt = ", corr_x: {:.3f}, corr_y: {:.3f}".format(corr_x[ix], corr_y[ix])
        txt += corr_txt
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
    fig.savefig(
        f"{RESULTSDIR}/{sampleid}_{quarter}_glob_local_corr.png", bbox_inches="tight"
    )
    plt.close(fig)

##############################
#### plot max correlation between ctd and eigenvectors ####
##############################
plot_max_corr = 0
if plot_max_corr:
    """This method computes the maximum correlation between the ctd and individual eigenvectors (max is taken among the stars). This helps to see which eigenvectors contribute the most to the ctd. In Q12 brightestnonsat100_rot_mo1, we see a sharp drop after the 3rd eigenvector. This method assumes the with basisvectors are cached. If not, use the comp_pca_basisvecs in helpers instead."""
    angstr = "lg"
    conv2glob = 1
    ncomp = 3

    globstr = "_glob" if conv2glob else ""
    x_eigvec, y_eigvec = get_pca_basisvecs(
        sampleid,
        "_dva",
        "_Q12",
        ncomp,
        "",
        globstr,
        addsuffix=f"_{angstr}" if len(angstr) > 0 else "",
    )

    angstr = f"{angstr}_" if len(angstr) > 0 else ""

    def conv2cart(ra_, dec_):
        mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
        mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
        dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
        dy = (dec_ - mean_dec) * deg2rad * rad2mas
        return dx, dy

    def demean(arr):
        arr_ = copy.deepcopy(arr)
        mean = np.mean(arr_, axis=1)[:, np.newaxis]
        arr_ -= mean
        mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
        arr_ -= mean_eigvec
        return arr_

    if conv2glob:
        fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
        with fits.open(fitsfn) as hdu:
            ra = hdu["REAL_MOM_RADEC_MODULE_OUTPUT"].data["ra"]
            dec = hdu["REAL_MOM_RADEC_MODULE_OUTPUT"].data["dec"]
        if len(angstr) > 0:
            dra = np.load(f"{CACHEDIR}/{angstr}dra.npy") * 180 / np.pi
            ddec = np.load(f"{CACHEDIR}/{angstr}ddec.npy") * 180 / np.pi
            ra += dra
            ddec += ddec
        dx, dy = conv2cart(ra, dec)
    else:
        dx = np.load(f"{CACHEDIR}/{angstr}x.npy")
        dy = np.load(f"{CACHEDIR}/{angstr}y.npy")

    dx_dm, dy_dm = demean(dx), demean(dy)

    def per_eig_corr(eig, arr):
        coef = np.einsum("ij,kj", eig, arr)
        model = np.einsum("ki,kj->kij", coef, eig)
        norm2 = np.einsum("ij, ij -> i", arr, arr)
        ip = np.einsum("kij, ij -> ki", model, arr)
        norm1 = np.einsum("kij, kij -> ki", model, model)
        corr = ip / np.sqrt(np.einsum("ij, j->ij", norm1, norm2))
        return corr

    corr_x = per_eig_corr(x_eigvec, dx_dm)
    corr_y = per_eig_corr(y_eigvec, dy_dm)

    fig, ax = plt.subplots()
    ax.plot(np.max(corr_x, axis=1), label="x")
    ax.plot(np.max(corr_y, axis=1), label="y")
    ax.legend()
    fig.savefig(f"{RESULTSDIR}/{sampleid}_{quarter}_max_corr.png", bbox_inches="tight")
    plt.close(fig)

##############################
#### plot eigenvectors on-chip ###
##############################
plot_on_chip = 0
if plot_on_chip:
    """This function plots the local coordinate of randomly selected stars and what individual eigenvectors. For Q12 brightestnonsat100_rot_mo1, we see that many stars are moving in the diagonal direction (dva), and the leading eigenvectors are also aligned or orthogonal to this direction."""

    angstr = "lg"
    conv2glob = 1
    ncomp = 3
    globstr = "_glob" if conv2glob else ""
    x_eigvec, y_eigvec = get_pca_basisvecs(
        sampleid,
        "_dva",
        "_Q12",
        ncomp,
        "",
        globstr,
        addsuffix=f"_{angstr}" if len(angstr) > 0 else "",
    )

    angstr = f"{angstr}_" if len(angstr) > 0 else ""

    def conv2cart(ra_, dec_):
        mean_ra = np.nanmean(ra_, axis=1)[:, np.newaxis]
        mean_dec = np.nanmean(dec_, axis=1)[:, np.newaxis]
        dx = (ra_ - mean_ra) * np.cos(np.deg2rad(mean_dec)) * deg2rad * rad2mas
        dy = (dec_ - mean_dec) * deg2rad * rad2mas
        return dx, dy

    def demean(arr):
        arr_ = copy.deepcopy(arr)
        mean = np.mean(arr_, axis=1)[:, np.newaxis]
        arr_ -= mean
        mean_eigvec = np.mean(arr_, axis=0)[np.newaxis, :]
        arr_ -= mean_eigvec
        return arr_

    if conv2glob:
        fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
        with fits.open(fitsfn) as hdu:
            ra = hdu["REAL_MOM_RADEC_MODULE_OUTPUT"].data["ra"]
            dec = hdu["REAL_MOM_RADEC_MODULE_OUTPUT"].data["dec"]
        if len(angstr) > 0:
            dra = np.load(f"{CACHEDIR}/{angstr}dra.npy") * 180 / np.pi
            ddec = np.load(f"{CACHEDIR}/{angstr}ddec.npy") * 180 / np.pi
            ra += dra
            ddec += ddec
        dx, dy = conv2cart(ra, dec)
    else:
        dx = np.load(f"{CACHEDIR}/{angstr}x.npy")
        dy = np.load(f"{CACHEDIR}/{angstr}y.npy")

    dx_dm, dy_dm = demean(dx), demean(dy)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    np.random.seed(0)
    for i in np.random.choice(len(dx), size=10):
        axs[0].scatter(dx_dm[i, :], dy_dm[i, :], alpha=0.5, s=1, linewidth=0)
    axs[0].scatter(x_eigvec[0], y_eigvec[0], s=1, linewidth=0, color="k", label="mean")

    ax = axs[1]
    for i in range(1, 4):
        alpha = 0.2 if i == 3 else 1.0
        ax.scatter(
            x_eigvec[i, :],
            y_eigvec[i, :],
            s=1,
            linewidth=0,
            label=f"eigvec {i}",
            alpha=alpha,
        )
    for ax in axs:
        ax.set_xlabel("dx (ra direction) [mas]")
        ax.set_ylabel("dy (dec direction) [mas]")

    axs[0].set_title("post-DVA residual")
    axs[1].set_title("eigenvectors")
    plt.savefig(f"{RESULTSDIR}/on_chip.png", bbox_inches="tight")

##############################
#### simulation for continuous wave amplitude uncertainty estimate ###
##############################
amplitude_best_match = 0
if amplitude_best_match:
    pi = np.pi
    from scipy.optimize import minimize

    sigma = 0.1

    def get_meas(f0_, T_, dt_, A0_):
        t_meas = np.arange(0, T_, dt_)
        signal_true = A0_ * np.cos(2 * pi * f0_ * t_meas)

        def loss(x):
            amp, freq = x
            template = amp * np.cos(2 * pi * freq * t_meas)
            return np.sum((signal - template) ** 2)

        Ntrial = 2000
        res = np.empty([Ntrial, 2])

        for i in range(Ntrial):
            noise = sigma * np.random.randn(len(signal_true))
            signal = signal_true + noise
            res[i] = minimize(
                loss,
                x0=np.array([A0_, f0_]),
                bounds=[[0.0, A0_ + 2.0], [0.1, f0_ + 2.0]],
            ).x
        return res

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    # basecase
    f0, T, dt, A0 = 1.0, 20, 0.01, 1.0
    res = get_meas(f0, T, dt, A0)
    x = np.linspace(min(res[:, 0]), max(res[:, 0]), 200)
    sig_theo = sigma / np.sqrt(int(T / dt)) * np.sqrt(2)
    lb = "(Ncyle={:.1f},Nmeas={})".format(T * f0, int(T / dt))
    g = ax.plot(
        x,
        np.exp(-((x - A0) ** 2) / (2 * sig_theo**2)) / np.sqrt(2 * pi * sig_theo**2),
        label=lb,
    )
    ax.hist(res[:, 0], bins=50, density=True, alpha=0.3, color=g[0].get_color())[2]

    # increase frequency, but nothing else:
    f0, T, dt, A0 = 2.0, 20, 0.01, 1.0
    res = get_meas(f0, T, dt, A0)
    x = np.linspace(min(res[:, 0]), max(res[:, 0]), 200)
    sig_theo = sigma / np.sqrt(int(T / dt)) * np.sqrt(2)
    lb = "(Ncyle={:.1f},Nmeas={})".format(T * f0, int(T / dt))
    g = ax.plot(
        x,
        np.exp(-((x - A0) ** 2) / (2 * sig_theo**2)) / np.sqrt(2 * pi * sig_theo**2),
        label=lb,
    )
    ax.hist(res[:, 0], bins=50, density=True, alpha=0.3, color=g[0].get_color())[2]

    # same frequency, double measurements
    f0, T, dt, A0 = 1.0, 20, 0.005, 1.0
    res = get_meas(f0, T, dt, A0)
    x = np.linspace(min(res[:, 0]), max(res[:, 0]), 200)
    sig_theo = sigma / np.sqrt(int(T / dt)) * np.sqrt(2)
    lb = "(Ncyle={:.1f},Nmeas={})".format(T * f0, int(T / dt))
    g = ax.plot(
        x,
        np.exp(-((x - A0) ** 2) / (2 * sig_theo**2)) / np.sqrt(2 * pi * sig_theo**2),
        label=lb,
    )
    ax.hist(res[:, 0], bins=50, density=True, alpha=0.3, color=g[0].get_color())[2]

    ax.legend()
    ax.set_xlabel("amplitude")
    fig.savefig(f"{RESULTSDIR}/amplitude_best_match.png", bbox_inches="tight")

##############################
#### axis shuffling: data caching ###
##############################
cache_axis = 0
if cache_axis:
    """***THIS METHOD HAS BEEN MOVED TO HELPERS.PY***To make fake local coordinates, we need to compute what the axis of each module and output is in global coordinates. Since all stars on the same module and output will share the same axis, we can cache this information as a dictionary. This also assumes that for the quarter and sample we are interested in, we have already cached the time slice information in a FITS file"""

    from raDec2Pix import raDec2Pix
    from raDec2Pix import raDec2PixModels as rm

    quarter = 12
    fn = f"{RESULTSDIR}/temporary/Q{quarter}_output_axis.npy"
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"

    # this assumes we already saved the cleaned object slices
    with fits.open(fitsfn) as hdu:
        time = hdu["TIME"].data["time"][0]
        time_bjd = time + 2454833  # bjdrefi
        time_obj = Time(time_bjd, format="jd", scale="tdb")

    print(f"number of exposures: {len(time_obj)}")

    rdp = raDec2Pix.raDec2PixClass()
    NCOL, NROW = rm.get_parameters("nColsImaging"), rm.get_parameters("nRowsImaging")
    groups = [
        [18, 19, 22, 23, 24],
        [6, 11, 12, 13, 16, 17],
        [9, 10, 14, 15, 20],
        [2, 3, 4, 7, 8],
    ]

    all_module = [groups[i][0] for i in range(len(groups))]
    group_key = {all_module[i]: i for i in range(len(all_module))}
    # all_module = np.array([i for i in range(2,5)] + [i for i in range(6,21)] + [i for i in range(22,25)])
    all_module, all_output = np.meshgrid(all_module, np.arange(1, 5))
    all_module = all_module.flatten()
    all_output = all_output.flatten()

    def get_unit_vec(start_ra, end_ra, start_dec, end_dec):
        dx = (end_ra - start_ra) * np.cos(np.deg2rad(start_dec))
        dy = end_dec - start_dec
        xvec = np.array([dx, dy])
        xvec /= np.linalg.norm(xvec, axis=0)
        return xvec

    hm = dict()

    # how do we want to do this, want to do this in parallel or to save as different files?
    for m, o in zip(all_module, all_output):
        origin_ra, origin_dec = rdp.pix_2_ra_dec(
            m, o, 0.0, 0.0, time_obj.mjd, aberrateFlag=1
        )
        xmax_ra, xmax_dec = rdp.pix_2_ra_dec(
            m, o, NROW, 0.0, time_obj.mjd, aberrateFlag=1
        )
        ymax_ra, ymax_dec = rdp.pix_2_ra_dec(
            m, o, 0.0, NCOL, time_obj.mjd, aberrateFlag=1
        )

        xvec = get_unit_vec(origin_ra, xmax_ra, origin_dec, xmax_dec)
        yvec = get_unit_vec(origin_ra, ymax_ra, origin_dec, ymax_dec)

        hm[(m, o)] = np.array([xvec, yvec])
        print(m, o)
    np.save(fn, hm)

##############################
#### check the difference between the true local and the fake local ####
##############################
check_fake_local = 0
if check_fake_local:
    """This assumes that the true local have been cached, after checking this is valid we are only caching global injection signal. If wanting to check again, generate coordinates (by looping over either star or time slice, see above)"""

    def remove_time_mean(arr):
        return arr - np.mean(arr, axis=1)[:, np.newaxis]

    CACHEDIR = os.path.join(RESULTSDIR, f"temporary/{sampleid}")

    quarter = 12
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    ## modify extension names based on which ra,dec and residual is used
    with fits.open(fitsfn) as hdu:
        ra = hdu["RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu["RADEC_MODULE_OUTPUT"].data["dec"]
        module = hdu["RADEC_MODULE_OUTPUT"].data["module"][:, 0]
        output = hdu["RADEC_MODULE_OUTPUT"].data["output"][:, 0]

        dx = hdu["LOC_DVA_CENTROIDS"].data["dva_centroid_x"]
        dy = hdu["LOC_DVA_CENTROIDS"].data["dva_centroid_y"]
        dva_x = hdu["CORR"].data["corr_x"]
        dva_y = hdu["CORR"].data["corr_y"]

        star_ra = hdu["PATHS"].data["survey_ra"]
        star_dec = hdu["PATHS"].data["survey_dec"]

    glob_dx, glob_dy = conv2cart(ra, dec)  # in mas
    glob_dx, glob_dy = remove_time_mean(glob_dx), remove_time_mean(glob_dy)

    ## no signal injection version!!!
    # getting the dva residual from the FITS file
    # the true local coord, center
    dx, dy = remove_time_mean(dx), remove_time_mean(dy)  # in units of mas

    # also load the fake version
    fake_dx, fake_dy = np.load(f"{CACHEDIR}/fake_x.npy"), np.load(
        f"{CACHEDIR}/fake_y.npy"
    )
    fake_dx -= dva_x
    fake_dy -= dva_y  # in pixel
    pix2mas = 3.98e3
    fake_dx *= pix2mas
    fake_dy *= pix2mas
    fake_dx, fake_dy = remove_time_mean(fake_dx), remove_time_mean(fake_dy)

    ## with signal injection
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    with fits.open(fitsfn) as hdu:
        ra = hdu["RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu["RADEC_MODULE_OUTPUT"].data["dec"]

    angstr = "lg"
    mag = 5e5
    if len(angstr) > 0:
        dra = np.load(f"{CACHEDIR}/{angstr}_dra.npy") * 180 / np.pi
        ddec = np.load(f"{CACHEDIR}/{angstr}_ddec.npy") * 180 / np.pi
        ra += dra * mag
        dec += ddec * mag

    glob_dx, glob_dy = conv2cart(ra, dec)  # in mas
    glob_dx, glob_dy = remove_time_mean(glob_dx), remove_time_mean(glob_dy)

    # the true local signal
    x = np.load(f"{CACHEDIR}/{angstr}_x.npy")  # in pix
    y = np.load(f"{CACHEDIR}/{angstr}_y.npy")
    dx = (x - dva_x) * pix2mas  # dva also in pix in mas
    dy = (y - dva_y) * pix2mas
    dx, dy = remove_time_mean(dx), remove_time_mean(dy)

    # the fake local signal
    x = np.load(f"{CACHEDIR}/{angstr}_fake_x.npy")  # in pix
    y = np.load(f"{CACHEDIR}/{angstr}_fake_y.npy")
    fake_dx = (x - dva_x) * pix2mas  # dva also in pix in mas
    fake_dy = (y - dva_y) * pix2mas
    fake_dx, fake_dy = remove_time_mean(fake_dx), remove_time_mean(fake_dy)

    # make histogram of the fake-true local coordinate difference
    avg_diff = np.mean(np.sqrt((fake_dx - dx) ** 2 + (fake_dy - dy) ** 2), axis=1)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(avg_diff, histtype="step", bins=25)[2]
    ax.set_title(
        "fake-true difference"
    )  # this is quite small difference, which is what we want
    ax.set_xlabel("mas")

##############################
#### eigenvector correlation test with fake local ###
##############################
live_test_fake_local = 0
if live_test_fake_local:
    from kepler_astrometry_catalog.helpers import comp_pca_basisvecs

    """This method plots the difference in the absolute value of correlation between global mean and global/local eigenvectors. We either vary the angle between the source and the fov or the magnitude of the signal"""

    ##### different angular separation with the fov, same magnification #####
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    axs = axs.flatten()
    axs[0].set_title("abs(<glob mean,glob>)-abs(<glob mean,fake_loc>)")
    axs[1].set_xlabel("nth eigenvector")

    mag = 5e5

    for angstr in ["0", "45", "90"]:  # this loop is very fast
        loc_eigenvecs_x, loc_eigenvecs_y = comp_pca_basisvecs(
            sampleid, quarter, -1, 0, sig_only=0, addsuffix=angstr, mag=mag
        )
        nstar = len(loc_eigenvecs_x) - 1

        glob_eigenvecs_x, glob_eigenvecs_y = comp_pca_basisvecs(
            sampleid, quarter, -1, 1, sig_only=0, addsuffix=angstr, mag=mag
        )

        # for x
        glob_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_x[0], glob_eigenvecs_x[1:])
            / np.linalg.norm(glob_eigenvecs_x[0])
            / np.linalg.norm(glob_eigenvecs_x[1:], axis=1)
        )

        loc_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_x[0], loc_eigenvecs_x[1:])
            / np.linalg.norm(glob_eigenvecs_x[0])
            / np.linalg.norm(loc_eigenvecs_x[1:], axis=1)
        )

        axs[0].plot(
            np.arange(nstar) + 1,
            abs(glob_corr) - abs(loc_corr),
            label=f"x:{angstr} deg",
        )

        # for y
        glob_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_y[0], glob_eigenvecs_y[1:])
            / np.linalg.norm(glob_eigenvecs_y[0])
            / np.linalg.norm(glob_eigenvecs_y[1:], axis=1)
        )

        loc_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_y[0], loc_eigenvecs_y[1:])
            / np.linalg.norm(glob_eigenvecs_y[0])
            / np.linalg.norm(loc_eigenvecs_y[1:], axis=1)
        )

        axs[1].plot(
            np.arange(nstar) + 1,
            abs(glob_corr) - abs(loc_corr),
            label=f"y:{angstr} deg",
        )
    for ax in axs:
        ax.legend()
        ax.set_xscale("log")
    fig.tight_layout()
    plt.savefig(
        f"{RESULTSDIR}/{sampleid}_{quarter}_abs_corr_loc_glob_angle.png",
        bbox_inches="tight",
    )

    #### 90 deg separation, varying magnification ####
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    axs = axs.flatten()
    axs[0].set_title("abs(<glob mean,glob>)-abs(<glob mean,fake_loc>)")
    axs[1].set_xlabel("nth eigenvector")

    magls = np.geomspace(1e5, 4e6, 6)
    import matplotlib as mpl

    cmap = mpl.colormaps["coolwarm"]
    angstr = "90"

    for i, mag in enumerate(magls):  # this loop is very fast
        loc_eigenvecs_x, loc_eigenvecs_y = comp_pca_basisvecs(
            sampleid, quarter, -1, 0, sig_only=0, addsuffix=angstr, mag=mag
        )
        nstar = len(loc_eigenvecs_x) - 1

        glob_eigenvecs_x, glob_eigenvecs_y = comp_pca_basisvecs(
            sampleid, quarter, -1, 1, sig_only=0, addsuffix=angstr, mag=mag
        )

        # for x
        glob_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_x[0], glob_eigenvecs_x[1:])
            / np.linalg.norm(glob_eigenvecs_x[0])
            / np.linalg.norm(glob_eigenvecs_x[1:], axis=1)
        )

        loc_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_x[0], loc_eigenvecs_x[1:])
            / np.linalg.norm(glob_eigenvecs_x[0])
            / np.linalg.norm(loc_eigenvecs_x[1:], axis=1)
        )

        axs[0].plot(
            np.arange(nstar) + 1,
            abs(glob_corr) - abs(loc_corr),
            label="x:{:.1e}".format(mag),
            color=cmap((i + 1) / len(magls)),
        )

        # for y
        glob_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_y[0], glob_eigenvecs_y[1:])
            / np.linalg.norm(glob_eigenvecs_y[0])
            / np.linalg.norm(glob_eigenvecs_y[1:], axis=1)
        )

        loc_corr = (
            np.einsum("j,ij->i", glob_eigenvecs_y[0], loc_eigenvecs_y[1:])
            / np.linalg.norm(glob_eigenvecs_y[0])
            / np.linalg.norm(loc_eigenvecs_y[1:], axis=1)
        )

        axs[1].plot(
            np.arange(nstar) + 1,
            abs(glob_corr) - abs(loc_corr),
            label="y:{:.1e}".format(mag),
            color=cmap((i + 1) / len(magls)),
        )
    for ax in axs:
        ax.legend(ncol=2, loc="upper right")
        ax.set_xscale("log")
    fig.tight_layout()
    plt.savefig(
        f"{RESULTSDIR}/{sampleid}_{quarter}_abs_corr_loc_glob_mag.png",
        bbox_inches="tight",
    )
    print(
        f"finished saving {RESULTSDIR}/{sampleid}_{quarter}_abs_corr_loc_glob_mag.png",
        flush=1,
    )

##############################
#### global mean and temperature parameter correlation ###
##############################
glob_mean_temp_corr = 0
if glob_mean_temp_corr:
    fitsfn = RESULTSDIR + f"/cleaned_centroids/{sampleid}_{quarter}.fits"
    tag = "REAL_MOM"
    with fits.open(fitsfn) as hdu:
        ra = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["ra"]
        dec = hdu[f"{tag}_RADEC_MODULE_OUTPUT"].data["dec"]
        time = hdu["TIME"].data["time"][0]
        sel = hdu["SEL_MASK"].data["sel_mask"]

    time_bjd = time[0] + 2454833  # bjdrefi
    time_obj = Time(time_bjd, format="jd", scale="tdb")
    time -= time[0]

    from kepler_astrometry_catalog.paths import DATADIR
    import pandas as pd

    outpath = os.path.join(DATADIR, "ancillary/anc_pca_ready.csv")
    df = pd.read_csv(outpath)
    temp_column = "PEDPMAT1"
    temp_data = np.array(df[temp_column])[sel]

    target_scale = 1e-5
    d = 10

    from raDec2Pix import raDec2PixModels as rm

    pointing = rm.pointingModel().get_pointing(time_obj.mjd)
    raPointing = np.mean(pointing[0])
    decPointing = np.mean(pointing[1])

    delta_ra = ra - raPointing + d
    delta_dec = dec - decPointing + d
    scale = (np.mean(delta_ra) + np.mean(delta_dec)) / 2
    mag = target_scale / scale

    Tls = np.geomspace(1, 180, 22)
    init_pls = np.linspace(0, 2 * np.pi, 20)

    x_temp = np.empty([len(Tls), len(init_pls)])
    y_temp = np.empty([len(Tls), len(init_pls)])

    from kepler_astrometry_catalog.helpers import corr, PCA_live, conv2cart

    for i, period in enumerate(Tls):
        for j, init_phase in enumerate(init_pls):

            thermal_freq = 1 / period
            thermal_phase = np.cos(2 * pi * time * thermal_freq + init_phase)

            inj_ra = ra + delta_ra * mag * thermal_phase
            inj_dec = dec + delta_dec * mag * thermal_phase

            glob_dx, glob_dy = conv2cart(inj_ra, inj_dec)

            glob_mean_x, _ = PCA_live(glob_dx)
            glob_mean_y, _ = PCA_live(glob_dy)

            x_temp[i, j] = corr(temp_data, glob_mean_x)
            y_temp[i, j] = corr(temp_data, glob_mean_y)

    T_grid = np.log10(Tls)
    T, P = np.meshgrid(T_grid, init_pls)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    axs = axs.flatten()

    ax = axs[0]
    g = ax.pcolormesh(T, P, x_temp[:, :].T, shading="nearest")
    ax.set_title(f"corr x <glob mean,{temp_column}>")

    ax = axs[1]
    g = ax.pcolormesh(T, P, y_temp[:, :].T, shading="nearest")
    ax.set_title(f"corr y <glob mean,{temp_column}>")

    for ax in axs:
        ax.set_xticks(T_grid[::3])
        ax.set_xticklabels(["{:.1f}".format(t) for t in Tls[::3]])
        fig.colorbar(g, ax=ax)
        ax.set_ylabel("initial phase [rad]")
        ax.hlines(pi, T_grid[0], T_grid[-1], color="k", linestyle="--")

    axs[1].set_xlabel("injected signal period [d]")
    fig.tight_layout()
    plt.savefig(
        f"{RESULTSDIR}/{sampleid}_{quarter}_glob_mean_temp_corr.png",
        bbox_inches="tight",
    )
    print(
        f"finished saving {RESULTSDIR}/{sampleid}_{quarter}_glob_mean_temp_corr.png",
        flush=1,
    )

##############################
## plot rotated local coordinates in global frame ######
plot_rot_local = 0
if plot_rot_local:
    # read the averaged scripts
    mo_fitsfn = RESULTSDIR + f"/cleaned_centroids/ppa_catalog_q12_12_mo.fits"
    with fits.open(mo_fitsfn) as hdu:
        time = hdu["TIME"].data["time"]  # in days, (Ntimes)
        avg_ra = hdu["RADEC"].data["RA"]  # in deg, (N_output, Ntimes)
        avg_dec = hdu["RADEC"].data["DEC"]  # in deg, (N_output, Ntimes)

    i = 10
    di = 1000
    fig, ax = plt.subplots(figsize=(5, 5))

    # copied from the plotting script method
    from raDec2Pix import raDec2Pix
    from raDec2Pix import raDec2PixModels as rm

    NCOL, NROW = rm.get_parameters("nColsImaging"), rm.get_parameters("nRowsImaging")
    rdp = raDec2Pix.raDec2PixClass()

    bjdrefi = 2454833
    time_bjd = time[0] + bjdrefi
    init_mjds = Time(time_bjd, format="jd", scale="tdb")
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

    draw_chip_patch_onsky(ax)

    ax.quiver(
        avg_ra[:, i],
        avg_dec[:, i],
        2 * (avg_ra[:, i + di] - avg_ra[:, i]),
        2 * (avg_dec[:, i + di] - avg_dec[:, i]),
        zorder=10,
    )

    ax.set_xlabel("ra [deg]")
    ax.set_ylabel("dec [deg]")
    ax.set_title("avg rotated local coordinate (PSF)")
    fig.savefig(f"{RESULTSDIR}/rot_local.png", bbox_inches="tight")
    plt.close(fig)

##############################
### read module, output averaged centroids convenience
##############################
read_mo = 0
if read_mo:
    from astropy.io import fits
    from kepler_astrometry_catalog.paths import RESULTSDIR

    # load time series from all 9k PPA stars
    fn = RESULTSDIR + f"/cleaned_centroids/ppa_catalog_q12_12.fits"
    with fits.open(fn) as hdu:
        time = hdu["TIME"].data["time"][0]  # in days, (Nstar, Ntimes)
        ext_hdr = "FAKE_MOM_RADEC_MODULE_OUTPUT"
        ra = hdu[ext_hdr].data["ra"]  # in deg, (Nstar, Ntimes)
        dec = hdu[ext_hdr].data["dec"]  # in deg, (Nstar, Ntimes)

    # load ra and dec averaged from individual module and output
    mo_fitsfn = RESULTSDIR + f"/cleaned_centroids/ppa_catalog_q12_12_mo.fits"
    with fits.open(mo_fitsfn) as hdu:
        time = hdu["TIME"].data["time"]  # in days, (Ntimes)
        avg_ra = hdu["RADEC"].data["RA"]  # in deg, (N_output, Ntimes)
        avg_dec = hdu["RADEC"].data["DEC"]  # in deg, (N_output, Ntimes)
