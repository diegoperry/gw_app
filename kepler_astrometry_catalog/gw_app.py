#app
import sys
sys.path.append("/mnt/c/Users/diego/Downloads/gw_animations/kepler_astrometry_catalog/estoiles/estoiles")

from gw_source import GWSource

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from astropy.coordinates import SkyCoord
import astropy.units as u
import tempfile

st.set_page_config(page_title="GW Astrometric Animation", layout="centered")
st.title("🌌 Gravitational Wave Astrometry Simulator")

st.markdown("Choose your gravitational wave parameters and generate the animation.")

# ➕ Input form
with st.form("gw_form"):
    freq_val = st.number_input("Frequency (Hz)", value=1e-8, format="%e")
    inc_val = st.slider("Inclination (degrees)", min_value=0, max_value=180, value=90)
    q_val = st.number_input("Mass ratio (q)", value=1.0, min_value=0.01, step=0.01)
    M_val = st.number_input("Chirp mass (M⊙)", value=1e8, format="%e")
    dl_val = st.number_input("Luminosity distance (Mpc)", value=1.0)
    source_l = st.number_input("Source longitude (ℓ, degrees)", min_value=0.0, max_value=360.0, value=0.1)
    source_b = st.number_input("Source latitude (b, degrees)", min_value=-90.0, max_value=90.0, value=35.0)
    psi_val = st.slider("Polarization angle ψ (degrees)", min_value=0, max_value=180, value=0)

    submit = st.form_submit_button("Generate Animation")


if submit:
    with st.spinner("Simulating gravitational wave deflections and generating animation..."):
        # GW source setup
        sourcecoord = SkyCoord(l=source_l*u.deg, b=source_b*u.deg, frame='galactic')
        telcoord = SkyCoord(l=0*u.deg, b=90*u.deg, frame='galactic')

        gw_source = GWSource(
            freq=freq_val * u.Hz,
            Mc=M_val * u.Msun,
            q=q_val,
            dl=dl_val * u.Mpc,
            inc=inc_val * u.deg,
            psi=psi_val * u.deg,  # updated to use user input
            telcoord=telcoord,
            sourcecoord=sourcecoord
        )


        # Grid and deflections
        n = 20
        fov_radius = np.sqrt(115/np.pi) / 180 * np.pi
        x_vals = np.linspace(-1, 1, n)
        y_vals = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        mask = X**2 + Y**2 <= 1.0
        x = X[mask] * fov_radius
        y = Y[mask] * fov_radius
        z = np.ones_like(x)
        n_hat = np.stack([x, y, z], axis=0)

        days = 3 * 367
        t = np.linspace(0, 3600*24*days, 100) * u.s
        gw_source.time = t
        deflections = gw_source.dn(n_hat, t) * 5e13
        positions_deflected = n_hat[None, :, :] + deflections

        # Set up animation
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter([], [], color='red', s=10)

        lims = np.array([-1.2, 1.2]) * fov_radius
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)

        # Static parameter annotation
        ax.text(
            0.02, 0.98,
            f"freq = {freq_val:.1e} Hz\n"
            f"inc  = {inc_val:.0f}°\n"
            f"q    = {q_val:.2f}\n"
            f"M    = {M_val:.1e} M⊙\n"
            f"dl   = {dl_val:.1f} Mpc\n"
            f"src  = ({source_l:.1f}°, {source_b:.1f}°)",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        def init():
            sc.set_offsets(np.zeros((x.shape[0], 2)))
            return sc,

        def update(frame):
            pos = positions_deflected[frame]
            sc.set_offsets(np.column_stack([pos[0], pos[1]]))
            ax.set_title(f"t = {t[frame].to_value(u.s):.2f} s")
            return sc,

        anim = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)

        # Save animation to a temporary file on disk
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_filename = tmpfile.name

        writer = FFMpegWriter(fps=20, metadata={'artist': 'Streamlit'}, bitrate=1800)
        anim.save(temp_filename, writer=writer, dpi=150)

        # Load file contents as bytes and display
        with open(temp_filename, "rb") as f:
            video_bytes = f.read()

        st.success("Done! Here's your animation:")
        st.video(video_bytes, format="video/mp4")
