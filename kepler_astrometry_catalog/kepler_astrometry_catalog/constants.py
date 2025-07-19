import astropy.units as u

PIX_SCALE_MAS = 3.98e3
PIX_TO_DEG = PIX_SCALE_MAS * (1.0 * u.mas).to(u.deg).value
