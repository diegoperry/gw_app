'''
This module calculates deflections to the star position using a specific GW waveform tensor, the GW source coordinate and the original star coordinate.
'''
## import necessary packages
import numpy as np
import astropy.units as u
import astropy.constants as const

## set constants
c = const.c
G = const.G
Gc3 = G/c**3

def dn(h_,q_,n_):
    '''Calculate coordinate deflections.

    Keyword arguments:
    h_ -- GW waveform tensor
    q_ -- GW source's cartesian coordinates in telescope frame, numpy array
    n_ -- star's cartesian coordinates in telescope frame, numpy array

    Returns:
    deflection vector of the star position, in cartesian coordinates in telescope frame, numpy array.
    Deflections are in units of radians, without an attached astropy unit.
    '''
    try:
        coeff1 = n_.dot(h_).dot(n_)
        coeff = .5/(1-q_.dot(n_))*coeff1
        if len(coeff.shape) == 1:   
            coeff = coeff[:, np.newaxis]
        term1 = coeff*(n_-q_)
        term2 = .5*h_.dot(n_)
    except:
        coeff1 = np.sum(np.matmul(h_,n_)*n_, axis=-2)
        coeff = .5/(1-np.matmul(q_, n_))*coeff1

        # Needed for proper batch multiplication: coeff is initially (B, N) while n_ - q_ is (3, N).
        # coeff after this will be (B, 3, N), repeated along the new axis 3 times.
        if len(coeff.shape) == 1:
            coeff = coeff[np.newaxis, :]
        coeff = np.repeat(coeff[:, np.newaxis, ...], 3, axis=1)

        term1 = coeff * np.subtract(n_,q_.reshape(3, 1))
        term2 = .5*h_.dot(n_)
    result = (term1 - term2).real
    # For a single input time, preserve old behavior by unwrapping batch axis (0th axis).
    # if result.shape[0] == 1:
    #     return result[0]
    # else:
    #     return result
    return result