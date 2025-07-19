from dataclasses import dataclass

import numpy as np
import astropy.units as u

from estoiles.interfaces import DeflectionSourceInterface

def make_file_path(directory, array_kwargs, extra_string=None, ext='.dat'):
    ## Makes file path given string and array kwargs.
    s = '_'
    string_kwargs = [str(int(i)) for i in array_kwargs]
    string_kwargs = np.array(string_kwargs, dtype='U25')
    if (extra_string !=None) and (len(extra_string)>25):
        raise TypeError('Extra string must have less than 25 characters')
    if extra_string !=None:
        string_kwargs = np.insert(string_kwargs, 0, extra_string)
    kwpath = s.join(string_kwargs)
    return directory+kwpath+ext

def compose_dn(deflection_source_list, n_, t_):
    return np.sum([src.dn(n_, t_) for src in deflection_source_list], axis=0)

# @dataclass
# class CombinedDeflections(DeflectionSourceInterface):
#     deflection_source_list: list[DeflectionSourceInterface]

#     def dn(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
#         return np.sum([src.dn(n_, t_) for src in self.deflection_source_list], axis=0)