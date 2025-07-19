import numpy as np


'''
Interface class with one method, dn. 
dn takes in two arguments:
n_ -- stars' cartesian coordinates in telescope frame, numpy array.
t_ -- numpy array of times to evaluate deflections at.
and outputs an (B, 3, N) array of deflections, where B is the batch size (# time points) and N is the number of stars.

WARNING: in implementations, calling dn may mutate any time variables associated with the class; 
the function isn't guaranteed to have no side effects!
'''
class DeflectionSourceInterface:
    def dn(self, n_: np.ndarray, t_: np.ndarray) -> np.ndarray:
        pass