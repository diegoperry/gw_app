#!/usr/bin/env python
"""

Defines the base SystematicModel class.

coded with help from chatgpt.

"""
import numpy as np
from abc import ABC, abstractmethod
import tables  # PyTables
import time


class SystematicModel:
    param_names = []  # Subclass must override

    def __init__(self, h5file=None):
        self.params = {k: None for k in self.param_names}
        self.is_fit = False
        self.fit_metadata = {}  # e.g. fit time, duration, quality metrics

        if h5file is not None:
            self.fit(h5file)  # Always fit on init
        else:
            raise ValueError("Must provide an h5file.")

    def fit(self, h5file):
        raise NotImplementedError

    def predict(self, t, *args, kicid=None, **kwargs):
        raise NotImplementedError

    def jacobian(self, t, *args, kicid=None, **kwargs):
        ## for use in setting up the design matrix
        raise NotImplementedError

    def validate_params(self):
        # should be called at the end of self.fit
        missing = [k for k in self.param_names if self.params[k] is None]
        if missing:
            raise ValueError(f"Missing required fit parameters: {missing}")
        self.is_fit = True

    def record_fit_metadata(self, **extra):
        # should be called at the end of self.fit
        self.fit_metadata.update(
            {
                "fit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fit_timestamp": time.time(),
            }
        )
        self.fit_metadata.update(extra)
