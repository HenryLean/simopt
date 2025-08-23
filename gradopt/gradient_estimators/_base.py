import numpy as np


class GradientCalBase(object):
    """Base class for stochastic gradient calculation.  
    In particular, it provides transient estimates"""
    temp = 0

    def __init__(self, init=.0, *, iter_fn=None):
        self.value = init
        if iter_fn is None:
            self.iter_fn = lambda phi, mul, val: phi + val * mul
        else:
            self.iter_fn = iter_fn

    def __call__(self, phi, mul):
        phi = 0 if np.isnan(phi) else float(phi)
        self.value = self.iter_fn(phi, mul, self.value)
        return self.value