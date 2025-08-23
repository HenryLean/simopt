from ._base import *


class PA_Base(GradientCalBase):
    est = 0
    def __init__(self, fn, chain_fn, init=0, iter_fn=None, **kwds):
        super(PA_Base, self).__init__(init, iter_fn=iter_fn)
        self.phi = fn
        self.chain_fn = chain_fn

    def __call__(self, busy:bool, **kwargs):
        phi = self.phi(**kwargs)
        cum = super().__call__(phi, busy)
        return self.chain_fn(cum, **kwargs)
    
    def get_est(self, weight:float, *args, **kwargs):
        y = self.__call__(*args, **kwargs)
        self.est += weight * (y - self.est)
        return self.est

    def get_est_regene(self, weight:float, estDenom, *args, **kwargs):
        y = self.__call__(*args, **kwargs)
        self.est += weight * (y - estDenom * self.est)
        return self.est
    
