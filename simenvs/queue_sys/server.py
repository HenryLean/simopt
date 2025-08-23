from functools import partial
from .._utils import rv_options



class Server(object):
    clock = 0
    def __init__(self, rate:float, Type="M", *args, **kwargs):
        self.rate = rate if rate > 1e-6 else abs(rate) + 1e-6
        self.rvs = partial(rv_options[Type], *args, **kwargs)

    def __call__(self, idle=0):
        rv = self.rvs(1/self.rate)
        self.clock += idle + rv
        return rv, self.clock
    
