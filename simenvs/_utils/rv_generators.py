import numpy as np
from scipy import stats



def get_constant(mean, *args, **kwds): 
    return mean


def get_exponent_rv(mean, *args, **kwds): 
    return -np.log(np.random.rand())*mean


def get_gamma_rv(mean, *args, **kwds):
    if "alph" in kwds.keys():
        a = kwds["alph"]
        scale = mean / a
    elif "scale" in kwds.keys():
        scale = kwds["scale"]
        a = mean / scale
    else:
        scale = 1
        a = mean / scale
    rvs = stats.gamma.rvs(a, scale=scale, size=1)
    return rvs[0]


def get_pareto_rv(mean, alpha=3, *args, **kwds): 
    scale = mean / alpha * (alpha-1)
    return scale * (np.random.rand()**(-1/alpha) - 1)


def get_lognorm_rv(mean, s=1, *args, **kwds):
    log = np.random.normal(loc=np.log(mean)-s*s/2, scale=s)
    return np.exp(log)


rv_options = {
    "U": lambda mean, scale=1: mean+scale*(2*np.random.rand()-1),
    "M": get_exponent_rv,
    "D": get_constant,
    "gamma": get_gamma_rv,
    "pareto": get_pareto_rv,
    "lognorm": get_lognorm_rv
}


jump_options = {
    "normal": np.random.normal,
    "uniform": lambda loc, scale, size: loc + scale*(2*np.random.rand(*size)-1),
    "exponent": lambda scale, size: -np.log(np.random.rand(*size))*scale,
    "constant": lambda loc, size: np.full(size, loc),
    "gamma": lambda a, scale, size: stats.gamma.rvs(a, scale=scale, size=size),
    "pareto": lambda scale, alpha, size: scale * (np.random.rand(*size)**(-1/alpha) - 1),
    "lognorm": lambda s, loc, size: np.exp(np.random.normal(loc=np.log(loc)-s*s/2, scale=s, size=size))
}
