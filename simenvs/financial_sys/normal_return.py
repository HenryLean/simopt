import numpy as np
from numpy.linalg import cholesky
from ._base import Portfolio



class PortfolioNormalReturn(Portfolio):
    def __init__(self, mean, cov=None):
        self.mean = mean
        self.dim = len(mean)
        self.cov_factor = self._validate_cov(cov, self.dim)
    

    def get_returns(self, n=1):
        n = 1 if n <= 0 else n
        
        rvs = np.random.normal(size=(n, self.dim))
        rvs = rvs @ self.cov_factor.T + self.mean
        return rvs
    

    def _validate_cov(self, cov, dim):
        default_cov = np.eye(dim)

        if cov is None: 
            return default_cov
        
        try:
            cov = np.array(cov, dtype=np.float64)
        except:
            return default_cov
        
        if cov.shape != (dim, dim):
            return default_cov
        
        if not np.allclose(cov, cov.T, atol=1e-8):
            return default_cov
        
        try:
            L = cholesky(cov)
        except np.linalg.LinAlgError:
            return default_cov
        
        return L