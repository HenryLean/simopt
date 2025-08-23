import numpy as np
from abc import ABC, abstractmethod


class Portfolio(ABC):
    def __call__(self, theta:np.ndarray, n=1):
        returns = self.get_returns(n)
        portfolio_return = returns @ self.get_weights(theta)
        return portfolio_return
    

    @abstractmethod
    def get_returns(self, n=1):
        pass
    

    def is_satisficed(self, theta:np.ndarray, target:float, n=1):
        portfolio_return = self.__call__(theta, n)
        return np.mean(portfolio_return >= target)
    

    def get_weights(self, theta:np.ndarray):
        if self.dim < 2:
            return np.ones(1)

        theta = self._validate_theta(theta)
        sin2 = np.sin(theta)**2
        cos2 = np.cos(theta)**2
        if self.dim == 2:
            weights = np.array([cos2[0], sin2[0]])
            return weights

        cum_prod = np.cumprod(sin2)
        weights = np.zeros(self.dim)
        weights[0] = np.cos(theta[0])**2
        weights[1:-1] = cos2[1:] * cum_prod[:-1]
        weights[-1] = cum_prod[-1]
        return weights


    def _validate_theta(self, theta):
        if not isinstance(theta, np.ndarray):
            return np.clip(np.random.random(self.dim-1), 0, np.pi/2)
        
        if theta.ndim != (self.dim-1,):
            theta = theta.flatten()
            d = self.dim -1 - len(theta)
            if d > 0:
                return np.clip(np.concatenate([theta, np.random.random(d)]), 0, np.pi/2)
            else:
                return np.clip(theta[:self.dim-1], 0, np.pi/2)

        return np.clip(theta, 0, np.pi/2)
