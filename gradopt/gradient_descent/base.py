import numpy as np
from abc import ABC, abstractmethod


class BaseGD(ABC):
    def __init__(self, init):
        self.vars = np.array(init)

    def __call__(self, gradient, *, lr=.01):
        temp_vars = self.vars - lr * gradient
        return temp_vars
    
    def iterate(self, gradient, *, lr=.01):
        self.vars = self._projection(self.__call__(vars - lr * gradient))
        return self.vars
    
    @abstractmethod
    def _projection(self, vars: np.ndarray):
        pass