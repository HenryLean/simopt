import numpy as np
from functools import partial
from .normal_return import PortfolioNormalReturn
from .._utils.rv_generators import rv_options


class PortfolioJumpReturn(PortfolioNormalReturn):
    def __init__(self, mean, cov=None, *, time_horizon=1, jump_rate=0.1, jump_type="normal", jump_kwargs={}):
        super().__init__(mean, cov)
        self.time_horizon = time_horizon
        self.jump_rate = jump_rate
        self.jump_rvs = partial(rv_options[jump_type], **jump_kwargs)


    def get_returns(self, n=1):
        n = 1 if n <= 0 else n

        rvs = super().get_returns(n) * self.time_horizon
        num_jumps = np.random.poisson(self.jump_rate * self.time_horizon, size=n)
        jumps = np.array([np.sum(self.jump_rvs(size=(j, self.dim)), axis=0) for _, j in enumerate(num_jumps)])

        return rvs + jumps
