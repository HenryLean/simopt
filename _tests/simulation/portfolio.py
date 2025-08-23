import numpy as np
from simenvs.financial_sys import PortfolioNormalReturn


if __name__ == "__main__":
    mean = np.zeros(3)
    cov = np.array([
        [4, -2, -4],
        [-2, 5, 0],
        [-4, 0, 6]
    ])

    env = PortfolioNormalReturn(mean, cov)

    np.random.seed(666)

    theta = np.random.random(2) * np.pi / 2
    
    # print(env.get_returns(), env.get_weights(theta))
    print(np.mean(env.get_returns(1000) @ env.get_weights(theta)>=0.3))