import numpy as np


class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        sample = np.random.normal(self.mu, self.sigma, N)
        sample = sorted(sample)
        return sample


class Noise(object):
    def __init__(self, range_, scale=0.01):
        self.range_ = range_
        self.scale = scale

    def sample(self, N):
        sample = np.linspace(-self.range_, self.range_, N)
        sample += np.random.random(N) * self.scale
        return sample
