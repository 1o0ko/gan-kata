import numpy as np


class Gaussian(object):
    def __init__(self, mu=4, sigma=0.5):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        sample = np.random.normal(self.mu, self.sigma, N)
        sample = sorted(sample)
        return sample


class Noise(object):
    def __init__(self, range_, scale=0.01):
        self.range = range_
        self.scale = scale

    def sample(self, N):
        sample = np.linspace(-self.range, self.range, N)
        sample += np.random.random(N) * self.scale
        return sample
