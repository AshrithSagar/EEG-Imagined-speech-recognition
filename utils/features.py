"""
features.py
Feature Utility scripts
"""

import numpy as np
from scipy import integrate, stats
import antropy as ant


class FeatureFunctions:
    """Feature functions to be used in the feature extraction process."""

    def __init__(self, sampling_freq):
        self.sampling_freq = sampling_freq

    def get(self):
        """Returns a list of feature functions to be used in the feature extraction process."""

        def mean(x):
            return np.mean(x)

        def absmean(x):
            return np.mean(np.abs(x))

        def maximum(x):
            return np.max(x)

        def absmax(x):
            return np.max(np.abs(x))

        def minimum(x):
            return np.min(x)

        def absmin(x):
            return np.min(np.abs(x))

        def minplusmax(x):
            return np.max(x) + np.min(x)

        def maxminusmin(x):
            return np.max(x) - np.min(x)

        def curvelength(x):
            cl = 0
            for i in range(x.shape[0] - 1):
                cl += abs(x[i] - x[i + 1])
            return cl

        def energy(x):
            return np.sum(np.multiply(x, x))

        def nonlinear_energy(x):
            # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
            x_squared = x[1:-1] ** 2
            subtrahend = x[2:] * x[:-2]
            return np.sum(x_squared - subtrahend)

        def spectral_entropy(x):
            return ant.spectral_entropy(
                x,
                self.sampling_freq,
                method="welch",
                normalize=True,
                nperseg=len(x),
            )

        def integral(x):
            return integrate.simps(x)

        def stddeviation(x):
            return np.std(x)

        def variance(x):
            return np.var(x)

        def skewness(x):
            return stats.skew(x)

        def kurtosis(x):
            return stats.kurtosis(x)

        def sum(x):
            return np.sum(x)

        def sample_entropy(x):
            return ant.sample_entropy(x, order=2, metric="chebyshev")

        def permutation_entropy(x):
            return ant.perm_entropy(x, order=3, normalize=True)

        def svd_entropy(x):
            return ant.svd_entropy(x, order=3, delay=1, normalize=True)

        def approximate_entropy(x):
            return ant.app_entropy(x, order=2, metric="chebyshev")

        def petrosian_fd(x):
            return ant.petrosian_fd(x)

        def katz_fd(x):
            return ant.katz_fd(x)

        def higuchi_fd(x):
            return ant.higuchi_fd(x, kmax=10)

        def rootmeansquare(x):
            return np.sqrt(np.mean(x**2))

        def detrended_fluctuation(x):
            return ant.detrended_fluctuation(x)

        feature_functions = [
            mean,
            absmean,
            maximum,
            absmax,
            minimum,
            absmin,
            minplusmax,
            maxminusmin,
            curvelength,
            energy,
            nonlinear_energy,
            integral,
            stddeviation,
            variance,
            skewness,
            kurtosis,
            sum,
            spectral_entropy,
            sample_entropy,
            permutation_entropy,
            svd_entropy,
            approximate_entropy,
            petrosian_fd,
            katz_fd,
            higuchi_fd,
            rootmeansquare,
            detrended_fluctuation,
        ]

        features_names = [f"{func.__name__}" for func in feature_functions]

        return feature_functions, features_names
