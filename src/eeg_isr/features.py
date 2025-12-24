"""
features.py
Feature Utility scripts
"""

from typing import Callable

import antropy as ant
import numpy as np
from scipy import integrate, stats


class FeatureFunctions:
    """Feature functions to be used in the feature extraction process."""

    def __init__(self, sampling_freq: float) -> None:
        self.sampling_freq = float(sampling_freq)

    def get(self) -> tuple[list[Callable[[np.ndarray], float]], list[str]]:
        """Returns a list of feature functions to be used in the feature extraction process."""

        def mean(x: np.ndarray) -> float:
            return float(np.mean(x))

        def absmean(x: np.ndarray) -> float:
            return float(np.mean(np.abs(x)))

        def maximum(x: np.ndarray) -> float:
            return float(np.max(x))

        def absmax(x: np.ndarray) -> float:
            return float(np.max(np.abs(x)))

        def minimum(x: np.ndarray) -> float:
            return float(np.min(x))

        def absmin(x: np.ndarray) -> float:
            return float(np.min(np.abs(x)))

        def minplusmax(x: np.ndarray) -> float:
            return float(np.max(x) + np.min(x))

        def maxminusmin(x: np.ndarray) -> float:
            return float(np.max(x) - np.min(x))

        def curvelength(x: np.ndarray) -> float:
            cl = 0
            for i in range(x.shape[0] - 1):
                cl += abs(x[i] - x[i + 1])
            return float(cl)

        def energy(x: np.ndarray) -> float:
            return float(np.sum(np.multiply(x, x)))

        def nonlinear_energy(x: np.ndarray) -> float:
            # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
            x_squared = x[1:-1] ** 2
            subtrahend = x[2:] * x[:-2]
            return float(np.sum(x_squared - subtrahend))

        def spectral_entropy(x: np.ndarray) -> float:
            return float(
                ant.spectral_entropy(
                    x,
                    self.sampling_freq,
                    method="welch",
                    normalize=True,
                    nperseg=len(x),
                )
            )

        def integral(x: np.ndarray) -> float:
            return float(integrate.simps(x))

        def stddeviation(x: np.ndarray) -> float:
            return float(np.std(x))

        def variance(x: np.ndarray) -> float:
            return float(np.var(x))

        def skewness(x: np.ndarray) -> float:
            return float(stats.skew(x))

        def kurtosis(x: np.ndarray) -> float:
            return float(stats.kurtosis(x))

        def sum(x: np.ndarray) -> float:
            return float(np.sum(x))

        def sample_entropy(x: np.ndarray) -> float:
            return float(ant.sample_entropy(x, order=2, metric="chebyshev"))

        def permutation_entropy(x: np.ndarray) -> float:
            return float(ant.perm_entropy(x, order=3, normalize=True))

        def svd_entropy(x: np.ndarray) -> float:
            return float(ant.svd_entropy(x, order=3, delay=1, normalize=True))

        def approximate_entropy(x: np.ndarray) -> float:
            return float(ant.app_entropy(x, order=2, metric="chebyshev"))

        def petrosian_fd(x: np.ndarray) -> float:
            return float(ant.petrosian_fd(x))

        def katz_fd(x: np.ndarray) -> float:
            return float(ant.katz_fd(x))

        def higuchi_fd(x: np.ndarray) -> float:
            return float(ant.higuchi_fd(x, kmax=10))

        def rootmeansquare(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x**2)))

        def detrended_fluctuation(x: np.ndarray) -> float:
            return float(ant.detrended_fluctuation(x))

        feature_functions: list[Callable[[np.ndarray], float]] = [
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

        features_names: list[str] = [f"{func.__name__}" for func in feature_functions]

        return feature_functions, features_names
