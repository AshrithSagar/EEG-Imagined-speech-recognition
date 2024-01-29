"""
ifs.py
Information Set Theory Utility scripts
"""
import os
import numpy as np
from rich.console import Console

from utils.config import line_separator


class InformationSet:
    def __init__(self, Set, console=None, verbose=False):
        self.Set = Set
        self.verbose = verbose
        self.console = console if console else Console()

    def extract_effective_information(self, verbose=None):
        """
        Extracts the effective information from the information set
        (n.epochs, n.windows, n.channels, n.features)
        """
        verbose = verbose if verbose is not None else self.verbose

        self.features = []
        for information_source_matrix in self.Set:
            first_fold_information = self.compute_information_values(
                information_source_matrix, function="gaussian", axis=0
            )
            second_fold_information = self.compute_information_values(
                information_source_matrix, function="gaussian", axis=1
            )
            two_fold_information = first_fold_information + second_fold_information
            effective_information = np.mean(two_fold_information, axis=(0, 1))
            self.features.append(effective_information)

        self.features = np.asarray(self.features)
        return self.features

    def compute_information_values(self, information_source_matrix, function, axis):
        membership_function = self.get_function(function, axis)
        membership_values = membership_function(information_source_matrix, axis=axis)
        information_values = information_source_matrix * membership_values
        return information_values

    def get_function(self, function=None, axis=None):
        def gaussian(x, axis):
            mean_x = np.mean(x, axis=axis, keepdims=True)
            std_x = np.std(x, axis=axis, keepdims=True)

            # Replace zero standard deviations with 1 to avoid division by zero
            std_x[std_x == 0] = 1

            z = (x - mean_x) / std_x
            return np.exp(-(z**2) / 2.0)

        if isinstance(function, str):
            if function == "gaussian":
                function = gaussian
            else:
                raise NotImplementedError(f"Function '{function}' not implemented")

        return function
