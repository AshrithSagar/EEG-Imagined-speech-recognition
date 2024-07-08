"""
ifs.py
Information Set Theory Utility scripts
"""

from typing import Callable, Optional, Union

import numpy as np
from rich.console import Console


class InformationSet:
    def __init__(
        self,
        Set: np.ndarray,
        console: Optional[Console] = None,
        verbose: bool = False,
    ) -> None:
        self.Set: np.ndarray = Set
        self.verbose: bool = verbose
        self.console: Console = console if console else Console()
        if verbose:
            self.console.rule(title="[bold blue3][Information Set][/]", style="blue3")

    def extract_effective_information(
        self, verbose: Optional[bool] = None
    ) -> np.ndarray:
        """
        Extracts the effective information from the information set
        (n.epochs, n.windows, n.channels, n.features)
        """
        verbose = verbose if verbose is not None else self.verbose

        effective_information = [
            self.compute_effective_information(information_source_matrix)
            for information_source_matrix in self.Set
        ]

        self.information: np.ndarray = np.asarray(effective_information)
        return self.information

    def compute_effective_information(
        self, information_source_matrix: np.ndarray, verbose: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute the effective information on the information set
        (n.windows, n.channels, n.features)
        """
        verbose = verbose if verbose is not None else self.verbose

        temporal_fold_information = self.compute_information_values(
            information_source_matrix, function="gaussian", axis=0
        )
        spatial_fold_information = self.compute_information_values(
            information_source_matrix, function="gaussian", axis=1
        )
        fusion_information = temporal_fold_information + spatial_fold_information
        effective_information = np.mean(fusion_information, axis=(0, 1))
        return effective_information

    def compute_information_values(
        self, information_source_matrix: np.ndarray, function: str, axis: int
    ) -> np.ndarray:
        membership_function = self.get_function(function, axis)
        membership_values = membership_function(information_source_matrix, axis=axis)
        information_values = information_source_matrix * membership_values
        return information_values

    def get_function(
        self, function: Union[str, Callable], axis: Optional[int] = None
    ) -> Callable:
        def gaussian(x: np.ndarray, axis: int) -> np.ndarray:
            mean_x = np.mean(x, axis=axis, keepdims=True)
            std_x = np.std(x, axis=axis, keepdims=True)

            # Replace zero standard deviations with 1 to avoid division by zero
            std_x[std_x == 0] = 1

            z = (x - mean_x) / std_x
            return np.exp(-(z**2) / 2.0)

        if isinstance(function, str):
            if function in ["gaussian"]:
                function = locals()[function]
            else:
                raise NotImplementedError(f"Function '{function}' not implemented")

        return function
