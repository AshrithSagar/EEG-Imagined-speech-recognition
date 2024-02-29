"""
test_ifs.py
"""

import os
import sys

import numpy as np
from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import line_separator
from utils.ifs import InformationSet


def test_extract_effective_information(verbose=True):

    np.random.seed(seed=42)

    # Mock dataset with n.epochs, n.windows, n.channels, n.features_per_window
    features = np.random.rand(5, 3, 2, 4)
    features_ifs = InformationSet(features)

    eff_features = features_ifs.extract_effective_information()

    assert isinstance(eff_features, np.ndarray), "Output should be a NumPy array"
    assert eff_features.shape == (
        5,
        4,
    ), "Output shape should be (n_epochs, n_features)"
    assert np.all(
        np.isfinite(eff_features)
    ), "Effective information values should be finite"

    if verbose:
        console = Console()
        console.print("Features:\n", features)
        line_separator(console)
        console.print("Effective Features:\n", eff_features)
        line_separator(console)


test_extract_effective_information(verbose=True)
