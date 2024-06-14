"""
features-feis.py
Extraction of features from FEIS dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.feis import FEISDataLoader


def main():
    d_args = load_config(config_file="config.yaml", key="feis")

    feis = FEISDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )
    feis.unzip_data_eeg(delete_zip=False)

    feis.extract_features(
        save_dir=d_args["features_dir"],
        epoch_type=d_args["epoch_type"],
        skip_if_exists=True,
    )

    features = feis.load_features(epoch_type=d_args["epoch_type"])

    feis.extract_labels()
    labels = feis.load_labels()

    flattened_features, flattened_labels = feis.flatten(features, labels, verbose=True)


if __name__ == "__main__":
    main()
