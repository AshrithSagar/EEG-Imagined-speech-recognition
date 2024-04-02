"""
features-dset.py
Extraction of features from dset dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import fetch_select, line_separator, load_config


if __name__ == "__main__":
    d_args = load_config(config_file="config.yaml", key="feis")
    dataset = fetch_select("dataset", "FEIS")

    dset = dataset(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )
    dset.unzip_data_eeg(delete_zip=False)

    dset.extract_features(
        save_dir=d_args["features_dir"],
        epoch_type=d_args["epoch_type"],
    )

    features = dset.load_features(epoch_type=d_args["epoch_type"])

    dset.extract_labels()
    labels = dset.load_labels()

    flattened_features, flattened_labels = dset.flatten(features, labels, verbose=True)
