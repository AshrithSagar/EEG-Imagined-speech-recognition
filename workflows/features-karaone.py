"""
features-karaone.py
Extraction of features from KaraOne dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import fetch_select, line_separator, load_config


if __name__ == "__main__":
    args = load_config(config_file="config.yaml", key="karaone")
    dataset = fetch_select("dataset", "KaraOne")

    dset = dataset(
        raw_data_dir=args["raw_data_dir"],
        subjects=args["subjects"],
        verbose=True,
    )

    dset.process_raw_data(
        save_dir=args["filtered_data_dir"],
        pick_channels=[-1],
        l_freq=0.5,
        h_freq=50.0,
        overwrite=False,
        verbose=True,
    )

    dset.process_epochs(epoch_type=args["epoch_type"])
    dset.epochs_info(verbose=True)
    labels = dset.all_epoch_labels

    dset.extract_features(
        save_dir=args["features_dir"],
        epoch_type=args["epoch_type"],
        length_factor=0.1,
        overlap=0.5,
    )

    features = dset.load_features(epoch_type=args["epoch_type"], verbose=True)

    flattened_features, flattened_labels = dset.flatten(features, labels, verbose=True)
