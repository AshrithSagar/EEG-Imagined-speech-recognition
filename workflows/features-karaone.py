"""
features-karaone.py
Extraction of features from KaraOne dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config, line_separator
from utils.karaone import KaraOneDataLoader


if __name__ == "__main__":
    args = load_config(key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=args["raw_data_dir"], subjects=args["subjects"], verbose=True
    )

    karaone.process_raw_data(
        save_dir=args["filtered_data_dir"],
        pick_channels=[-1],
        l_freq=0.5,
        h_freq=50.0,
        overwrite=False,
        verbose=True,
    )

    karaone.process_epochs(epoch_type="thinking")
    karaone.epochs_info(verbose=True)
    labels = karaone.all_epoch_labels

    karaone.extract_features(
        save_dir=args["features_dir"],
        epoch_type="thinking",
        length_factor=0.1,
        overlap=0.5,
    )

    features = karaone.load_features(epoch_type="thinking", verbose=True)

    flattened_features, flattened_labels = karaone.flatten(
        features, labels, verbose=True
    )
