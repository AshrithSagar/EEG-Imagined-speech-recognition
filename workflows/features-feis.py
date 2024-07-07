"""
features-feis.py
Extraction of features from FEIS dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import Config, setup_parser
from utils.feis import FEISDataLoader


def main(config_file="config.yaml"):
    d_args = Config(config_file)["feis"]

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
    args = setup_parser("Extract features from FEIS dataset")
    main(args.config)
