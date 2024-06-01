"""
features-karaone.py
Extraction of features from KaraOne dataset.
"""

import os
import sys

sys.path.append(os.getcwd())
from utils.config import load_config
from utils.karaone import KaraOneDataLoader

if __name__ == "__main__":
    d_args = load_config(config_file="config.yaml", key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )

    if d_args["epoch_type"] == "acoustic":
        karaone.load_audio_data()
        labels = karaone.get_all_epoch_labels()

    elif d_args["epoch_type"] in ["clearing", "thinking", "stimuli", "speaking"]:
        karaone.process_raw_data(
            save_dir=d_args["filtered_data_dir"],
            pick_channels=d_args["channels"],
            l_freq=0.5,
            h_freq=50.0,
            overwrite=False,
            verbose=True,
        )

        karaone.process_epochs(epoch_type=d_args["epoch_type"])
        karaone.epochs_info(verbose=True)
        labels = karaone.all_epoch_labels

    else:
        raise ValueError(
            "Invalid epoch type. Choose from 'acoustic', 'clearing', 'thinking', 'stimuli', 'speaking'."
        )

    karaone.extract_features(
        save_dir=d_args["features_dir"],
        epoch_type=d_args["epoch_type"],
        length_factor=d_args["length_factor"],
        overlap=d_args["overlap"],
        skip_if_exists=True,
    )

    features = karaone.load_features(epoch_type=d_args["epoch_type"], verbose=True)

    flattened_features, flattened_labels = karaone.flatten(
        features, labels, verbose=True
    )
