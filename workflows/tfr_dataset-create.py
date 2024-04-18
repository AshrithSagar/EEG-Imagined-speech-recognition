"""
tfr_dataset-create.py
"""

import os
import sys

import numpy as np
from rich.console import Console

sys.path.append(os.getcwd())
from utils.config import line_separator, load_config
from utils.karaone import KaraOneDataLoader
from utils.tfr import TFRDataset


if __name__ == "__main__":
    console = Console()
    d_args = load_config(config_file="config.yaml", key="karaone")

    karaone = KaraOneDataLoader(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
    )

    karaone.process_raw_data(
        save_dir=d_args["filtered_data_dir"],
        pick_channels=d_args["channels"],
        l_freq=0.5,
        h_freq=50.0,
        overwrite=False,
        verbose=True,
    )
    karaone.process_epochs(
        epoch_type=d_args["epoch_type"],
        pick_channels=d_args["channels"],
    )
    karaone.epochs_info(verbose=True)
    labels = karaone.all_epoch_labels
    line_separator(line="thick")

    for subject in d_args["subjects"]:
        console.print(f"[bold]Subject: {subject}[/bold]")

        tfr_ds = TFRDataset(
            dataset_dir=d_args["tfr_dataset_dir"],
            data=karaone,
            console=console,
        )
        tfr_data, epoch_labels = tfr_ds.create(
            freq_bins=256, timestamps=np.arange(0, 4900, 4900 // 256)
        )
        tfr_ds.save_dataset_based_on_subjects(verbose=False)
        tfr_ds.reshape_based_on_channels(verbose=False)
        line_separator(line="thick")

    tfr_ds.directory_info(filter="ch")
