"""
tfr_dataset-create.py
"""
import numpy as np
from rich.console import Console

from utils.config import line_separator
from utils.karaone import KaraOneDataLoader, subjects
from utils.tfr_dataset import TFRDataset


if __name__ == "__main__":
    console = Console()

    karaone = KaraOneDataLoader(
        data_folder="/Users/ashrith/Documents/Hons/Datasets/KaraOne/EEG_data/",
        console=console,
    )

    pick_channels = [
        "FC6",
        "FT8",
        "C5",
        "CP3",
        "P3",
        "T7",
        "CP5",
        "C3",
        "CP1",
        "C4",
    ]

    for subject in subjects[:]:
        karaone.load_data(subject=subject, pick_channels=pick_channels)
        karaone.apply_bandpass_filter(l_freq=0.5, h_freq=50.0)
        karaone.make_epochs(sampling_freq=1000)
        karaone.apply_baseline_correction(baseline=(0, 0))

        tfr_ds = TFRDataset(
            data=karaone,
            dataset_dir="/Users/ashrith/Documents/Hons/Datasets/KaraOne/TFR_dataset",
            console=console,
        )
        tfr_data, epoch_labels = tfr_ds.create(
            freq_bins=256, timestamps=np.arange(0, 4900, 4900 // 256)
        )
        tfr_ds.save_dataset_based_on_subjects(verbose=False)
        tfr_ds.reshape_based_on_channels(verbose=False)
        line_separator(line="thick")

    tfr_ds.directory_info(filter="ch")
