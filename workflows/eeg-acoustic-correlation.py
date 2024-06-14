"""
eeg-acoustic-correlation.py
Compute the mean pearson correlations between Acoustic and EEG features for each channel.
Between the acoustic features and each of the channels over all imagined speech segments.
"""

import os
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr

sys.path.append(os.getcwd())
from utils.config import fetch_select, load_config, save_console, setup_parser


def main(config_file="config.yaml"):
    args = load_config(config_file)

    dataset_name = args.get("_select").get("dataset")
    dataset = fetch_select("dataset", dataset_name)
    d_args = args[dataset_name.lower()]
    console = Console(record=True)

    dset = dataset(
        raw_data_dir=d_args["raw_data_dir"],
        subjects=d_args["subjects"],
        verbose=True,
        console=console,
    )

    acoustic_features = dset.load_features(
        features_dir=d_args["features_dir"], epoch_type="acoustic"
    )
    acoustic_features, _ = dset.flatten(acoustic_features)

    eeg_features = dset.load_features(
        features_dir=d_args["features_dir"], epoch_type="thinking"
    )
    eeg_features, labels = dset.flatten(eeg_features)

    channel_correlations = []
    num_channels = eeg_features.shape[2]
    for acoustic_feats, eeg_feats in zip(acoustic_features, eeg_features):
        acoustic_feats = acoustic_feats.transpose((1, 0, 2)).reshape(-1)
        eeg_feats = eeg_feats.transpose((1, 0, 2)).reshape(num_channels, -1)

        channel_correlations.append(
            [pearsonr(acoustic_feats, channel_feats) for channel_feats in eeg_feats]
        )
    channel_correlations = np.array(channel_correlations)
    mean_correlations = channel_correlations.mean(axis=0)

    all_df = pd.DataFrame(
        {
            "Channel": dset.channels,
            "Pearson r": mean_correlations[:, 0],
            "p-Value": mean_correlations[:, 1],
        }
    )
    filename = os.path.join(d_args["features_dir"], "eeg-acoustic-correlation.csv")
    all_df.to_csv(filename, index=False)
    top_df = all_df.nlargest(10, "Pearson r")

    table = Table(title="[bold underline]Mean correlations:[/]")
    table.add_column("Channel", justify="right", style="magenta", no_wrap=True)
    table.add_column("Pearson r", justify="center", style="cyan", no_wrap=True)
    table.add_column("p-Value", justify="center", style="cyan", no_wrap=True)
    for _, row in top_df.iterrows():
        table.add_row(
            row["Channel"], f"{row['Pearson r']:.4f}", f"{row['p-Value']:.4f}"
        )
    console.print(table)

    file = os.path.join(d_args["features_dir"], "eeg-acoustic-correlation.txt")
    save_console(console, file)


if __name__ == "__main__":
    args = setup_parser(
        "Compute the mean pearson correlations between Acoustic and EEG features for each channel."
    )
    main(args.config)
