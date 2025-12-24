"""
tfr.py
Time Frequency Representation utility class
"""

import os

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from tftb.processing import smoothed_pseudo_wigner_ville


class TFRDataset:
    """
    Time Frequency Representation using Smoothed Pseudo Wigner-Ville Distribution (SPWVD)
    """

    def __init__(self, dataset_dir, data=None, console=None) -> None:
        self.console = console if console else Console()
        self.data = data  # KaraOneDataLoader instance
        self.dataset_dir = os.path.join(dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def create(self, freq_bins=None, timestamps=None):
        self.get_labels()
        self.num_classes = len(self.labels)
        epochs = self.data.epochs.get_data(copy=True)
        self.num_epochs, self.num_channels, _ = epochs.shape
        tfr_shape = (freq_bins, len(timestamps))
        tfr_data = np.zeros((self.num_epochs, self.num_channels, *tfr_shape))

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            epoch_task = progress.add_task(
                "[cyan]Processing epochs...",
                total=self.num_epochs,
            )
            channel_task = progress.add_task(
                "[magenta]Processing channels...",
                total=self.num_channels,
            )
            epochs = self.data.epochs.get_data(copy=True)
            for epoch_index, epoch_data in enumerate(epochs):
                progress.update(epoch_task, advance=1)
                for channel_index, channel in enumerate(self.data.epochs.ch_names):
                    progress.update(channel_task, advance=1)

                    # tfr = np.random.rand(*tfr_shape)
                    tfr = smoothed_pseudo_wigner_ville(
                        signal=epoch_data[channel_index],
                        freq_bins=freq_bins,
                        timestamps=timestamps,
                    )
                    tfr_data[epoch_index, channel_index] = tfr
                progress.reset(channel_task)

        self.get_epoch_labels()
        self.tfr_data = tfr_data

        return self.tfr_data, self.epoch_labels

    def get_labels(self):
        self.labels = [
            key
            for key, _ in sorted(
                self.data.epochs.event_id.items(),
                key=lambda x: x[1],
            )
        ]
        return self.labels

    def get_epoch_labels(self):
        self.epoch_labels = [
            self.data.epochs.events[ind][2] - 1 for ind in range(self.num_epochs)
        ]
        return self.epoch_labels

    def save_dataset_based_on_subjects(self, verbose=False):
        def create_class_folder(label):
            class_name = label.replace("/", "|")
            class_folder = os.path.join(self.dataset_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            return class_name, class_folder

        def get_label_inds(label_index):
            return [i for i, x in enumerate(self.epoch_labels) if x == label_index]

        for label_index, label in enumerate(self.labels):
            class_name, class_folder = create_class_folder(label)
            label_inds = get_label_inds(label_index)

            class_tfr = self.tfr_data[label_inds]
            class_tfr_filename = f"sb{self.data.subject}.npy"
            np.save(os.path.join(class_folder, class_tfr_filename), class_tfr)
            if verbose:
                self.console.print(
                    f"Saved: [purple]{class_name}[/]/\t{class_tfr_filename}\t{class_tfr.shape}"
                )

    def reshape_based_on_channels(self, verbose=False):
        def get_class_folder(label):
            class_name = label.replace("/", "|")
            class_folder = os.path.join(self.dataset_dir, class_name)
            return class_name, class_folder

        for label in self.labels:
            class_name, class_folder = get_class_folder(label)
            channel_data = {channel: [] for channel in self.data.channels}

            for subject_file in sorted(os.listdir(class_folder)):
                if subject_file.endswith(".npy") and subject_file.startswith("sb"):
                    subject_data = np.load(os.path.join(class_folder, subject_file))
                    for channel_index, channel in enumerate(self.data.channels):
                        channel_index_data = subject_data[:, channel_index, :, :]
                        channel_data[channel].append(channel_index_data)

            for channel_index, channel in enumerate(self.data.channels):
                channel_data_filename = f"ch{channel}.npy"
                channel_data_file = os.path.join(class_folder, channel_data_filename)
                class_channel_data = np.concatenate(channel_data[channel])
                np.save(channel_data_file, class_channel_data)
                if verbose:
                    self.console.print(
                        f"Saved: [purple]{class_name}[/]/\t{channel_data_filename} \t{class_channel_data.shape}"
                    )

    def directory_info(self, filter=""):
        def print_tree(directory, indent=""):
            items = sorted(os.listdir(directory))
            if filter:
                filtered_items = [
                    item
                    for item in items
                    if os.path.isdir(os.path.join(directory, item))
                    or (item.endswith(".npy") and filter in item)
                ]
            else:
                filtered_items = items

            output = ""
            for index, item in enumerate(filtered_items):
                item_path = os.path.join(directory, item)
                is_last_item = index == len(filtered_items) - 1

                if os.path.isdir(item_path):
                    output += f"{indent}{'└── ' if is_last_item else '├── '}Class: [purple]{item}[/]\n"
                    next_indent = f"{indent}{'    ' if is_last_item else '│   '}"
                    output += print_tree(item_path, next_indent)
                elif item.endswith(".npy") and (not filter or filter in item):
                    npy_data = np.load(item_path)
                    output += f"{indent}{'└── ' if is_last_item else '├── '}File: {item}, \t Shape: {npy_data.shape}\n"

            return output

        if filter and filter not in ["sb", "ch"]:
            self.console.print("[red]ERROR[/]: Unknown filter specified")
            return

        result = print_tree(self.dataset_dir)
        self.console.print(
            f"{os.path.basename(self.dataset_dir)}/\n{result}",
            end="",
            style="reset",
        )

    def load(self, channel, verbose=False):
        self.channel = channel
        self.classes = [
            class_name
            for class_name in sorted(
                os.listdir(self.dataset_dir), key=lambda x: x.replace("|", "/")
            )
            if os.path.isdir(os.path.join(self.dataset_dir, class_name))
        ]
        dataset = [[]] * len(self.classes)

        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)

            tfr_data = []
            if verbose:
                self.console.print(Rule())
                self.console.print(f"Class: [purple]{class_name}[/]")

            for file_name in os.listdir(class_dir):
                file = os.path.join(class_dir, file_name)

                if file_name == f"ch{channel}.npy":
                    channel_tfr_data = np.load(file)[:, :, :-2]
                    tfr_data.append(channel_tfr_data)
                    if verbose:
                        self.console.print(file_name, channel_tfr_data.shape)

            dataset[class_index] = tfr_data
        dataset = np.squeeze(np.array(dataset, dtype=object))

        self.class_labels = np.array(
            [
                class_index
                for class_index, _ in enumerate(self.classes)
                for _ in dataset[class_index]
            ]
        )
        self.dataset = np.concatenate(dataset, axis=0)

    def dataset_info(self):
        self.console.print(f"Channel: [purple]{self.channel}[/]")
        self.console.print(f"Classes: {self.classes}")
        self.console.print(f"Dataset shape: {self.dataset.shape}")
        self.console.print(f"Class labels shape: {self.class_labels.shape}")

    def split_info(self, x_train, x_test, y_train, y_test):
        self.console.print(f"Train dataset: {x_train.shape}")
        self.console.print(f"Test dataset: {x_test.shape}")
