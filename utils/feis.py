"""
feis.py
FEIS Dataset Utility scripts
"""

import os
import subprocess

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from utils.dataset import DatasetLoader
from utils.features import FeatureFunctions

# Subjects: 01-21 and "chinese-1", "chinese-2"
all_subjects = [
    str(i).zfill(2) if i <= 21 else f"chinese-{i - 21}" for i in range(1, 24)
]

channels = [
    "F3",
    "FC5",
    "AF3",
    "F7",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "F8",
    "AF4",
    "FC6",
    "F4",
]

labels = [
    "/i/",
    "/u:/",
    "/æ/",
    "/ɔ:/",
    "/m/",
    "/n/",
    "/ŋ/",
    "/f/",
    "/s/",
    "/ʃ/",
    "/v/",
    "/z/",
    "/ʒ/",
    "/p",
    "/t/",
    "/k/",
]


class FEISDataLoader(DatasetLoader):
    """FEIS Utility class"""

    def __init__(
        self,
        raw_data_dir,
        subjects="all",
        sampling_freq=256,
        num_seconds_per_trial=5,
        console=None,
        verbose=False,
    ):
        """Parameters:
        - raw_data_dir (str): Path to the data folder.
        - subjects (list): List of subjects to load. Use "all" for all subjects (default)
        - sampling_freq (int): Sampling frequency for EEG data (default: 256 Hz).
        - num_seconds_per_trial (int): Number of seconds per trial (default: 5 seconds).
        """
        super().__init__(raw_data_dir, subjects, all_subjects, console, verbose)
        self.data_dir = raw_data_dir
        self.sampling_freq = sampling_freq
        self.num_seconds_per_trial = num_seconds_per_trial
        self.progress = None
        if verbose:
            self.console.rule(title="[bold blue3][FEIS Dataset][/]", style="blue3")
            self.subjects_info()

    def unzip_data_eeg(self, delete_zip=False):
        with self.create_progress_bar() as self.progress:
            task_subjects = self.progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )

            for subject in self.subjects:
                subject_dir = os.path.join(self.data_dir, subject)

                for file in sorted(os.listdir(subject_dir)):
                    if file.endswith(".zip"):
                        csv_file = os.path.splitext(file)[0] + ".csv"
                        csv_path = os.path.join(subject_dir, csv_file)

                        if not os.path.exists(csv_path):
                            command = [
                                "unzip",
                                os.path.join(subject_dir, file),
                                "-d",
                                subject_dir,
                            ]
                            result = subprocess.run(
                                command,
                                capture_output=True,
                                text=True,
                            )
                            if result.stderr:
                                self.console.print(result.stderr)

                        if delete_zip:
                            os.remove(os.path.join(subject_dir, file))
                self.progress.update(task_subjects, advance=1)

    def extract_labels(
        self, subject=None, epoch_type: str = "speaking", features_dir=None
    ):
        features_dir = features_dir or self.features_dir
        subject = subject or self.subjects[0]
        file = os.path.join(self.data_dir, subject, f"{epoch_type}.csv")

        df = pd.read_csv(file, header=None, skiprows=range(1, 1280), usecols=[16])
        eeg_labels = df.values.flatten()
        self.labels = eeg_labels[1::1280]

        self.save_labels(features_dir=features_dir)
        return self.labels

    def extract_features(self, save_dir, epoch_type: str, skip_if_exists=True):
        """Parameters:
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").
        """
        with self.create_progress_bar() as self.progress:
            task_subjects = self.progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )
            task_features = self.progress.add_task("Computing features ...")

            self.epoch_type = epoch_type
            self.features_dir = save_dir
            self.get_features_functions()

            for subject in self.subjects:
                if skip_if_exists:
                    if os.path.exists(
                        os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
                    ):
                        self.progress.update(task_subjects, advance=1)
                        continue

                data = self.load_data_eeg(subject, epoch_type)
                epochs = self.make_epochs(data)
                self.progress.update(task_features, total=len(epochs))
                features = self.compute_features(epochs, task=task_features)
                self.progress.reset(task_features)
                self.save_features(subject, features)
                self.progress.update(task_subjects, advance=1)

    def load_data_eeg(self, subject, epoch_type):
        file = os.path.join(self.data_dir, subject, f"{epoch_type}.csv")
        subject_data = pd.read_csv(
            file, header=None, skiprows=1, usecols=range(2, 16), dtype=np.float32
        )
        return subject_data.values

    def make_epochs(self, data):
        num_epochs = len(data) / self.sampling_freq / self.num_seconds_per_trial
        epochs = np.split(data, num_epochs)
        epochs = np.asarray(epochs, dtype=np.float32)
        return epochs

    def compute_features(self, epochs, task=None):
        features = []
        for epoch in epochs:
            windowed_epoch = self.window_data(epoch, split=10)
            feats = self.make_simple_feats(windowed_epoch, flatten=False)
            all_feats = self.add_deltas(feats)
            features.append(all_feats)
            if task:
                self.progress.update(task, advance=1)

        return np.asarray(features, dtype=np.float32)

    def window_data(self, data: np.ndarray, split: int = 10):
        """Windows the data with a stride length of 1."""
        w_len = data.shape[0] // split
        stride = w_len // 2
        no_offset_windows = np.split(data, split)
        offset_windows = np.split(data[stride:-stride], split - 1)
        windows = [0] * (2 * split - 1)
        windows[::2] = no_offset_windows
        windows[1::2] = offset_windows
        windows = np.array(windows, dtype=np.float32)
        return windows

    def make_simple_feats(self, windowed_data: np.ndarray, flatten: bool = True):
        feats = [self.features_per_window(window, flatten) for window in windowed_data]
        return np.asarray(feats, dtype=np.float32)

    def features_per_window(self, window: np.ndarray, flatten: bool = True):
        """
        Takes a single window, returns an array of features of shape
        (n.features, electrodes), and then flattens it into a vector
        """
        outvec = np.zeros((len(self.feature_functions), window.shape[1]))
        for i in range(len(self.feature_functions)):
            for j in range(window.shape[1]):
                outvec[i, j] = self.feature_functions[i](window[:, j])

        outvec = outvec.transpose()
        if flatten:
            outvec = outvec.reshape(-1)

        return outvec

    def add_deltas(self, feats_array: np.ndarray):
        deltas = np.diff(feats_array, axis=0)
        double_deltas = np.diff(deltas, axis=0)
        # all_feats = np.hstack((feats_array[2:], deltas[1:], double_deltas))
        all_feats = np.concatenate((feats_array, deltas, double_deltas), axis=0)
        return all_feats

    def save_features(self, subject: str, features: np.ndarray):
        subject_features_dir = os.path.join(self.features_dir, subject)
        os.makedirs(subject_features_dir, exist_ok=True)

        filename = os.path.join(subject_features_dir, f"{self.epoch_type}.npy")
        np.save(filename, features)

    def get_features_functions(self):
        ff = FeatureFunctions(self.sampling_freq)
        self.feature_functions, features_names = ff.get()
        self.features_names = [
            f"{prefix}{feat_name}"
            for prefix in ["", "d_", "dd_"]
            for feat_name in features_names
        ]

    def load_features(self, features_dir=None, epoch_type: str = None, verbose=None):
        """Parameters:
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").

        Returns:
        - features (np.ndarray): Features of shape (n.subjects, n.epochs, n.windows, n.features_per_window).
        """
        self.features = []
        self.features_dir = features_dir or self.features_dir
        epoch_type = epoch_type or self.epoch_type
        verbose = verbose if verbose is not None else self.verbose
        self.get_features_functions()

        for subject in self.subjects:
            filename = os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
            subject_features = np.load(filename)
            self.features.append(subject_features)

        if verbose:
            labels = self.labels if "labels" in self.__dict__ else self.extract_labels()
            self.features_info(self.features, labels, verbose=verbose)

        return self.features

    def save_labels(self, labels=None, filename="labels.npy", features_dir=None):
        labels = labels or self.labels
        features_dir = features_dir or self.features_dir

        file = os.path.join(features_dir, filename)
        np.save(file, labels)

    def load_labels(self, filename="labels.npy"):
        """Returns:
        - labels (np.ndarray): Labels of shape (n.epochs, ).
        """
        file = os.path.join(self.features_dir, filename)
        labels = np.load(file, allow_pickle=True)
        return labels

    def flatten(self, features=None, labels=None, reshape=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        features = features if features is not None else self.features
        labels = labels if labels is not None else self.extract_labels()

        flattened_features = (
            [feats.reshape(feats.shape[0], -1) for feats in features]
            if reshape
            else features
        )
        flattened_features = np.vstack(flattened_features)
        if not reshape:
            flattened_features = np.transpose(flattened_features, (0, 1, 3, 2))

        flattened_labels = np.tile(labels, len(features))

        if verbose:
            self.dataset_info(flattened_features, flattened_labels, verbose=verbose)

        return flattened_features, flattened_labels

    def create_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        )

    def subjects_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            message = "[bold underline]Subjects:[/]\n"
            message += ", ".join(self.subjects)
            self.console.print(message)

    def dataset_info(self, features=None, labels=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        features = features if features is not None else self.features
        labels = labels if labels is not None else self.extract_labels()

        if verbose:
            table = Table(title="[bold underline]Dataset Info[/]")
            table.add_column("Data", justify="right", no_wrap=True)
            table.add_column("Shape", style="cyan", no_wrap=True)
            table.add_row("Features", str(features.shape))
            table.add_row("Labels", str(labels.shape))

            self.console.print(table)

    def features_info(self, features, labels, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            table = Table(title="[bold underline]Features Info[/]")
            table.add_column("Subject", justify="right", style="magenta", no_wrap=True)
            table.add_column("Features", justify="center", style="cyan", no_wrap=True)
            table.add_column("Labels", style="cyan", no_wrap=True)

            for feats, subject in zip(features, self.subjects):
                table.add_row(str(subject), str(feats.shape), str(labels.shape))

            self.console.print(table)

    def get_task(self, labels, task=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        labels = labels if labels is not None else self.extract_labels()

        # Task-0: [±voice]
        voiceless = {"p", "t", "k", "f", "s", "sh"}
        voiced = {"v", "z", "zh", "m", "n", "ng"}

        task_labels = np.zeros_like(labels, dtype=np.int8)
        for index, label in enumerate(labels):
            if label in voiced:
                task_labels[index] = 1
            elif label in voiceless:
                task_labels[index] = 0
            else:
                task_labels[index] = -1

        return task_labels
