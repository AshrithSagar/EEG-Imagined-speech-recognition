"""
feis.py
FEIS Dataset Utility scripts
"""
import os
import subprocess
import numpy as np
import pandas as pd
from scipy import integrate, stats
import antropy
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from utils.config import line_separator


# Subjects: 01-21 and "chinese-1", "chinese-2"
all_subjects = [str(i).zfill(2) if i <= 21 else f"chinese-{i-21}" for i in range(1, 24)]

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


class FEISDataLoader:
    """
    Load data from FEIS folder
    """

    def __init__(
        self,
        data_dir,
        subjects="all",
        sampling_freq=256,
        num_seconds_per_trial=5,
        console=None,
    ):
        """Parameters:
        - data_dir (str): Path to the data folder.
        - subjects (list): List of subjects to load. Use "all" for all subjects (default)
        - sampling_freq (int): Sampling frequency for EEG data (default: 256 Hz).
        - num_seconds_per_trial (int): Number of seconds per trial (default: 5 seconds).
        """
        self.data_dir = data_dir
        self.subjects = all_subjects if subjects == "all" else subjects
        self.sampling_freq = sampling_freq
        self.num_seconds_per_trial = num_seconds_per_trial
        self.console = console if console else Console()

    def unzip_data_eeg(self, delete_zip=False):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_subjects = progress.add_task(
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
                progress.update(task_subjects, advance=1)

    def extract_labels(self, subject=None, epoch_type: str = "speaking"):
        subject = subject or self.subjects[0]
        file = os.path.join(self.data_dir, subject, f"{epoch_type}.csv")

        df = pd.read_csv(file, header=None, skiprows=range(1, 1280), usecols=[16])
        eeg_labels = df.values.flatten()
        labels = eeg_labels[1::1280]

        self.save_labels(labels)

    def extract_features(self, features_dir, epoch_type: str, skip_if_exists=True):
        """Parameters:
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").
        """

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_subjects = progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )
            task_features = progress.add_task("Computing features ...")

            self.epoch_type = epoch_type
            self.features_dir = features_dir
            self.get_features_functions()

            for subject in self.subjects:
                if skip_if_exists:
                    if os.path.exists(
                        os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
                    ):
                        progress.update(task_subjects, advance=1)
                        continue

                data = self.load_data_eeg(subject, epoch_type)
                epochs = self.make_epochs(data)
                progress.update(task_features, total=len(epochs))
                self.compute_features(epochs, progress=progress, task=task_features)
                progress.reset(task_features)
                self.save_features(subject)
                progress.update(task_subjects, advance=1)

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

    def compute_features(self, epochs, progress=None, task=None):
        features = []
        for epoch in epochs:
            epoch = self.window_data(epoch, split=10)
            feats = self.make_simple_feats(epoch)
            feats = self.add_deltas(feats)
            features.append(feats)
            if progress:
                progress.update(task, advance=1)
        self.features = np.asarray(features, dtype=np.float32)
        return self.features

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

    def make_simple_feats(self, windowed_data: np.ndarray):
        simple_feats = []
        for w in range(len(windowed_data)):
            simple_feats.append(self.features_per_window(windowed_data[w]))
        return np.array(simple_feats)

    def features_per_window(self, window: np.ndarray):
        """
        Takes a single window, returns an array of features of shape
        (n.features, electrodes), and then flattens it into a vector
        """
        outvec = np.zeros((len(self.feature_functions), window.shape[1]))
        for i in range(len(self.feature_functions)):
            for j in range(window.shape[1]):
                outvec[i, j] = self.feature_functions[i](window[:, j])
        outvec = outvec.reshape(-1)
        return outvec

    def get_features_functions(self):
        def mean(x):
            return np.mean(x)

        def absmean(x):
            return np.mean(np.abs(x))

        def maximum(x):
            return np.max(x)

        def absmax(x):
            return np.max(np.abs(x))

        def minimum(x):
            return np.min(x)

        def absmin(x):
            return np.min(np.abs(x))

        def minplusmax(x):
            return np.max(x) + np.min(x)

        def maxminusmin(x):
            return np.max(x) - np.min(x)

        def curvelength(x):
            cl = 0
            for i in range(x.shape[0] - 1):
                cl += abs(x[i] - x[i + 1])
            return cl

        def energy(x):
            return np.sum(np.multiply(x, x))

        def nonlinear_energy(x):
            # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
            x_squared = x[1:-1] ** 2
            subtrahend = x[2:] * x[:-2]
            return np.sum(x_squared - subtrahend)

        def spec_entropy(x):
            return antropy.spectral_entropy(
                x, self.sampling_freq, method="welch", normalize=True, nperseg=len(x)
            )

        def integral(x):
            return integrate.simps(x)

        def stddeviation(x):
            return np.std(x)

        def variance(x):
            return np.var(x)

        def skew(x):
            return stats.skew(x)

        def kurtosis(x):
            return stats.kurtosis(x)

        def sample_entropy(x):
            return antropy.sample_entropy(x, order=2, metric="chebyshev")

        def perm_entropy(x):
            return antropy.perm_entropy(x, order=3, normalize=True)

        def svd_entropy(x):
            return antropy.svd_entropy(x, order=3, delay=1, normalize=True)

        def app_entropy(x):
            return antropy.app_entropy(x, order=2, metric="chebyshev")

        def petrosian(x):
            return antropy.petrosian_fd(x)

        def katz(x):
            return antropy.katz_fd(x)

        def higuchi(x):
            return antropy.higuchi_fd(x, kmax=10)

        def rootmeansquare(x):
            return np.sqrt(np.mean(x**2))

        def dfa(x):
            return antropy.detrended_fluctuation(x)

        self.feature_functions = [
            mean,
            absmean,
            maximum,
            absmax,
            minimum,
            absmin,
            minplusmax,
            maxminusmin,
            curvelength,
            energy,
            nonlinear_energy,
            integral,
            stddeviation,
            variance,
            skew,
            kurtosis,
            np.sum,
            spec_entropy,
            sample_entropy,
            perm_entropy,
            svd_entropy,
            app_entropy,
            petrosian,
            katz,
            higuchi,
            rootmeansquare,
            dfa,
        ]
        return self.feature_functions

    def add_deltas(self, feats_array: np.ndarray):
        deltas = np.diff(feats_array, axis=0)
        double_deltas = np.diff(deltas, axis=0)
        all_feats = np.hstack((feats_array[2:], deltas[1:], double_deltas))
        return all_feats

    def save_features(self, subject):
        subject_features_dir = os.path.join(self.features_dir, subject)
        os.makedirs(subject_features_dir, exist_ok=True)

        filename = os.path.join(subject_features_dir, self.epoch_type + ".npy")
        np.save(filename, self.features)

    def load_features(self, epoch_type: str = None, verbose=False):
        """Parameters:
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").

        Returns:
        - features (np.ndarray): Features of shape (n.subjects, n.epochs, n.windows, n.features_per_window).
        """
        features = []
        epoch_type = epoch_type or self.epoch_type

        for subject in self.subjects:
            filename = os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
            subject_features = np.load(filename)
            features.append(subject_features)

        if verbose:
            message = f"[bold underline]Features:[/]\n"
            message += "\n".join(
                [
                    f"{subject}: {feats.shape}"
                    for feats, subject in zip(features, self.subjects)
                ]
            )
            self.console.print(message)

        return features

    def save_labels(self, labels, filename="labels.npy"):
        file = os.path.join(self.features_dir, filename)
        np.save(file, labels)

    def load_labels(self, filename="labels.npy"):
        """Returns:
        - labels (np.ndarray): Labels of shape (n.epochs, ).
        """
        file = os.path.join(self.features_dir, filename)
        labels = np.load(file, allow_pickle=True)
        return labels

    def flatten(self, features, labels, verbose=False):
        flattened_features = [feats.reshape(feats.shape[0], -1) for feats in features]
        flattened_features = np.vstack(flattened_features)

        flattened_labels = np.tile(labels, len(features))

        if verbose:
            self.console.print(f"Features: {flattened_features.shape}")
            self.console.print(f"Labels: {flattened_labels.shape}")

        return flattened_features, flattened_labels
