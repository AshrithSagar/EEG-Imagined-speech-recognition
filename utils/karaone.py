"""
karaone_utils.py
KaraOne Utility scripts
"""

import glob
import math
import os

import antropy
import mne
import numpy as np
import scipy.io
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from scipy import integrate, stats
from scipy.spatial.distance import cdist

from utils.config import line_separator


all_subjects = [
    "MM05",
    "MM08",
    "MM09",
    "MM10",
    "MM11",
    "MM12",
    "MM14",
    "MM15",
    "MM16",
    "MM18",
    "MM19",
    "MM20",
    "MM21",
    "P02",
]


class KaraOneDataLoader:
    """Load data from KaraOne folder"""

    def __init__(
        self,
        raw_data_dir,
        subjects="all",
        sampling_freq=1000,
        num_milliseconds_per_trial=4900,
        verbose=False,
        console=None,
    ):
        """Parameters:
        - raw_data_dir (str): Path to the raw data folder.
        - subjects (str or list): List of subjects to load. Use "all" for all subjects (default),
            or a list of subject indices, or a list of subject names.
        - sampling_freq (int): Sampling frequency for EEG data (default: 1000 Hz).
        - num_milliseconds_per_trial (int): Number of milliseconds per trial (default: 4900 ms).
        """
        self.raw_data_dir = raw_data_dir
        self.subjects = self.get_subjects(subjects)
        self.sampling_freq = sampling_freq
        self.num_milliseconds_per_trial = num_milliseconds_per_trial
        self.epoch_type = None
        self.verbose = verbose
        mne.set_log_level(verbose=verbose)
        self.console = console if console else Console()
        self.progress = None
        if verbose:
            self.console.rule(title="[bold blue3][KaraOne Dataset][/]", style="blue3")
            self.subjects_info()

    def get_subjects(self, subjects):
        """Retrieve a list of subjects based on input criteria.

        'subjects' parameter can be one of the following:
            - "all": Retrieve all available subjects.
            - A list of integers: Retrieve subjects by their indices.
            - A list of subject names: Retrieve subjects by their names.
        """

        if subjects == "all":
            return all_subjects
        elif isinstance(subjects, list):
            if all(isinstance(subject, int) for subject in subjects):
                return [all_subjects[index] for index in subjects]
            elif all(subject in all_subjects for subject in subjects):
                return subjects

        raise ValueError(
            """Invalid value for 'subjects'.
            Should be 'all', a list of subject indices, or a list of subject names."""
        )

    def load_raw_data(self, subject, verbose=None):
        """Load data from KaraOne folder"""
        self.subject = subject
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            self.console.print(f"Subject: [purple]{subject}[/]")
            line_separator(self.console)

        data_dir = self.raw_data_dir
        eeglab_raw_filename = glob.glob(os.path.join(data_dir, subject, "*.set"))
        eeglab_raw_file = os.path.join(data_dir, subject, eeglab_raw_filename[0])
        self.raw = mne.io.read_raw_eeglab(
            eeglab_raw_file, montage_units="mm", verbose=verbose or "critical"
        )

        return self.raw

    def load_data(self, data_dir, subject, verbose=None):
        """Load data from a .fif file"""
        self.subject = subject
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            self.console.print(f"Subject: [purple]{subject}[/]")
            line_separator(self.console)

        raw_filename = os.path.join(data_dir, subject, "raw.fif")
        self.raw = mne.io.read_raw_fif(
            raw_filename, preload=True, verbose=verbose or "critical"
        )

        return self.raw

    def pick_channels(self, pick_channels=[-1], verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if pick_channels == [-1]:
            self.channels = self.raw.ch_names  # All channels
        else:
            if not all(channel in self.raw.ch_names for channel in pick_channels):
                raise ValueError("Invalid channel(s) specified.")
            self.channels = pick_channels

        if verbose:
            self.console.print("All channels: \n[", ", ".join(self.raw.ch_names), "]")
            line_separator(self.console)
            self.console.print("Chosen channels: [", ", ".join(self.channels), "]")
            line_separator(self.console)

        self.raw.pick_channels(self.channels, verbose=verbose)
        if verbose:
            self.console.print(
                f"Raw data shape: [black]{self.raw.get_data(copy=True).shape}[/]"
            )
            line_separator(self.console)

        return self.raw

    def split_data(self, inds, data):
        n_epochs = inds.shape[1]
        n_channels = data.shape[0]
        n_times = self.num_milliseconds_per_trial  # Making fixed dimension n_times

        # epoched_data = []
        epoched_data = np.zeros([n_epochs, n_channels, n_times])

        for i, ind in enumerate(inds[0]):
            ind_start = ind[0][0] - 1

            # ind_end = ind[0][1]
            ind_end = ind_start + n_times

            # epoched_data.append(data[:, ind_start:ind_end])
            epoched_data[i] = data[:, ind_start:ind_end]

        return epoched_data

    def trim_speaking_mats(self, spk_mats):
        # Matrices trimmed to contain only the "speaking" segments of the EEG data.
        kinect_folder = os.path.join(self.raw_data_dir, self.subject, "kinect_data")
        wav_fns = [
            file
            for file in os.listdir(kinect_folder)
            if not file.startswith(".") and file.endswith(".wav")
        ]

        assert len(wav_fns) == len(spk_mats)

        speaking_mats = []
        num_files = len(wav_fns)

        for index in range(num_files):
            mat = spk_mats[index]
            wav_file = os.path.join(
                self.raw_data_dir, self.subject, "kinect_data", str(index) + ".wav"
            )
            sample_rate, data = scipy.io.wavfile.read(wav_file)
            num_samples = math.floor((len(data) / sample_rate) * self.sampling_freq)
            e_ind = min(num_samples, mat.shape[1])
            mat = mat[:, 0:e_ind]
            speaking_mats.append(mat)

        return speaking_mats

    def get_epoch_labels(self, subject=None):
        subject = subject or self.subject

        labels_file = os.path.join(
            self.raw_data_dir, subject, "kinect_data", "labels.txt"
        )
        with open(labels_file, encoding="utf-8") as F:
            prompts = F.read().splitlines()

        self.epoch_labels = np.asarray(prompts)
        return self.epoch_labels

    def get_all_epoch_labels(self):
        self.all_epoch_labels = [
            self.get_epoch_labels(subject) for subject in self.subjects
        ]
        return self.all_epoch_labels

    def get_events(self, epoch_type: str = None):
        epoch_type = epoch_type or self.epoch_type

        # event_id = {}
        # for index, prompt in enumerate(sorted(set(self.epoch_labels), key=len)):
        #     event_id.update({prompt: index + 1})
        event_id = {
            "/n/": 1,
            "/m/": 2,
            "/uw/": 3,
            "/iy/": 4,
            "/diy/": 5,
            "/tiy/": 6,
            "/piy/": 7,
            "pat": 8,
            "pot": 9,
            "gnaw": 10,
            "knew": 11,
        }

        events = []
        for epoch, prompt in enumerate(self.epoch_labels):
            inds = self.epoch_inds[f"{epoch_type}_inds"][0][epoch][0]
            event = (inds[0] - 1, 0, event_id[prompt])
            events.append(event)

        self.epoch_type = epoch_type
        self.event_id = event_id
        self.events = np.array(events)
        return events, event_id

    def apply_bandpass_filter(self, l_freq=0.5, h_freq=50.0, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        raw_data = mne.filter.filter_data(
            self.raw.get_data(copy=True),
            sfreq=self.raw.info["sfreq"],
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=verbose,
        )
        self.raw = mne.io.RawArray(raw_data, self.raw.info, verbose=verbose)
        return self.raw

    def assemble_epochs(self, verbose=None):
        """Assembles and organizes different types of epochs from raw EEG data."""
        verbose = verbose if verbose is not None else self.verbose
        epoch_inds_file = os.path.join(
            self.raw_data_dir, self.subject, "epoch_inds.mat"
        )
        self.epoch_inds = scipy.io.loadmat(epoch_inds_file)

        data = self.raw.get_data(copy=True)
        data = data * 10**6  # Use microVolts instead of Volts

        self.all_mats = {
            "clearing": self.split_data(self.epoch_inds["clearing_inds"], data),
            "thinking": self.split_data(self.epoch_inds["thinking_inds"], data),
            "stimuli": self.split_data(self.epoch_inds["speaking_inds"], data)[0::2],
            "speaking": self.trim_speaking_mats(
                self.split_data(self.epoch_inds["speaking_inds"], data)[1::2]
            ),
        }
        return self.all_mats

    def make_epochs(self, epoch_type: str = None, verbose=None):
        epoch_type = epoch_type or self.epoch_type
        verbose = verbose if verbose is not None else self.verbose

        self.epochs = mne.EpochsArray(
            self.all_mats[epoch_type],
            self.raw.info,
            events=self.events,
            event_id=self.event_id,
            verbose=verbose,
        )

        if verbose:
            line_separator(self.console)
            self.console.print(
                f"Epochs shape: [black]{self.epochs.get_data(copy=True).shape}[/]"
            )
            line_separator(self.console)

        return self.epochs

    def epochs_apply_baseline_correction(self, baseline=(0, 0), verbose=None):
        """
        baseline = (0, 0) ==> Baseline from the beginning of the epoch to t=0 seconds
        """
        verbose = verbose if verbose is not None else self.verbose
        wbc_epochs = self.epochs.copy()  # Without baseline correction epochs
        self.epochs.apply_baseline(
            baseline, verbose=verbose
        )  # Baseline corrected epochs

        # Verify baseline correction & centering signal around zero
        baseline_period = (0, 0)

        # Extract the data within the baseline period
        baseline_data = self.epochs.copy().crop(*baseline_period, verbose=verbose)
        baseline_data = baseline_data.get_data(copy=True)

        # Calculate the mean within the baseline period
        baseline_mean = np.mean(baseline_data)

        if verbose:
            # Check if the mean is close to zero
            if np.isclose(baseline_mean, 0, atol=1e-10):
                self.console.print("The signal is centered after baseline correction.")
            else:
                self.console.print("The signal may not be centered around zero.")
            line_separator(self.console)

        return self.epochs

    def epochs_info(self, verbose=None):
        """Display the shape of the epochs"""
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            table = Table(title="[bold underline]Epochs Info[/]")
            table.add_column("Subject", justify="right", style="magenta", no_wrap=True)
            table.add_column("Epochs", justify="center", style="cyan", no_wrap=True)
            table.add_column("Labels", style="cyan", no_wrap=True)

            for epochs, subject, labels in zip(
                self.all_epochs, self.subjects, self.all_epoch_labels
            ):
                table.add_row(
                    str(subject),
                    str(epochs.get_data(copy=True).shape),
                    str(labels.shape),
                )
            self.console.print(table)

    def create_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        )

    def process_raw_data(
        self,
        save_dir=None,
        pick_channels=[-1],
        l_freq=0.5,
        h_freq=50.0,
        num_neighbors=4,
        overwrite=False,
        verbose=False,
    ):
        self.verbose = verbose
        self.data_dir = save_dir

        with self.create_progress_bar() as self.progress:
            task_subjects = self.progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )
            task_filter = self.progress.add_task("Applying Laplacian filter ...")

            for subject in self.subjects:
                if not overwrite and os.path.exists(
                    os.path.join(self.data_dir, subject, "raw.fif")
                ):
                    self.progress.update(task_subjects, advance=1)
                    continue

                self.load_raw_data(subject)
                self.pick_channels(pick_channels)
                self.apply_bandpass_filter(l_freq=l_freq, h_freq=h_freq)
                # self.apply_laplacian_filter(num_neighbors, task=task_filter)
                self.save_raw(self.data_dir, overwrite=overwrite)
                self.progress.update(task_subjects, advance=1)

    def process_epochs(
        self,
        epoch_type: str,
        data_dir=None,
        pick_channels=[-1],
        verbose=False,
    ):
        self.verbose = verbose
        data_dir = data_dir or self.data_dir
        self.epoch_type = epoch_type
        self.all_epochs = []
        self.all_epoch_labels = []

        with self.create_progress_bar() as self.progress:
            task_subjects = self.progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )

            for subject in self.subjects:
                self.load_data(data_dir, subject)
                self.pick_channels(pick_channels)

                self.assemble_epochs()
                epoch_labels = self.get_epoch_labels()
                self.get_events()
                self.make_epochs()
                subject_epochs = self.epochs_apply_baseline_correction(baseline=(0, 0))

                self.all_epochs.append(subject_epochs)
                self.all_epoch_labels.append(epoch_labels)
                self.progress.update(task_subjects, advance=1)

        if verbose:
            self.epochs_info()

        return self.all_epochs, self.all_epoch_labels

    def save_raw(self, save_dir, overwrite=False, verbose=None):
        """Save the raw data to disk"""

        subject_dir = os.path.join(save_dir, self.subject)
        os.makedirs(subject_dir, exist_ok=True)

        raw_filename = os.path.join(subject_dir, "raw.fif")
        if overwrite or not os.path.exists(raw_filename):
            self.raw.save(raw_filename, overwrite=overwrite, verbose=verbose)

    def extract_features(
        self,
        save_dir,
        epoch_type=None,
        length_factor=0.1,
        overlap=0.5,
        skip_if_exists=True,
    ):
        with self.create_progress_bar() as self.progress:
            task_subjects = self.progress.add_task(
                "Subjects ...",
                total=len(self.subjects),
                completed=1,
            )
            task_features = self.progress.add_task("Computing features ...")

            epoch_type = epoch_type or self.epoch_type
            self.features_dir = save_dir
            self.get_features_functions()

            for index, subject in enumerate(self.subjects):
                if skip_if_exists:
                    if os.path.exists(
                        os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
                    ):
                        self.progress.update(task_subjects, advance=1)
                        continue

                subject_epochs = self.all_epochs[index].get_data(copy=True)
                self.progress.update(task_features, total=len(subject_epochs))
                features = self.compute_features(
                    subject_epochs, length_factor, overlap, task=task_features
                )
                self.progress.reset(task_features)
                self.save_features(subject, features)
                self.progress.update(task_subjects, advance=1)

    def compute_features(self, epochs, length_factor=0.1, overlap=0.5, task=None):
        features = []
        for epoch in epochs:
            windowed_epoch = self.window_data(epoch, length_factor, overlap)
            feats = self.make_simple_feats(windowed_epoch, flatten=False)
            all_feats = self.add_deltas(feats)
            features.append(all_feats)
            if task:
                self.progress.update(task, advance=1)

        return np.asarray(features, dtype=np.float32)

    def window_data(
        self,
        data: np.ndarray,
        length: int = None,
        length_factor: float = 0.1,
        overlap: float = 0.5,
    ):
        """Windows the data
        Parameters:
        - data (np.ndarray): EEG data of shape (n_channels, n_samples).
        - length (int): Length of the window.
        - length_factor (float): Factor to calculate the window length.
        - overlap (float): Overlap factor between consecutive windows.
        """
        if length:
            w_len = length
        elif length_factor:
            w_len = int(data.shape[1] * length_factor)
        else:
            raise ValueError("Invalid window length")

        stride = int(w_len * overlap)
        split = (data.shape[1] - w_len) // stride + 1

        no_offset_windows = np.split(data, split, axis=1)
        offset_windows = np.split(data[:, stride:-stride], split - 1, axis=1)

        windows = [0] * (2 * split - 1)
        windows[::2] = no_offset_windows
        windows[1::2] = offset_windows

        return np.array(windows, dtype=np.float32)

    def make_simple_feats(self, windowed_data: np.ndarray, flatten: bool = True):
        feats = [self.features_per_window(window, flatten) for window in windowed_data]
        return np.asarray(feats, dtype=np.float32)

    def features_per_window(self, window: np.ndarray, flatten: bool = True):
        """
        Takes a single window, returns an array of features of shape
        (n.features, electrodes), and then flattens it into a vector
        """
        outvec = np.zeros((window.shape[0], len(self.feature_functions)))
        for i in range(window.shape[0]):
            for j in range(len(self.feature_functions)):
                outvec[i, j] = self.feature_functions[j](window[i, :])

        if flatten:
            outvec = outvec.transpose().reshape(-1)

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
        """Calculates the first-order delta and second-order delta (double delta) features
        and concatenate them horizontally to the input feature array.

        The shape of the returned array is (3 * n_windows - 2, n_features);
        """

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

    def load_features(self, features_dir=None, epoch_type: str = None, verbose=None):
        """Parameters:
        - features_dir (str): Path to the features directory.
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").

        Returns:
        - features (np.ndarray): Features of shape (n.subjects, n.epochs, n.windows, n.features_per_window).
        """
        self.features = []
        features_dir = features_dir or self.features_dir
        epoch_type = epoch_type or self.epoch_type
        verbose = verbose if verbose is not None else self.verbose

        for subject in self.subjects:
            filename = os.path.join(features_dir, subject, f"{epoch_type}.npy")
            if os.path.exists(filename):
                subject_features = np.load(filename)
            else:
                raise FileNotFoundError(f"File not found: {filename}")
            self.features.append(subject_features)

        if verbose:
            labels = (
                self.all_epoch_labels
                if "all_epoch_labels" in self.__dict__
                else self.get_all_epoch_labels()
            )
            self.features_info(self.features, labels, verbose=verbose)

        return self.features

    def flatten(self, features=None, labels=None, reshape=False, verbose=None):
        """
        Flatten the features and concatenate labels.
        """

        verbose = verbose if verbose is not None else self.verbose
        features = features if features is not None else self.features
        labels = labels if labels is not None else self.get_all_epoch_labels()

        flattened_features = (
            [feats.reshape(feats.shape[0], -1) for feats in features]
            if reshape
            else features
        )
        flattened_features = np.vstack(flattened_features)

        flattened_labels = np.concatenate(labels)

        if verbose:
            self.dataset_info(flattened_features, flattened_labels, verbose=verbose)

        return flattened_features, flattened_labels

    def dataset_info(self, features=None, labels=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        features = features if features is not None else self.features
        labels = labels if labels is not None else self.get_all_epoch_labels()

        if verbose:
            table = Table(title="[bold underline]Dataset Info[/]")
            table.add_column("Data", justify="right", no_wrap=True)
            table.add_column("Shape", style="cyan", no_wrap=True)
            table.add_row("Features", str(features.shape))
            table.add_row("Labels", str(labels.shape))

            self.console.print(table)

    def subjects_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            message = f"[bold underline]Subjects:[/]\n"
            message += ", ".join(self.subjects)
            self.console.print(message)

    def features_info(self, features, labels, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            table = Table(title="[bold underline]Features Info[/]")
            table.add_column("Subject", justify="right", style="magenta", no_wrap=True)
            table.add_column("Features", justify="center", style="cyan", no_wrap=True)
            table.add_column("Labels", style="cyan", no_wrap=True)

            for feats, subject, label in zip(features, self.subjects, labels):
                table.add_row(str(subject), str(feats.shape), str(label.shape))

            self.console.print(table)

    def apply_laplacian_filter(self, num_neighbors=4, task=None, verbose=None):
        """Apply a spatial neighbourhood Laplacian filter to EEG data.

        Parameters:
        - eeg_data: ndarray, shape (num_channels, num_samples)
            EEG data matrix.
        - num_neighbors: int
            Number of nearest neighbors to consider.

        Returns:
        - filtered_data: ndarray, shape (num_channels, num_samples)
            Filtered EEG data matrix.
        """

        filtered_data = self.raw.get_data(copy=True)
        num_channels, num_samples = filtered_data.shape
        if task:
            self.progress.reset(task)
            self.progress.update(task, total=num_samples)

        # locations = np.asarray([self.raw.info["chs"][idx]["loc"][:3] for idx in picks])
        channel_locations = mne.channels.find_layout(self.raw.info, ch_type="eeg")
        channel_positions = channel_locations.pos[:, :3]

        distance_matrix = cdist(channel_positions, channel_positions)
        nearest_neighbors = np.argsort(distance_matrix, axis=1)
        nearest_neighbors = nearest_neighbors[:, 1 : (num_neighbors + 1)]

        # Normalized nearest neighbor inverse distance matrix
        inverse_distances = np.zeros((num_channels, num_neighbors))
        for i in range(num_channels):
            inverse_distances[i, :] = 1.0 / distance_matrix[i, nearest_neighbors[i, :]]
            inverse_distances[i, :] /= np.sum(inverse_distances[i, :])

        for sample in range(num_samples):
            # Sum of n-closest channel values weighted by normalized inverse distance
            sum_values = np.zeros(num_channels)
            for channel in range(num_channels):
                sum_values[channel] = np.dot(
                    inverse_distances[channel, :],
                    filtered_data[nearest_neighbors[channel, :], sample],
                )
            filtered_data[:, sample] -= sum_values
            if task:
                self.progress.update(task, advance=1)

        self.raw = mne.io.RawArray(filtered_data, self.raw.info, verbose=verbose)
        return self.raw

    def apply_laplacian_filter_csd(self, num_neighbors=4, task=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        self.raw = mne.preprocessing.compute_current_source_density(
            self.raw, n_legendre_terms=num_neighbors, verbose=verbose
        )
        return self.raw

    def get_task(self, labels=None, task=None, verbose=None):
        """Classiﬁcation of phonological categories.
        Classes:
        - 0: Vowel only (0) vs Consonant (1)
        - 1: Non-nasal (0) vs Nasal (1)
        - 2: Non-bilabial (0) vs Bilabial (1)
        - 3: Non-iy (0) vs iy (1)  [High-front vowel]
        - 4: Non-uw (0) vs uw (1)  [High-back vowel]
        """
        verbose = verbose if verbose is not None else self.verbose
        labels = labels if labels is not None else self.all_epoch_labels

        class_dict = {
            "/diy/": [1, 0, 0, 1, 0],
            "/iy/": [0, 0, 0, 1, 0],
            "/m/": [1, 1, 1, 0, 0],
            "/n/": [1, 1, 0, 0, 0],
            "/piy/": [1, 0, 1, 1, 0],
            "/tiy/": [1, 0, 0, 1, 0],
            "/uw/": [0, 0, 0, 0, 1],
            "gnaw": [1, 1, 0, 0, 0],
            "knew": [1, 1, 0, 0, 0],
            "pat": [1, 0, 1, 0, 0],
            "pot": [1, 0, 1, 0, 0],
        }

        task_labels = [class_dict[label][task] for label in labels]
        return np.asarray(task_labels)
