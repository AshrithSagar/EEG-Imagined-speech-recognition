"""
karaone_utils.py
KaraOne Utility scripts
"""
import os
import math
import glob
import mne
import numpy as np
import scipy.io
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
    """
    Load data from KaraOne folder
    """

    def __init__(
        self,
        data_dir,
        subjects="all",
        sampling_freq=1000,
        num_milliseconds_per_trial=4900,
        verbose=False,
        console=None,
    ):
        """Parameters:
        - data_dir (str): Path to the data folder.
        - subjects (list): List of subjects to load. Use "all" for all subjects (default)
        - sampling_freq (int): Sampling frequency for EEG data (default: 1000 Hz).
        - num_milliseconds_per_trial (int): Number of milliseconds per trial (default: 4900 ms).
        """
        self.data_dir = data_dir
        self.subjects = all_subjects if subjects == "all" else subjects
        self.sampling_freq = sampling_freq
        self.num_milliseconds_per_trial = num_milliseconds_per_trial
        self.epoch_type = None
        self.verbose = verbose
        self.console = console if console else Console()

    def load_data(self, subject, verbose=False):
        self.subject = subject
        verbose = verbose or self.verbose
        if verbose:
            self.console.print(f"Subject: [purple]{subject}[/]")
            line_separator(self.console)

        eeglab_raw_filename = glob.glob(os.path.join(self.data_dir, subject, "*.set"))
        eeglab_raw_file = os.path.join(self.data_dir, subject, eeglab_raw_filename[0])
        self.raw = mne.io.read_raw_eeglab(
            eeglab_raw_file, montage_units="mm", verbose=verbose or "critical"
        )

        return self.raw

    def pick_channels(self, pick_channels=[-1], verbose=False):
        verbose = verbose or self.verbose

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
            self.console.print(f"Raw data shape: [black]{self.raw.get_data().shape}[/]")
            line_separator(self.console)

        return self.raw

    @staticmethod
    def split_data(inds, data):
        n_epochs = inds.shape[1]
        n_channels = data.shape[0]
        n_times = 4900  # Making fixed dimension n_times

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
        kinect_folder = os.path.join(self.data_dir, self.subject, "kinect_data")
        wav_fns = list(filter(lambda x: ".wav" in x, os.listdir(kinect_folder)))

        assert len(wav_fns) == len(spk_mats)

        speaking_mats = []
        num_files = len(wav_fns)

        for index in range(num_files):
            mat = spk_mats[index]
            wav_file = os.path.join(
                self.data_dir, self.subject, "kinect_data", str(index) + ".wav"
            )
            sample_rate, data = scipy.io.wavfile.read(wav_file)
            num_samples = math.floor((len(data) / sample_rate) * self.sampling_freq)
            e_ind = min(num_samples, mat.shape[1])
            mat = mat[:, 0:e_ind]
            speaking_mats.append(mat)

        return speaking_mats

    def get_epoch_labels(self):
        labels_file = os.path.join(
            self.data_dir, self.subject, "kinect_data", "labels.txt"
        )
        with open(labels_file, encoding="utf-8") as F:
            prompts = F.read().splitlines()

        self.epoch_labels = np.asarray(prompts)
        return self.epoch_labels

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

    def apply_bandpass_filter(self, l_freq=0.5, h_freq=50.0, verbose=False):
        verbose = verbose or self.verbose
        raw_data = mne.filter.filter_data(
            self.raw.get_data(),
            sfreq=self.raw.info["sfreq"],
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=verbose,
        )
        self.raw = mne.io.RawArray(raw_data, self.raw.info, verbose=verbose)
        return self.raw

    def assemble_epochs(self, verbose=False):
        """Assembles and organizes different types of epochs from raw EEG data."""
        verbose = verbose or self.verbose
        epoch_inds_file = os.path.join(self.data_dir, self.subject, "epoch_inds.mat")
        self.epoch_inds = scipy.io.loadmat(epoch_inds_file)

        data = self.raw.get_data()
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

    def make_epochs(self, epoch_type: str = None, verbose=False):
        epoch_type = epoch_type or self.epoch_type

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
                f"Epochs shape: [black]{self.epochs.get_data().shape}[/]"
            )
            line_separator(self.console)

        # epochs_psd = self.epochs.compute_psd()
        # epochs_psd.shape
        # epochs_psd.get_data().shape
        # epochs.plot()

        return self.epochs

    def epochs_apply_baseline_correction(self, baseline=(0, 0), verbose=False):
        """
        baseline = (0, 0)  # Baseline from the beginning of the epoch to t=0 seconds
        """
        verbose = verbose or self.verbose
        wbc_epochs = self.epochs.copy()  # Without baseline correction epochs
        self.epochs.apply_baseline(
            baseline, verbose=verbose
        )  # Baseline corrected epochs

        # Verify baseline correction & centering signal around zero
        baseline_period = (0, 0)

        # Extract the data within the baseline period
        baseline_data = self.epochs.copy().crop(*baseline_period, verbose=verbose)
        baseline_data = baseline_data.get_data()

        # Calculate the mean within the baseline period
        baseline_mean = np.mean(baseline_data)

        if verbose:
            # Check if the mean is close to zero
            if np.isclose(baseline_mean, 0, atol=1e-10):
                self.console.print("The signal is centered after baseline correction.")
            else:
                self.console.print("The signal may not be centered around zero.")
            line_separator(self.console)

        # epochs.average().plot()
        # plt.show()

        # wbc_epochs.average().plot()
        # plt.show()

        # mne.viz.plot_events(epochs.events)
        # plt.show()

        # mne.viz.plot_epochs(epochs, scalings="auto")
        # plt.show()

        # epochs.save(+'.fif', verbose='error')

        return self.epochs

    def epochs_info(self, verbose=False):
        """Display the shape of the epochs"""
        verbose = verbose or self.verbose
        if verbose:
            message = f"[bold underline]Epochs:[/]\n"
            message += "\n".join(
                [
                    f"{subject}: {epoch.get_data().shape}"
                    for epoch, subject in zip(self.all_epochs, self.subjects)
                ]
            )
            self.console.print(message)

    def process_data(self, epoch_type: str, pick_channels=[-1], verbose=False):
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

            self.verbose = verbose
            self.epoch_type = epoch_type
            self.all_epochs = []
            self.all_epoch_labels = []

            for subject in self.subjects:
                self.load_data(subject)
                self.pick_channels(pick_channels)
                self.apply_bandpass_filter(l_freq=0.5, h_freq=50.0)

                self.assemble_epochs()
                epoch_labels = self.get_epoch_labels()
                self.get_events()
                self.make_epochs()
                subject_epochs = self.epochs_apply_baseline_correction(baseline=(0, 0))

                self.all_epochs.append(subject_epochs)
                self.all_epoch_labels.append(epoch_labels)
                progress.update(task_subjects, advance=1)

    def extract_features(self, features_dir, epoch_type=None, skip_if_exists=True):
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

            epoch_type = epoch_type or self.epoch_type
            self.features_dir = features_dir
            self.get_features_functions()

            for index, subject in enumerate(self.subjects):
                if skip_if_exists:
                    if os.path.exists(
                        os.path.join(self.features_dir, subject, f"{epoch_type}.npy")
                    ):
                        progress.update(task_subjects, advance=1)
                        continue

                subject_epochs = self.all_epochs[index].get_data()
                progress.update(task_features, total=len(subject_epochs))
                features = self.compute_features(
                    subject_epochs, progress=progress, task=task_features
                )
                progress.reset(task_features)
                self.save_features(subject, features)
                progress.update(task_subjects, advance=1)

    def compute_features(self, epochs, progress=None, task=None):
        features = []
        for epoch in epochs:
            epoch = self.window_data(epoch, split=10)
            feats = self.make_simple_feats(epoch)
            feats = self.add_deltas(feats)
            features.append(feats)
            if progress:
                progress.update(task, advance=1)

        return np.asarray(features, dtype=np.float32)

    def window_data(self, data: np.ndarray, split: int = 10):
        """Windows the data with a stride length of 1."""
        w_len = data.shape[1] // split
        stride = w_len // 2
        no_offset_windows = np.split(data, split, axis=1)
        offset_windows = np.split(data[:, stride:-stride], split - 1, axis=1)
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
        outvec = np.zeros((window.shape[0], len(self.feature_functions)))
        for i in range(window.shape[0]):
            for j in range(len(self.feature_functions)):
                outvec[i, j] = self.feature_functions[j](window[i, :])
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
            # sample_entropy,
            # perm_entropy,
            # svd_entropy,
            # app_entropy,
            # petrosian,
            # katz,
            # higuchi,
            # rootmeansquare,
            # dfa,
        ]
        return self.feature_functions

    def add_deltas(self, feats_array: np.ndarray):
        deltas = np.diff(feats_array, axis=0)
        double_deltas = np.diff(deltas, axis=0)
        all_feats = np.hstack((feats_array[2:], deltas[1:], double_deltas))
        return all_feats

    def save_features(self, subject: str, features: np.ndarray):
        subject_features_dir = os.path.join(self.features_dir, subject)
        os.makedirs(subject_features_dir, exist_ok=True)

        filename = os.path.join(subject_features_dir, f"{self.epoch_type}.npy")
        np.save(filename, features)

    def load_features(self, epoch_type: str = None, verbose=False):
        """Parameters:
        - epoch_type (str): Type of epoch (e.g., "stimuli", "thinking", "speaking").

        Returns:
        - features (np.ndarray): Features of shape (n.subjects, n.epochs, n.windows, n.features_per_window).
        """
        features = []
        epoch_type = epoch_type or self.epoch_type
        verbose = verbose or self.verbose

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

    def flatten(self, features, labels, verbose=False):
        flattened_features = [feats.reshape(feats.shape[0], -1) for feats in features]
        flattened_features = np.vstack(flattened_features)

        flattened_labels = np.concatenate(labels)

        if verbose:
            self.console.print(f"Features: {flattened_features.shape}")
            self.console.print(f"Labels: {flattened_labels.shape}")

        return flattened_features, flattened_labels
