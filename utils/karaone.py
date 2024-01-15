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
from rich.console import Console

from utils.config import line_separator


subjects = [
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

    def __init__(self, data_folder, console=None):
        self.data_folder = data_folder
        self.subjects = subjects
        self.console = console if console else Console()

    def load_data(self, subject=subjects[0], pick_channels=[-1]):
        self.subject = subject
        self.console.print(f"Subject: [purple]{subject}[/]")
        line_separator()

        eeglab_raw_filename = glob.glob(
            os.path.join(self.data_folder, subject, "*.set")
        )[0]
        eeglab_raw_file = os.path.join(self.data_folder, subject, eeglab_raw_filename)
        self.raw = mne.io.read_raw_eeglab(eeglab_raw_file, montage_units="mm")

        if pick_channels == [-1]:
            self.channels = self.raw.ch_names  # All channels
        else:
            if not all(channel in self.raw.ch_names for channel in pick_channels):
                raise ValueError("Invalid channel(s) specified.")
            self.channels = pick_channels

        self.console.print("All channels: \n[", ", ".join(self.raw.ch_names), "]")
        line_separator()
        self.console.print("Chosen channels: [", ", ".join(self.channels), "]")
        line_separator()

        self.raw.pick_channels(self.channels)
        self.console.print(f"Raw data shape: [black]{self.raw.get_data().shape}[/]")
        line_separator()

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
        kinect_folder = os.path.join(self.data_folder, self.subject, "kinect_data")
        wav_fns = list(filter(lambda x: ".wav" in x, os.listdir(kinect_folder)))

        assert len(wav_fns) == len(spk_mats)

        speaking_mats = []
        num_files = len(wav_fns)

        for index in range(num_files):
            mat = spk_mats[index]
            wav_file = os.path.join(
                self.data_folder, self.subject, "kinect_data", str(index) + ".wav"
            )
            sample_rate, data = scipy.io.wavfile.read(wav_file)
            num_samples = math.floor((len(data) / sample_rate) * self.sampling_freq)
            e_ind = min(num_samples, mat.shape[1])
            mat = mat[:, 0:e_ind]
            speaking_mats.append(mat)

        return speaking_mats

    def get_labels(self):
        labels_file = os.path.join(
            self.data_folder, self.subject, "kinect_data", "labels.txt"
        )
        with open(labels_file, encoding="utf-8") as F:
            prompts = F.read().splitlines()

        self.prompts = prompts
        return self.prompts

    def get_events(self):
        # event_id = {}
        # for index, prompt in enumerate(sorted(set(self.prompts), key=len)):
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
        for epoch, prompt in enumerate(self.prompts):
            inds = self.epoch_inds["thinking_inds"][0][epoch][0]
            event = (inds[0] - 1, 0, event_id[prompt])
            events.append(event)
        events = np.array(events)

        self.event_id = event_id
        self.events = events
        return events, event_id

    def make_epochs(self, sampling_freq=1000):
        epoch_inds_file = os.path.join(self.data_folder, self.subject, "epoch_inds.mat")
        self.epoch_inds = scipy.io.loadmat(epoch_inds_file)

        data = self.raw.get_data()
        data = data * 10**6  # Use microVolts instead of Volts

        clearing_mats = self.split_data(self.epoch_inds["clearing_inds"], data)
        speaking_mats = self.split_data(self.epoch_inds["speaking_inds"], data)
        thinking_mats = self.split_data(self.epoch_inds["thinking_inds"], data)

        num_epochs = len(clearing_mats)

        stimuli_mats = speaking_mats[0::2]
        spk_mats = speaking_mats[1::2]

        self.sampling_freq = (
            sampling_freq  # The sampling rate for EEG data, which is 1 kHz, by default.
        )
        speaking_mats = self.trim_speaking_mats(spk_mats)
        self.get_labels()
        self.get_events()

        # # Usage
        # epoch = 10
        # channel = 5
        # ep_data = thinking_mats[epoch][channel]

        epochs = mne.EpochsArray(
            thinking_mats, self.raw.info, events=self.events, event_id=self.event_id
        )
        line_separator()
        self.console.print(f"Epochs shape: [black]{epochs.get_data().shape}[/]")
        line_separator()

        # epochs_psd = epochs.compute_psd()
        # epochs_psd.shape
        # epochs_psd.get_data().shape
        # epochs.plot()

        self.epochs = epochs
        return epochs

    def apply_baseline_correction(self, baseline=(0, 0)):
        """
        baseline = (0, 0)  # Baseline from the beginning of the epoch to t=0 seconds
        """
        wbc_epochs = self.epochs.copy()  # Without baseline correction epochs

        self.epochs.apply_baseline(baseline)  # Baseline corrected epochs

        # Verify baseline correction & centering signal around zero
        baseline_period = (0, 0)

        # Extract the data within the baseline period
        baseline_data = self.epochs.copy().crop(*baseline_period).get_data()

        # Calculate the mean within the baseline period
        baseline_mean = np.mean(baseline_data)

        # Check if the mean is close to zero
        if np.isclose(baseline_mean, 0, atol=1e-10):
            self.console.print("The signal is centered after baseline correction.")
        else:
            self.console.print("The signal may not be centered around zero.")
        line_separator()

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
