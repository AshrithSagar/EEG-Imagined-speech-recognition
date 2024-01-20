"""
feis.py
FEIS Dataset Utility scripts
"""
import os
import math
import glob
import mne
import subprocess
import numpy as np
import pandas as pd
import scipy.io
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


class FEISDataLoader:
    """
    Load data from FEIS folder
    """

    def __init__(self, data_dir, subjects="all", console=None):
        """Parameters:
        - data_dir (str): Path to the data folder.
        - subjects (list): List of subjects to load. Use "all" for all subjects (default)
        """
        self.data_dir = data_dir
        self.subjects = all_subjects if subjects == "all" else subjects
        self.console = console if console else Console()
        self.data = {}

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

    def load_data_eeg(self, epoch_type, verbose=False):
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

            data = []
            for subject in self.subjects:
                file = os.path.join(self.data_dir, subject, f"{epoch_type}.csv")
                if verbose:
                    self.console.print(f"Loading {subject}/{file} ...")
                subject_data = pd.read_csv(file)
                data.append(subject_data)
                progress.update(task_subjects, advance=1)
            self.data.update({epoch_type: data})
        return data

    def extract_labels(self, epoch_type="speaking"):
        subject = self.subjects[0]
        file = os.path.join(self.data_dir, subject, f"{epoch_type}.csv")
        df = pd.read_csv(file, header=None, skiprows=range(1, 1280), usecols=[16])
        eeg_labels = df.values.flatten()
        self.labels = eeg_labels[1::1280]
        return self.labels
