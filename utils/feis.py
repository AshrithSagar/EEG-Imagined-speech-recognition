"""
feis.py
FEIS Dataset Utility scripts
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
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "chinese-1",
    "chinese-2",
]


class FEISDataLoader:
    """
    Load data from FEIS folder
    """

    def __init__(self, data_folder, console=None):
        self.data_folder = data_folder
        self.subjects = subjects
        self.console = console if console else Console()
