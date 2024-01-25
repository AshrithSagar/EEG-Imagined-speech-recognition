"""
ifs.py
Information Set Theory Utility scripts
"""
import os
import numpy as np
from rich.console import Console


from utils.config import line_separator


class InformationSet:
    def __init__(self, console=None, verbose=False):
        self.verbose = verbose
        self.console = console if console else Console()
