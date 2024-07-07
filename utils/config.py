"""
config.py
Configuration utils
"""

import argparse
import os

import toml
import yaml
from rich.console import Console
from rich.traceback import install

install()


def line_separator(line="normal", color="", width="full", console=None):
    """
    Print a horizontal line to distinguish between blocks
    """
    console = console if console else Console()

    line_characters = {"normal": "\u2500", "thick": "\u2501", "double": "\u2550"}
    selected_line = line_characters.get(line, "\u2500")

    widths = {"full": 1.0, "half": 0.5, "quarter": 0.25}
    effective_width = int(console.width * widths.get(width, 1.0))

    if color:
        separator = f"[{color}]{selected_line * effective_width}[/]"
    else:
        separator = f"{selected_line * effective_width}"
    console.print(separator)


class Config:
    """Class to manage configuration settings"""

    def __init__(self, file="config.yaml"):
        self.file = file
        self.config = self.load(file)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __delitem__(self, key):
        del self.config[key]

    def __contains__(self, key):
        return key in self.config

    def __str__(self):
        return f"{self.__class__.__name__}({self.file})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file})"

    def get(self, key, default=None):
        return self.config.get(key, default)

    def load(self, file=None):
        """Load configuration settings from a YAML or TOML file"""

        config_file = file if file else self.file
        with open(config_file, "r") as file:
            if config_file.endswith(".toml"):
                config = toml.load(file)
            elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                config = yaml.safe_load(file)
            else:
                raise ValueError(
                    "Unsupported file format. Only .toml and .yaml/.yml are supported."
                )
        return config


def fetch_select(key, choice):
    from utils.classifier import (
        ClassifierGridSearch,
        EvaluateClassifier,
        RegularClassifier,
    )
    from utils.feis import FEISDataLoader
    from utils.karaone import KaraOneDataLoader

    options = {
        "dataset": {
            "FEIS": FEISDataLoader,
            "KaraOne": KaraOneDataLoader,
        },
        "classifier": {
            "RegularClassifier": RegularClassifier,
            "EvaluateClassifier": EvaluateClassifier,
            "ClassifierGridSearch": ClassifierGridSearch,
        },
    }

    if key not in options:
        raise ValueError(f"Invalid key: {key}")

    if choice not in options[key]:
        raise ValueError(f"Invalid {key} name: {choice}")

    return options[key][choice]


def save_console(console, file, mode="w"):
    """Save the rich Console output to a file.
    Args:
    - console: The rich Console object to be saved.
    - file: The path to the file where the console output will be saved.
    - mode: The mode in which the file will be opened. Defaults to "w" (write mode).
    """
    with open(file, mode) as file_handle:
        file_handle.write(console.export_text())


def setup_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        metavar="config_file",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the configuration file [.toml or .yaml/.yml] (default: config.yaml)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    return args
