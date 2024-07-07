"""
config.py
Configuration utils
"""

import argparse
import os
from typing import Any, Optional, Union

import toml
import yaml
from rich.console import Console
from rich.traceback import install

install()
ConfigValue = Optional[Union[dict, list, str, int, float, Any]]


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

    def __init__(self, file: str = "config.yaml", verbose: bool = True):
        self.file: str = file
        if not os.path.exists(self.file):
            raise FileNotFoundError(f"Configuration file not found: {self.file}")
        self.config: dict = self.load()
        if verbose:
            console = Console()
            console.print(f"Configuration loaded from [bold]{self.file}[/]")

    def __getitem__(self, key: str) -> ConfigValue:
        return self.config[key]

    def __setitem__(self, key: str, value: ConfigValue):
        self.config[key] = value

    def __delitem__(self, key: str) -> None:
        del self.config[key]

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.file})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.file})"

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        return self.config.get(key, default)

    def load(self) -> ConfigValue:
        """Load configuration settings from a YAML or TOML file"""
        with open(self.file, "r") as f:
            if self.file.endswith(".toml"):
                config = toml.load(f)
            elif self.file.endswith(".yaml") or self.file.endswith(".yml"):
                config = yaml.safe_load(f)
            else:
                raise ValueError("Invalid configuration file format")

        return config

    @classmethod
    def from_args(cls, description: Optional[str] = None):
        """Create a Config instance from command-line arguments"""
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
        return cls(args.config)


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
