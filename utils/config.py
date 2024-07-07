"""
config.py
Configuration utils
"""

import argparse
import os
from typing import Any, Optional, Type, Union

import toml
import yaml
from rich.console import Console
from rich.traceback import install

from utils.classifier import ClassifierGridSearch, EvaluateClassifier, RegularClassifier
from utils.feis import FEISDataLoader
from utils.karaone import KaraOneDataLoader

install()
ConfigValue = Optional[Union[dict, list, str, int, float, Any]]


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

    def __delitem__(self, key: str):
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
        config_file: str = getattr(args, "config", "config.yaml")
        return cls(config_file)


def fetch_select(key: str, choice: str) -> Union[
    Type[FEISDataLoader],
    Type[KaraOneDataLoader],
    Type[RegularClassifier],
    Type[EvaluateClassifier],
    Type[ClassifierGridSearch],
]:
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
