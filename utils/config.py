"""
config.py
Configuration utils
"""

import argparse
import os
from typing import Any, Callable, Dict, Optional, Type, Union

import toml
import yaml
from rich.console import Console
from rich.rule import Rule
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


def fetch_dataset(choice: str) -> Type[Union[FEISDataLoader, KaraOneDataLoader]]:
    options: Dict[str, Callable[..., object]] = {
        "FEIS": FEISDataLoader,
        "KaraOne": KaraOneDataLoader,
    }

    if choice not in options:
        raise ValueError(f"Invalid dataset name: {choice}")

    return options[choice]


def fetch_classifier(
    choice: str,
) -> Type[Union[RegularClassifier, EvaluateClassifier, ClassifierGridSearch]]:
    options: Dict[str, Callable[..., object]] = {
        "RegularClassifier": RegularClassifier,
        "EvaluateClassifier": EvaluateClassifier,
        "ClassifierGridSearch": ClassifierGridSearch,
    }

    if choice not in options:
        raise ValueError(f"Invalid classifier name: {choice}")

    return options[choice]


class ConsoleHandler(Console):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def line(self, line: str = "normal"):
        """Print a horizontal rule with a specified line style.
        Args:
        - line: The style of the line. Options are "normal", "thick", and "double".
        """
        options = {"normal": "\u2500", "thick": "\u2501", "double": "\u2550"}
        characters = options.get(line, "\u2500")
        self.print(Rule(characters=characters))

    def save(self, file: str, mode: str = "w"):
        """Save the rich Console output to a file.
        Args:
        - file: The path to the file where the console output will be saved.
        - mode: The mode in which the file will be opened. Defaults to "w" (write mode).
        """
        with open(file, mode) as file_handle:
            file_handle.write(self.export_text())
