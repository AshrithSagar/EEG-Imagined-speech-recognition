"""
config.py
Configuration utils
"""

import argparse
import os
from typing import Any, Literal, Self

import toml
import yaml
from rich.console import Console
from rich.rule import Rule

from eeg_isr.classifier import (
    ClassifierGridSearch,
    ClassifierMixin,
    EvaluateClassifier,
    RegularClassifier,
)
from eeg_isr.dataset import DatasetLoader
from eeg_isr.feis import FEISDataLoader
from eeg_isr.karaone import KaraOneDataLoader

type ConfigValue = dict[Any, Any] | list[Any] | str | int | float | Any | None


class Config:
    """Class to manage configuration settings"""

    def __init__(self, file: str = "config.yaml", verbose: bool = True) -> None:
        self.file: str = file
        if not os.path.exists(self.file):
            raise FileNotFoundError(f"Configuration file not found: {self.file}")
        self.config = self.load()
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

    def load(self) -> dict[str, ConfigValue]:
        """Load configuration settings from a YAML or TOML file"""
        with open(self.file, "r") as f:
            match os.path.splitext(self.file.lower())[1]:
                case ".toml":
                    config = toml.load(f)
                case ".yaml" | ".yml":
                    config = yaml.safe_load(f)
                case _:
                    raise ValueError("Invalid configuration file format")

        return config

    @classmethod
    def from_args(cls, description: str | None = None) -> Self:
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


def fetch_dataset(choice: Literal["FEIS", "KaraOne"]) -> type[DatasetLoader]:
    match choice:
        case "FEIS":
            return FEISDataLoader
        case "KaraOne":
            return KaraOneDataLoader
        case _:
            raise ValueError(f"Invalid dataset name: {choice}")


def fetch_classifier(
    choice: Literal["RegularClassifier", "EvaluateClassifier", "ClassifierGridSearch"],
) -> type[ClassifierMixin]:
    match choice:
        case "RegularClassifier":
            return RegularClassifier
        case "EvaluateClassifier":
            return EvaluateClassifier
        case "ClassifierGridSearch":
            return ClassifierGridSearch
        case _:
            raise ValueError(f"Invalid classifier name: {choice}")


class ConsoleHandler(Console):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def line(self, line: str = "normal") -> None:
        """Print a horizontal rule with a specified line style.
        Args:
        - line: The style of the line. Options are "normal", "thick", and "double".
        """
        options = {"normal": "\u2500", "thick": "\u2501", "double": "\u2550"}
        characters = options.get(line, "\u2500")
        self.print(Rule(characters=characters))

    def save(self, file: str, mode: str = "w") -> None:
        """Save the rich Console output to a file.
        Args:
        - file: The path to the file where the console output will be saved.
        - mode: The mode in which the file will be opened. Defaults to "w" (write mode).
        """
        with open(file, mode) as file_handle:
            file_handle.write(self.export_text())
