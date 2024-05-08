"""
config.py
Configuration utils
"""

import yaml
from rich.console import Console


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


def load_config(config_file="config.yaml", key=None):
    """
    Load configuration settings from a YAML file
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if key:
        return config.get(key, {})
    else:
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
