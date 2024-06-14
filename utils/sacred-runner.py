"""
sacred-runner.py
Run workflows with sacred
"""

import argparse
import importlib.util
import os
import sys

sys.path.append(os.getcwd())
from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils.config import load_config


class SacredRunner:
    def __init__(self, name, save_dir, file, config):
        self.name = name
        self.save_dir = save_dir
        self.file = file
        self.config = config

        self.experiment = Experiment(name=self.name)
        self.experiment.observers.append(FileStorageObserver(self.save_dir))
        self.experiment.add_config(self.config)

    def load(self, file=None):
        if file is None:
            file = self.file
        self.file = file
        spec = importlib.util.spec_from_file_location("module", self.file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run(self):
        module = self.load(self.file)
        module.experiment = self.experiment

        @self.experiment.main
        def main(_run, _config):
            if hasattr(module, "main") and callable(getattr(module, "main")):
                module.main()
            else:
                raise AttributeError("No 'main' function found in the specified file.")

        self.experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run workflows with Sacred")
    parser.add_argument("file", type=str, help="Path to the workflow file")
    parser.add_argument(
        "--config",
        metavar="config_file",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )

    args = parser.parse_args()
    s_args = load_config(args.config, key="sacred")

    runner = SacredRunner(
        name=s_args["name"],
        save_dir=s_args["base_dir"],
        file=args.file,
        config=args.config,
    )
    runner.run()
