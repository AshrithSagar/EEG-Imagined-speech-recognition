"""
info.py
Info Utility scripts
"""

import os

import joblib
import matplotlib.pyplot as plt
import yaml
from rich.console import Console
from rich.table import Table


class ModelSummary:
    def __init__(self, model_base_dir, models, console=None, verbose=False):
        self.model_base_dir = model_base_dir
        self.models = models
        self.verbose = verbose
        self.console = console if console else Console()
        self.tasks = [f"task-{task}" for task in range(5)]

    def show(self, plots=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        for model in self.models:
            self.model_dir = os.path.join(self.model_base_dir, model)
            self.metrics_info(verbose)
            if plots:
                self.plot_confusion_matrix(verbose)

    def metrics_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        self.metrics = []
        for task in self.tasks:
            task_dir = os.path.join(self.model_dir, task)

            filename = os.path.join(task_dir, "metrics.yaml")
            with open(filename, "r") as file:
                metrics = yaml.safe_load(file)
            self.metrics.append(metrics)

        filename = os.path.join(task_dir, "params.yaml")
        with open(filename, "r") as file:
            self.params = yaml.safe_load(file)

        if verbose:
            self.console.rule(title="[bold blue3][Model Summary][/]", style="blue3")
            self.console.print(
                f"model_dir: [bold black]{os.path.basename(self.model_dir)}[/]"
            )
            self.console.print(f"model: [yellow]{self.params['model']}[/]")

            table = Table(title="[bold underline]Evaluation Metrics:[/]")
            table.add_column("Metric", justify="right", style="magenta", no_wrap=True)
            for task, metrics in zip(self.tasks, self.metrics):
                table.add_column(task, justify="left", style="cyan", no_wrap=True)
            for metric in self.metrics[0].keys():
                row = [metric]
                for task_metrics in self.metrics:
                    row.append(task_metrics[metric])
                table.add_row(*row)

            self.console.print(table)

    def plot_confusion_matrix(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        num_tasks = len(self.tasks)
        fig, axes = plt.subplots(num_tasks, 1, figsize=(8, 6 * num_tasks))

        for i, task in enumerate(self.tasks):
            task_dir = os.path.join(self.model_dir, task)
            filename = os.path.join(task_dir, "confusion_matrix.png")
            if os.path.exists(filename):
                cm = plt.imread(filename)
                axes[i].imshow(cm)
                axes[i].axis("off")
                axes[i].set_title(f"Task {i}")

        plt.tight_layout()
        plt.show()
