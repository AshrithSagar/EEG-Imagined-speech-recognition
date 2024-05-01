"""
info.py
Info Utility scripts
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay


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
                    row.append(str(task_metrics[metric]))
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


class KBestSummary:
    def __init__(self, task_dir, save_ext="png"):
        self.task_dir = task_dir
        self.save_ext = save_ext
        self.save_dir = os.path.join(task_dir, "Summary")
        os.makedirs(self.save_dir, exist_ok=True)
        self.results = None

    def get_results(self):
        results = {}
        for k_dir_name in os.listdir(self.task_dir):
            k_dir = os.path.join(self.task_dir, k_dir_name)
            if not os.path.isdir(k_dir) or "k_best-" not in k_dir_name:
                continue

            k = k_dir_name.replace("k_best-", "")
            cv_metrics = pd.read_csv(os.path.join(k_dir, "cv_metrics.csv"))
            cv_metrics["k"] = k
            results.update({k: cv_metrics})

        self.results = pd.concat(results, ignore_index=True)
        self.results.set_index(["k", self.results.index], inplace=True)

        index = sorted(self.results.index, key=lambda x: int(x[0]))
        index = [x[0] for x in index]
        self.results.index = index
        return self.results

    def get_metrics(self, metric=None):
        df = self.get_results()
        metric_df = df[df["Metric"] == metric].drop(columns=["Metric"])
        return metric_df

    def plot(self, metrics=None):
        if metrics == "all":
            metrics = self.results["Metric"].unique()
        elif not metrics:
            metrics = ["accuracy"]  # Default

        for metric in metrics:
            df = self.get_metrics(metric=metric)

            plt.clf()
            plt.plot(df.index, df["Mean"], label=metric)
            plt.fill_between(
                df.index, df["Mean"] - df["Std"], df["Mean"] + df["Std"], alpha=0.3
            )
            plt.xlabel("Number of features")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.tight_layout()

            filename = os.path.join(self.save_dir, f"{metric}.{self.save_ext}")
            plt.savefig(filename)
            plt.close()


class CurvePlotter:
    def __init__(self, estimator, X, y, cv=5):
        self.estimator = estimator
        self.X, self.y = X, y
        self.cv = cv

    def plot_learning_curve(self, **kwargs):
        params = {
            "estimator": self.estimator,
            "X": self.X,
            "y": self.y,
            "cv": self.cv,
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "scoring": "accuracy",
            "random_state": 42,
            "n_jobs": -1,
            **kwargs,
        }
        LearningCurveDisplay.from_estimator(**params)
        plt.title(f"Learning Curve for {self.estimator.__class__.__name__}")
        plt.show()

    def plot_validation_curve(self, param_name, param_range, **kwargs):
        params = {
            "estimator": self.estimator,
            "X": self.X,
            "y": self.y,
            "cv": self.cv,
            "param_name": param_name,
            "param_range": param_range,
            "score_name": "Accuracy",
            "n_jobs": -1,
            **kwargs,
        }
        ValidationCurveDisplay.from_estimator(**params)
        plt.title(f"Validation Curve for {self.estimator.__class__.__name__}")
        plt.show()
