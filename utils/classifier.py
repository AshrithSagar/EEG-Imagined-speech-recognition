"""
classifier.py
Classifier Utility scripts
"""

import importlib.util
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


class Classifier:
    def __init__(
        self,
        X,
        y,
        save_dir,
        test_size=0.2,
        random_state=42,
        trial_size=None,
        verbose=False,
        console=None,
    ):
        """Parameters:
        - trial_size (int): Only use part of the dataset for trial (default: Entire dataset)
        """
        self.X, self.y = X, y
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state
        self.trial_size = trial_size
        self.verbose = verbose
        self.console = console if console else Console()
        self.model = None

    def compile(self, model=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        X = self.X[: self.trial_size]
        y = self.y[: self.trial_size]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if model:
            self.model = model
        else:
            model_file = os.path.join(self.save_dir, "model.py")
            spec = importlib.util.spec_from_file_location("model", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            self.model = model_module.model()

        if verbose:
            self.params_info()
            self.model_info()

    def train(self, model=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        self.model = model if model is not None else self.model
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, X_test=None, y_test=None, show_plots=False, verbose=None):
        self.predict(X_test=X_test, y_test=y_test, verbose=False)
        self.get_metrics(show_plots=show_plots, verbose=verbose)

    def predict(self, X_test=None, y_test=None, model=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test

        self.get_model(model)
        self.y_pred = self.model.predict(X_test)

        if verbose:
            table = Table(title="[bold underline]Predictions:[/]")
            table.add_column(
                "True Label", justify="right", style="magenta", no_wrap=True
            )
            table.add_column("Prediction", justify="left", style="cyan", no_wrap=True)
            for label, pred in zip(y_test, self.y_pred):
                table.add_row(str(label), str(pred))

            self.console.print(table)

    def perform_grid_search(
        self, param_grid=None, scoring="accuracy", cv=5, verbose=None
    ):
        verbose = verbose if verbose is not None else self.verbose

        if param_grid is None:
            model_file = os.path.join(self.save_dir, "model.py")
            spec = importlib.util.spec_from_file_location("model", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            param_grid = model_module.param_grid()

        self.grid_search = GridSearchCV(
            self.model, param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=-1
        )
        self.grid_search.fit(self.X_train, self.y_train)

        if verbose:
            self.console.print(f"Best Parameters: {self.grid_search.best_params_}")
            self.console.print(
                f"Best {scoring.capitalize()}: {self.grid_search.best_score_}"
            )

        return self.grid_search

    def grid_search_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            params = self.grid_search.cv_results_["params"]

            table = Table(title="[bold underline]Grid Search Results:[/]")
            table.add_column(
                "Parameter", justify="right", style="magenta", no_wrap=True
            )
            for idx, param in enumerate(params):
                header = "\n".join(f"{key}={value}" for key, value in param.items())
                table.add_column(header, justify="left", style="cyan", no_wrap=True)

            for param, value in self.grid_search.cv_results_.items():
                if param == "params":
                    continue
                table.add_row(param, *[f"{val:g}" for val in value])

            self.console.print(table)

    def get_model(self, model=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if model:
            self.model = model
            # self.classes
            return self.model

        if (
            hasattr(self, "grid_search")
            and hasattr(self.grid_search, "cv_results_")
            and self.grid_search.cv_results_ is not None
            and hasattr(self.grid_search, "best_estimator_")
        ):
            self.model = self.grid_search.best_estimator_
            self.classes = self.grid_search.classes_
            if verbose:
                self.console.print(f"Best Parameters: {self.grid_search.best_params_}")
            return self.model

        self.classes = self.model.classes_
        return self.model

    def get_metrics(self, y_test=None, y_pred=None, show_plots=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        y_test = y_test if y_test is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm,
        }

        self.metrics_info(show_plots=show_plots, verbose=verbose)

    def plot_confusion_matrix(self, confusion_matrix, labels, save_path=None):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def metrics_info(self, show_plots=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            table = Table(title="[bold underline]Evaluation Metrics:[/]")
            table.add_column("Metric", justify="right", style="magenta", no_wrap=True)
            table.add_column("Value", justify="left", style="cyan", no_wrap=True)
            for metric, value in self.metrics.items():
                if metric != "confusion_matrix":
                    value_str = f"{value:.2%}"
                    table.add_row(metric, value_str)

            self.console.print(table)

        if show_plots:
            self.plot_confusion_matrix(self.metrics["confusion_matrix"], self.classes)

    def get_params(self):
        self.params = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "trial_size": self.trial_size,
            "model": str(self.model),
            "data": {
                "train": str(self.X_train.shape),
                "test": str(self.X_test.shape),
            },
        }

        return self.params

    def params_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        self.get_params()

        if verbose:
            table = Table(title="[bold underline]Parameters:[/]")
            table.add_column(
                "Parameter", justify="right", style="magenta", no_wrap=True
            )
            table.add_column("Value", justify="left", style="cyan", no_wrap=True)
            for param, value in self.params.items():
                if param == "data":
                    for data, shape in value.items():
                        table.add_row(data, shape)
                else:
                    table.add_row(param, str(value))

            self.console.print(table)

    def model_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            table = Table(title="[bold underline]Model Parameters:[/]")
            table.add_column(
                "Parameter", justify="right", style="magenta", no_wrap=True
            )
            table.add_column("Value", justify="left", style="cyan", no_wrap=True)
            for param, value in self.model.get_params().items():
                table.add_row(param, str(value))

            self.console.print(table)

    def save(self, verbose=False):
        verbose = verbose if verbose is not None else self.verbose
        self.get_model(verbose=False)

        filename = os.path.join(self.save_dir, "params.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.get_params(), file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "model_params.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.model.get_params(), file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "model.joblib")
        joblib.dump(self.model, filename)

        filename = os.path.join(self.save_dir, "metrics.yaml")
        metrics = {
            key: f"{value:.2%}"
            for key, value in self.metrics.items()
            if key != "confusion_matrix"
        }
        with open(filename, "w") as file:
            yaml.dump(metrics, file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(
            self.metrics["confusion_matrix"], self.classes, save_path=filename
        )

        if (
            hasattr(self, "grid_search")
            and hasattr(self.grid_search, "cv_results_")
            and self.grid_search.cv_results_ is not None
            and hasattr(self.grid_search, "best_estimator_")
        ):
            gs = {
                "best_params": self.grid_search.best_params_,
                "best_score": float(f"{self.grid_search.best_score_:g}"),
            }
            filename = os.path.join(self.save_dir, "grid_search.yaml")
            with open(filename, "w") as file:
                yaml.dump(gs, file, default_flow_style=False)

            filename = os.path.join(self.save_dir, "grid_search.joblib")
            joblib.dump(self.grid_search, filename)