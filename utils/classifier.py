"""
classifier.py
Classifier Utility scripts
"""

import importlib.util
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split


class Classifier:
    def __init__(
        self,
        X,
        y,
        save_dir,
        test_size=0.2,
        random_state=42,
        trial_size=None,
        run_grid_search=False,
        verbose=False,
        console=None,
    ):
        """Parameters:
        - trial_size (int | float | None): Only use part of the dataset for trial.
            (default: Entire dataset)
        """
        self.X_raw, self.y_raw = X, y
        self.save_dir = save_dir
        self.test_size = test_size
        self.random_state = random_state
        self.trial_size = trial_size
        self.run_grid_search = run_grid_search
        self.verbose = verbose
        self.console = console if console else Console()
        self.model = None
        self.model_config = None

    def get_model_config(self, model_file=None, reload=False):
        if reload or not self.model_config:
            if not model_file:
                model_file = os.path.join(self.save_dir, "model.py")
            spec = importlib.util.spec_from_file_location("model", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            self.model_config = model_module

    def compile(self, model=None, sampler=None, cv=None, load=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        X, y = self.resample(self.X_raw, self.y_raw, sampler)

        if isinstance(self.trial_size, float) and self.trial_size <= 1:
            self.trial_size = int(self.trial_size * len(X))

        # Stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=int(self.trial_size * (1 - self.test_size)),
            test_size=int(self.trial_size * self.test_size),
            random_state=self.random_state,
            stratify=y,
        )
        self.X = np.concatenate((X_train, X_test))
        self.y = np.concatenate((y_train, y_test))

        # Train:Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        self.scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(
                precision_score, average="binary", zero_division=0
            ),
            "precision_macro": make_scorer(
                precision_score, average="macro", zero_division=0
            ),
            "precision_micro": make_scorer(
                precision_score, average="micro", zero_division=0
            ),
            "precision_weighted": make_scorer(
                precision_score, average="weighted", zero_division=0
            ),
            "recall": make_scorer(recall_score, average="binary", zero_division=0),
            "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
            "recall_micro": make_scorer(recall_score, average="micro", zero_division=0),
            "recall_weighted": make_scorer(
                recall_score, average="weighted", zero_division=0
            ),
            "f1": make_scorer(f1_score, average="binary", zero_division=0),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_micro": make_scorer(f1_score, average="micro", zero_division=0),
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
            "roc_auc": make_scorer(roc_auc_score),
            "matthews_corrcoef": make_scorer(matthews_corrcoef),
            "cohen_kappa": make_scorer(cohen_kappa_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
        }

        if model:
            self.model = model
        else:
            self.get_model_config()
            self.model = self.model_config.model()

        if cv:
            self.cv = cv
        else:
            self.get_model_config()
            self.cv = self.model_config.cross_validation()

        if load:
            filename = os.path.join(self.save_dir, "model.joblib")
            self.model = joblib.load(filename)

        if verbose:
            self.params_info()
            self.model_info()

    def resample(self, X, y, sampler=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        self.sampler = None

        if sampler is False:
            return X, y

        if sampler is None:
            self.get_model_config()
            if not hasattr(self.model_config, "resample"):
                return X, y
            sampler = self.model_config.resample()

        self.sampler = sampler
        X, y = self.sampler.fit_resample(X, y)
        return X, y

    def train(self, model=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if self.run_grid_search:
            self.perform_grid_search(verbose=10)
            # self.grid_search_info(verbose=verbose)
        else:
            self.model = model if model is not None else self.model
            self.model.fit(self.X_train, self.y_train)

    def evaluate(
        self,
        X=None,
        y=None,
        return_train_score=False,
        return_estimator=False,
        show_plots=False,
        n_jobs=None,
        verbose=None,
    ):
        X = X if X is not None else self.X_train
        y = y if y is not None else self.y_train
        self.predict(X_test=X, y_test=y, verbose=False or verbose)
        verbose = verbose if verbose is not None else self.verbose
        self.get_metrics(y_test=y, show_plots=show_plots, verbose=verbose)

        self.scores = cross_validate(
            estimator=self.model,
            X=X,
            y=y,
            scoring=self.scoring,
            cv=self.cv,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            n_jobs=n_jobs,
            verbose=False or verbose,
        )
        self.evaluation_metrics_info(verbose=verbose)

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
            self.get_model_config()
            param_grid = self.model_config.param_grid()

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
        roc_auc = roc_auc_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        self.metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "roc_auc": float(roc_auc),
            "classification_report": classification_rep,
            "matthews_corrcoef": float(mcc),
            "cohen_kappa": float(kappa),
            "balanced_accuracy": float(balanced_accuracy),
        }

        self.metrics_info(show_plots=show_plots, verbose=verbose)

    def format_metrics(self, metrics=None, as_str=True):
        metrics = metrics if metrics is not None else self.metrics

        format_options = {
            "accuracy": lambda x: f"{x:.2%}" if as_str else round(x, 4),
            "precision": lambda x: f"{x:.2%}" if as_str else round(x, 4),
            "recall": lambda x: f"{x:.2%}" if as_str else round(x, 4),
            "f1_score": lambda x: f"{x:.2%}" if as_str else round(x, 4),
            "roc_auc": lambda x: f"{x:.2%}" if as_str else round(x, 4),
            "matthews_corrcoef": lambda x: (f"{x:.4f}" if as_str else round(x, 4)),
            "cohen_kappa": lambda x: f"{x:.4f}" if as_str else round(x, 4),
            "balanced_accuracy": lambda x: (f"{x:.2%}" if as_str else round(x, 4)),
        }

        return {
            metric: format_options.get(metric, str)(value)
            for metric, value in metrics.items()
            if metric not in ["confusion_matrix", "classification_report"]
        }

    def plot_confusion_matrix(
        self,
        confusion_matrix,
        classes,
        save_path=None,
        normalize=False,
        cmap=plt.cm.Blues,
    ):
        if normalize:
            confusion_matrix = (
                confusion_matrix.astype("float")
                / confusion_matrix.sum(axis=1)[:, np.newaxis]
            )

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=classes
        )
        disp.plot(cmap=cmap)
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def evaluation_metrics_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        metrics_mean = {
            metric: np.mean(self.scores[f"test_{metric}"]) for metric in self.scoring
        }
        metrics_std = {
            metric: np.std(self.scores[f"test_{metric}"]) for metric in self.scoring
        }

        if verbose:
            table = Table(title="[bold underline]Evaluation Metrics:[/]")
            table.add_column("Metric", justify="right", style="magenta", no_wrap=True)
            table.add_column("Mean ± Std", justify="center", style="cyan", no_wrap=True)
            for metric, mean, std in zip(
                self.scoring.keys(), metrics_mean.values(), metrics_std.values()
            ):
                table.add_row(metric, f"{mean:.4f} ± {std:.4f}")

            self.console.print(table)

    def metrics_info(self, show_plots=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        if verbose:
            table = Table(title="[bold underline]Evaluation Metrics:[/]")
            table.add_column("Metric", justify="right", style="magenta", no_wrap=True)
            table.add_column("Value", justify="left", style="cyan", no_wrap=True)
            for metric, value in self.format_metrics().items():
                table.add_row(metric, value)

            self.console.print(table)

        if show_plots:
            self.plot_confusion_matrix(self.metrics["confusion_matrix"], self.classes)

    def get_params(self):
        self.params = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "trial_size": self.trial_size,
            "model": str(self.model),
            "sampler": str(self.sampler),
            "data": {
                "train": str(self.X_train.shape),
                "test": str(self.X_test.shape),
            },
            "run_grid_search": self.run_grid_search,
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
        os.makedirs(self.save_dir, exist_ok=True)
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
        with open(filename, "w") as file:
            yaml.dump(self.format_metrics(as_str=False), file, default_flow_style=False)

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

        if hasattr(self, "scores"):
            filename = os.path.join(self.save_dir, "scores.joblib")
            joblib.dump(self.scores, filename)
