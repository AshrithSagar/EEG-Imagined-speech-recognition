"""
classifier.py
Classifier Utility scripts
"""

import importlib.util
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


class ClassifierMixin:
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
        - trial_size (int | float | None): Only use part of the dataset for trial.
            (default: Entire dataset)
        """
        self.X_raw, self.y_raw = X, y
        self.save_dir = save_dir
        self.test_size = test_size
        self.random_state = random_state
        self.trial_size = trial_size
        self.verbose = verbose
        self.console = console if console else Console()
        self.model = None
        self.model_config = None

    def set_verbose(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        return verbose

    def get_model_config(self, model_file=None, reload=False):
        if reload or not self.model_config:
            if not model_file:
                model_file = os.path.join(self.save_dir, "model.py")
            spec = importlib.util.spec_from_file_location("model", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            self.model_config = model_module

        return self.model_config

    def resample(self, X, y, sampler=None, verbose=None):
        verbose = self.set_verbose(verbose)
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

    def get_scoring(self, scoring=None):
        if scoring is not None:
            self.scoring = scoring
            return self.scoring

        self.scoring = {
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "cohen_kappa": make_scorer(cohen_kappa_score),
            "f1": make_scorer(f1_score, average="binary", zero_division=0),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_micro": make_scorer(f1_score, average="micro", zero_division=0),
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
            "matthews_corrcoef": make_scorer(matthews_corrcoef),
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
            "roc_auc": make_scorer(roc_auc_score),
        }

        return self.scoring

    def get_metrics(self, y_test=None, y_pred=None, show_plots=False, verbose=None):
        verbose = self.set_verbose(verbose)
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

    def metrics_info(self, show_plots=False, verbose=None):
        verbose = self.set_verbose(verbose)

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
            "test_size": getattr(self, "test_size", None),
            "random_state": getattr(self, "random_state", None),
            "trial_size": getattr(self, "trial_size", None),
            "model": str(self.model) if hasattr(self, "model") else None,
            "sampler": str(self.sampler) if hasattr(self, "sampler") else None,
            "param_grid": getattr(self, "param_grid", None),
            "cv": str(self.cv) if hasattr(self, "cv") else None,
            "data": {
                "train": str(self.X_train.shape) if hasattr(self, "X_train") else None,
                "test": str(self.X_test.shape) if hasattr(self, "X_test") else None,
            },
        }

        return self.params

    def params_info(self, verbose=None):
        verbose = self.set_verbose(verbose)
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
        verbose = self.set_verbose(verbose)

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
        verbose = self.set_verbose(verbose)
        os.makedirs(self.save_dir, exist_ok=True)

        filename = os.path.join(self.save_dir, "params.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.get_params(), file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "model_params.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.model.get_params(), file, default_flow_style=False)


class RegularClassifier(ClassifierMixin):
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
        super().__init__(
            X=X,
            y=y,
            save_dir=save_dir,
            test_size=test_size,
            random_state=random_state,
            trial_size=trial_size,
            verbose=verbose,
            console=console,
        )

    def get_model(self, model=None, verbose=None):
        verbose = self.set_verbose(verbose)

        if model:
            self.model = model
            # self.classes
            return self.model

        self.classes = self.model.classes_
        return self.model

    def compile(self, model=None, sampler=None, cv=None, verbose=None):
        verbose = self.set_verbose(verbose)

        X, y = self.resample(self.X_raw, self.y_raw, sampler)

        if self.trial_size is None:
            self.trial_size = len(X)
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        self.get_scoring()
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.cv = cv if cv else self.model_config.cross_validation()

        if verbose:
            self.params_info()
            self.model_info()

    def train(self, verbose=None):
        verbose = self.set_verbose(verbose)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, X=None, y=None, model=None, show_plots=False, verbose=None):
        self.predict(X=X, y=y, verbose=verbose if verbose is not None else False)
        verbose = self.set_verbose(verbose)
        self.get_metrics(show_plots=show_plots, verbose=verbose)

    def predict(self, X=None, y=None, model=None, verbose=None):
        verbose = self.set_verbose(verbose)
        X_test = X if X is not None else self.X_test
        y_test = y if y is not None else self.y_test

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

    def save(self, verbose=False):
        super().save(verbose)
        self.get_model(verbose=False)

        filename = os.path.join(self.save_dir, "model.joblib")
        joblib.dump(self.model, filename)

        filename = os.path.join(self.save_dir, "metrics.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.format_metrics(as_str=False), file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(
            self.metrics["confusion_matrix"], self.classes, save_path=filename
        )

    def run(self):
        self.compile()
        self.train()
        self.evaluate()
        self.save()


class ClassifierGridSearch(ClassifierMixin):
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
        super().__init__(
            X=X,
            y=y,
            save_dir=save_dir,
            test_size=test_size,
            random_state=random_state,
            trial_size=trial_size,
            verbose=verbose,
            console=console,
        )

    def get_model(self, model=None, verbose=None):
        verbose = self.set_verbose(verbose)

        if model:
            self.model = model
            # self.classes
            return self.model

        self.model = self.grid_search.best_estimator_
        self.classes = self.grid_search.classes_

        if verbose:
            self.console.print(f"Best Parameters: {self.grid_search.best_params_}")

        return self.model

    def compile(
        self,
        model=None,
        sampler=None,
        param_grid=None,
        scoring=None,
        cv=None,
        verbose=None,
    ):
        verbose = self.set_verbose(verbose)

        X, y = self.resample(self.X_raw, self.y_raw, sampler)

        if self.trial_size is None:
            self.trial_size = len(X)
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        self.get_scoring(scoring)
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.param_grid = param_grid if param_grid else self.model_config.param_grid()
        self.cv = cv if cv else self.model_config.cross_validation()

        if verbose:
            self.params_info()
            self.model_info()

    def train(self, n_jobs=None, verbose=None):
        verbose = self.set_verbose(verbose)

        self.grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.grid_search.fit(self.X_train, self.y_train)

        if verbose:
            self.console.print(
                f"Best {self.scoring.capitalize()}: {self.grid_search.best_score_:g}"
            )

        return self.grid_search

    def evaluate(self, X=None, y=None, model=None, show_plots=False, verbose=None):
        self.predict(X=X, y=y, verbose=verbose if verbose is not None else False)
        verbose = self.set_verbose(verbose)
        self.get_metrics(show_plots=show_plots, verbose=verbose)

    def predict(self, X=None, y=None, model=None, verbose=None):
        verbose = self.set_verbose(verbose)
        X_test = X if X is not None else self.X_test
        y_test = y if y is not None else self.y_test

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

    def grid_search_info(self, verbose=None):
        verbose = self.set_verbose(verbose)
        params = self.grid_search.cv_results_["params"]

        if verbose:
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

    def save(self, verbose=False):
        super().save(verbose)
        self.get_model(verbose=False)

        filename = os.path.join(self.save_dir, "model.joblib")
        joblib.dump(self.model, filename)

        filename = os.path.join(self.save_dir, "metrics.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.format_metrics(as_str=False), file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(
            self.metrics["confusion_matrix"], self.classes, save_path=filename
        )

        gs = {
            "best_params": self.grid_search.best_params_,
            "best_score": float(f"{self.grid_search.best_score_:g}"),
        }
        filename = os.path.join(self.save_dir, "grid_search.yaml")
        with open(filename, "w") as file:
            yaml.dump(gs, file, default_flow_style=False)

        filename = os.path.join(self.save_dir, "grid_search.joblib")
        joblib.dump(self.grid_search, filename)

    def run(self):
        self.compile(scoring="accuracy")
        self.train()
        self.evaluate()
        self.grid_search_info()
        self.save()


class EvaluateClassifier(ClassifierMixin):
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
        super().__init__(
            X=X,
            y=y,
            save_dir=save_dir,
            test_size=test_size,
            random_state=random_state,
            trial_size=trial_size,
            verbose=verbose,
            console=console,
        )

    def compile(self, model=None, sampler=None, cv=None, verbose=None):
        verbose = self.set_verbose(verbose)

        X, y = self.resample(self.X_raw, self.y_raw, sampler)

        if self.trial_size is None:
            self.trial_size = len(X)
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
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

        self.get_scoring()
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.cv = cv if cv else self.model_config.cross_validation()

        if verbose:
            self.params_info()
            self.model_info()

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
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        self.scores = cross_validate(
            estimator=self.model,
            X=X,
            y=y,
            scoring=self.scoring,
            cv=self.cv,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            n_jobs=n_jobs,
            verbose=verbose if verbose is not None else False,
        )
        verbose = self.set_verbose(verbose)
        self.evaluation_metrics_info(show_plots=show_plots, verbose=verbose)

    def evaluation_metrics_info(self, show_plots=False, verbose=None):
        verbose = self.set_verbose(verbose)

        metrics_mean = {
            metric: np.mean(self.scores[f"test_{metric}"]) for metric in self.scoring
        }
        metrics_std = {
            metric: np.std(self.scores[f"test_{metric}"]) for metric in self.scoring
        }

        self.cv_metrics_df = pd.DataFrame(
            {
                "Metric": list(self.scoring.keys()),
                "Mean": list(metrics_mean.values()),
                "Std": list(metrics_std.values()),
            }
        )

        if verbose:
            table = Table(title="[bold underline]Evaluation Metrics:[/]")
            table.add_column("Metric", justify="right", style="magenta", no_wrap=True)
            table.add_column("Mean ± Std", justify="center", style="cyan", no_wrap=True)
            for metric, mean, std in zip(
                self.scoring.keys(),
                metrics_mean.values(),
                metrics_std.values(),
            ):
                table.add_row(metric, f"{mean:.4f} ± {std:.4f}")

            self.console.print(table)

        if show_plots:
            self.plot_scores_boxplot()

    def plot_scores_boxplot(self, scores=None, save_path=None):
        scores = scores if scores is not None else self.scores

        test_scores = {
            key: value for key, value in scores.items() if key.startswith("test_")
        }

        plt.clf()
        plt.figure(figsize=(15, 8))
        plt.boxplot(
            test_scores.values(),
            labels=[
                key.lstrip("test_").replace("_", "\n") for key in test_scores.keys()
            ],
        )
        plt.title("Cross-Validation Scores")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def save(self, verbose=False):
        super().save(verbose)

        filename = os.path.join(self.save_dir, "scores.joblib")
        joblib.dump(self.scores, filename)

        filename = os.path.join(self.save_dir, "cv_metrics.csv")
        self.cv_metrics_df.to_csv(filename, index=False)

        filename = os.path.join(self.save_dir, "cv_scores_boxplot.png")
        self.plot_scores_boxplot(save_path=filename)

    def run(self):
        self.compile()
        self.evaluate(n_jobs=-1, show_plots=False, verbose=True)
        self.save()
