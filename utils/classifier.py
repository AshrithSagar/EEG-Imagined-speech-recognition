"""
classifier.py
Classifier Utility scripts
"""

import importlib.util
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
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
        n_splits=5,
        random_state=42,
        trial_size=None,
        features_names=None,
        features_select_k_best=None,
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
        self.n_splits = n_splits
        self.random_state = random_state
        self.trial_size = trial_size
        self.features_names = np.array(
            [f"Feature-{i}" for i in range(1, X.shape[1] + 1)]
            if features_names is None
            else features_names
        )
        self.features_select_k_best = features_select_k_best
        self.verbose = verbose
        self.console = console if console else Console()
        self.model = None
        self.model_config = None
        self.score_func = None
        self.selected_features_indices = None
        self.X, self.y = None, None

    def get_value(self, value, default):
        """Handle None values"""
        return value if value is not None else default

    def set_verbose(self, verbose=None):
        verbose = self.get_value(verbose, self.verbose)
        return verbose

    def get_model_config(self, model_file=None, reload=False):
        if reload or not self.model_config:
            if not model_file:
                model_file = os.path.join(self.save_dir, "model.py")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file '{model_file}' not found.")
            spec = importlib.util.spec_from_file_location("model", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            self.model_config = model_module

        return self.model_config

    def get_sampler(self, sampler=None):
        if sampler is False:
            return None

        if sampler is None:
            self.get_model_config()
            if not hasattr(self.model_config, "resample"):
                return None
            sampler = self.model_config.resample()
            sampler.random_state = self.random_state

        self.sampler = sampler
        return self.sampler

    def resample(self, X, y, sampler=None, verbose=None):
        verbose = self.set_verbose(verbose)

        if not hasattr(self, "sampler"):
            self.sampler = self.get_sampler(sampler)

        if self.sampler:
            X, y = self.sampler.fit_resample(X, y)

        return X, y

    def get_anova_f(self, X=None, y=None, verbose=None):
        """ANOVA F-Test"""
        verbose = self.set_verbose(verbose)
        X = self.get_value(X, self.X)
        y = self.get_value(y, self.y)

        self.f_statistic, self.p_values = f_classif(X, y)
        self.anova_f = pd.DataFrame(
            {
                "Feature": self.features_names,
                "F-Statistic": self.f_statistic,
                "p-Value": self.p_values,
            },
            index=range(1, len(self.features_names) + 1),
        )

        if verbose:
            table = Table(title="[bold underline]ANOVA F-Test:[/]")
            table.add_column("Feature", justify="right", style="magenta", no_wrap=True)
            table.add_column(
                "F-Statistic", justify="center", style="cyan", no_wrap=True
            )
            table.add_column("p-Value", justify="center", style="cyan", no_wrap=True)

            for i, (f_stat, p_val) in enumerate(zip(self.f_statistic, self.p_values)):
                table.add_row(self.features_names[i], f"{f_stat:.4f}", f"{p_val:.4f}")

            self.console.print(table)

    @staticmethod
    def pearsonr_score(X, y):
        """Absolute Pearson correlation coefficients"""
        num_features = X.shape[1]
        correlation_coeffs = [
            pearsonr(X[:, feature], y)[0] for feature in range(num_features)
        ]
        return np.abs(correlation_coeffs)

    def get_pearsonr(self, X=None, y=None, verbose=None):
        verbose = self.set_verbose(verbose)
        X = self.get_value(X, self.X)
        y = self.get_value(y, self.y)

        self.correlation_coeffs = self.pearsonr_score(X, y)
        self.pearsonr_df = pd.DataFrame(
            {"Feature": self.features_names, "Pearsonr": self.correlation_coeffs},
            index=range(1, len(self.features_names) + 1),
        )

        if verbose:
            table = Table(title="[bold underline]Pearson Correlation:[/]")
            table.add_column("Feature", justify="right", style="magenta", no_wrap=True)
            table.add_column("\u03C1-Value", style="cyan")

            for i, coeff in enumerate(self.correlation_coeffs):
                table.add_row(self.features_names[i], f"{coeff:.4f}")

            self.console.print(table)

    def select_features(self, X=None, y=None, select=None, k_best=None, verbose=None):
        """Select specific features from the feature matrix.

        Parameters:
        - select (list, optional): List of indices of features to select.
        Defaults to selecting all features.
        """
        verbose = self.set_verbose(verbose)
        X = self.get_value(X, self.X)
        y = self.get_value(y, self.y)
        k_best = self.get_value(k_best, self.features_select_k_best)

        def fetch_score_func(score_func):
            self.score_func = None
            if score_func == "pearsonr":
                self.score_func = self.pearsonr_score
            elif score_func == "f_classif":
                self.score_func = f_classif
            return self.score_func

        if k_best is not None:
            fetch_score_func(k_best["score_func"])
            if self.score_func:
                selector = SelectKBest(score_func=self.score_func, k=k_best["k"])
                X = selector.fit_transform(X, y)
                select_indices = selector.get_support(indices=True)
                self.features_names = self.features_names[select_indices]

        X_select = X[:, select] if select is not None else X

        if verbose:
            self.console.print(
                f"[bold underline]Feature selection:[/]\n{X_select.shape[1]} / {X.shape[1]} selected"
            )

        return X_select

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
        y_test = self.get_value(y_test, self.y_test)
        y_pred = self.get_value(y_pred, self.y_pred)

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
            plt.savefig(save_path, bbox_inches="tight")
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

        if hasattr(self, "anova_f"):
            filename = os.path.join(self.save_dir, "anova_f.csv")
            self.anova_f.to_csv(filename, index=True)

        if hasattr(self, "pearsonr_df"):
            filename = os.path.join(self.save_dir, "pearsonr.csv")
            self.pearsonr_df.to_csv(filename, index=True)


class RegularClassifier(ClassifierMixin):
    def __init__(
        self,
        X,
        y,
        save_dir,
        test_size=0.2,
        random_state=42,
        trial_size=None,
        features_names=None,
        features_select_k_best=None,
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
            features_names=features_names,
            features_select_k_best=features_select_k_best,
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

        if self.trial_size is None:
            self.X, self.y = self.X_raw, self.y_raw
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
            self.trial_size = int(self.trial_size * len(self.X_raw))

            # Stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_raw,
                self.y_raw,
                train_size=int(self.trial_size * (1 - self.test_size)),
                test_size=int(self.trial_size * self.test_size),
                random_state=self.random_state,
                stratify=self.y_raw,
            )
            self.X = np.concatenate((X_train, X_test))
            self.y = np.concatenate((y_train, y_test))

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        self.sampler = self.get_sampler(sampler)
        self.X_train, self.y_train = self.resample(X_train, y_train)

        self.X = self.select_features(k_best=self.features_select_k_best)
        self.get_anova_f()
        self.get_pearsonr()
        self.X = self.select_features(select=self.selected_features_indices)
        self.get_scoring()
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.cv = cv if cv else self.model_config.cross_validation()
        self.model.random_state = self.random_state
        self.cv.random_state = self.random_state
        self.cv.test_size = self.test_size
        self.cv.n_splits = self.n_splits

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
        X_test = self.get_value(X, self.X_test)
        y_test = self.get_value(y, self.y_test)

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
        features_names=None,
        features_select_k_best=None,
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
            features_names=features_names,
            features_select_k_best=features_select_k_best,
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

        if self.trial_size is None:
            self.X, self.y = self.X_raw, self.y_raw
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
            self.trial_size = int(self.trial_size * len(self.X_raw))

            # Stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_raw,
                self.y_raw,
                train_size=int(self.trial_size * (1 - self.test_size)),
                test_size=int(self.trial_size * self.test_size),
                random_state=self.random_state,
                stratify=self.y_raw,
            )
            self.X = np.concatenate((X_train, X_test))
            self.y = np.concatenate((y_train, y_test))

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        self.sampler = self.get_sampler(sampler)
        self.X_train, self.y_train = self.resample(X_train, y_train)

        self.X = self.select_features(k_best=self.features_select_k_best)
        self.get_anova_f()
        self.get_pearsonr()
        self.X = self.select_features(select=self.selected_features_indices)
        self.get_scoring(scoring)
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.param_grid = param_grid if param_grid else self.model_config.param_grid()
        self.cv = cv if cv else self.model_config.cross_validation()
        self.model.random_state = self.random_state
        self.cv.random_state = self.random_state
        self.cv.test_size = self.test_size
        self.cv.n_splits = self.n_splits

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
        X_test = self.get_value(X, self.X_test)
        y_test = self.get_value(y, self.y_test)

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
        features_names=None,
        features_select_k_best=None,
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
            features_names=features_names,
            features_select_k_best=features_select_k_best,
            verbose=verbose,
            console=console,
        )

    def compile(self, model=None, sampler=None, cv=None, verbose=None):
        verbose = self.set_verbose(verbose)

        if self.trial_size is None:
            self.X, self.y = self.X_raw, self.y_raw
        elif isinstance(self.trial_size, float) and self.trial_size <= 1.0:
            self.trial_size = int(self.trial_size * len(self.X_raw))

            # Stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_raw,
                self.y_raw,
                train_size=int(self.trial_size * (1 - self.test_size)),
                test_size=int(self.trial_size * self.test_size),
                random_state=self.random_state,
                stratify=self.y_raw,
            )
            self.X = np.concatenate((X_train, X_test))
            self.y = np.concatenate((y_train, y_test))

        self.X = self.select_features(k_best=self.features_select_k_best)
        self.get_anova_f()
        self.get_pearsonr()
        self.X = self.select_features(select=self.selected_features_indices)
        self.get_scoring()
        self.get_model_config()
        self.model = model if model else self.model_config.model()
        self.cv = cv if cv else self.model_config.cross_validation()
        self.model.random_state = self.random_state
        self.cv.random_state = self.random_state
        self.cv.test_size = self.test_size
        self.cv.n_splits = self.n_splits
        self.sampler = self.get_sampler(sampler)

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
        self.confusion_matrices = []

        self.scores = {}
        for metric in self.scoring:
            self.scores[f"test_{metric}"] = []
            if return_train_score:
                self.scores[f"train_{metric}"] = []

        if return_estimator:
            self.scores["estimator"] = []

        fit_times, score_times = [], []
        for train_index, val_index in self.cv.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_val, y_val = X[val_index], y[val_index]

            model = self.model
            self.X_train, self.y_train = self.resample(X_train, y_train)

            start_fit = time.time()
            model.fit(self.X_train, self.y_train)
            fit_times.append(time.time() - start_fit)

            if return_estimator:
                self.scores["estimator"].append(model)

            for metric, scorer in self.scoring.items():
                start_score = time.time()
                score = scorer(model, X_val, y_val)
                score_times.append(time.time() - start_score)
                self.scores[f"test_{metric}"].append(score)

                if return_train_score:
                    train_score = scorer(model, self.X_train, self.y_train)
                    self.scores[f"train_{metric}"].append(train_score)

            self.classes = model.classes_
            y_pred = model.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            self.confusion_matrices.append(cm)

        self.scores["fit_time"] = fit_times
        self.scores["score_time"] = score_times

        verbose = self.set_verbose(verbose)
        if return_train_score:
            self.evaluation_metrics_info(
                scores={k: v for k, v in self.scores.items() if k.startswith("train_")},
                prefix="train",
                show_plots=show_plots,
                verbose=verbose,
            )
        self.evaluation_metrics_info(
            scores={k: v for k, v in self.scores.items() if k.startswith("test_")},
            prefix="test",
            show_plots=show_plots,
            verbose=verbose,
        )

        return self.scores

    def evaluation_metrics_info(
        self, scores=None, prefix=None, show_plots=False, verbose=None
    ):
        verbose = self.set_verbose(verbose)
        scores = scores if scores is not None else self.scores
        prefix = prefix if prefix is not None else "test"

        metrics_mean = {
            metric: np.mean(scores[f"{prefix}_{metric}"]) for metric in self.scoring
        }
        metrics_std = {
            metric: np.std(scores[f"{prefix}_{metric}"]) for metric in self.scoring
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
            table.add_column(
                f"{prefix.capitalize()} Metric",
                justify="right",
                style="magenta",
                no_wrap=True,
            )
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
        test_scores = {k: v for k, v in scores.items() if k.startswith("test_")}

        plt.clf()
        plt.figure(figsize=(15, 8))
        plt.boxplot(
            test_scores.values(),
            labels=[k.lstrip("test_").replace("_", "\n") for k in test_scores.keys()],
        )
        plt.title("Cross-Validation Scores")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def get_confusion_matrix_cv(self, confusion_matrices=None, verbose=None):
        verbose = self.set_verbose(verbose)
        confusion_matrices = self.get_value(confusion_matrices, self.confusion_matrices)

        cm_mean = np.mean(confusion_matrices, axis=0)
        cm_std = np.std(confusion_matrices, axis=0)
        cm_cv = np.empty_like(cm_mean, dtype=object)
        for i in range(cm_mean.shape[0]):
            for j in range(cm_mean.shape[1]):
                cm_cv[i, j] = f"{cm_mean[i, j]:.2f} ± {cm_std[i, j]:.2f}"
        self.confusion_matrix_cv = cm_cv

    def plot_confusion_matrix_cv(
        self, confusion_matrices=None, minimal=False, save_path=None, verbose=None
    ):
        verbose = self.set_verbose(verbose)
        confusion_matrices = self.get_value(confusion_matrices, self.confusion_matrices)

        cm_mean = np.mean(confusion_matrices, axis=0)
        cm_std = np.std(confusion_matrices, axis=0)

        fontsize = 25 if minimal else None
        dpi = 100 if minimal else None
        figsize = (2, 2) if minimal else (8, 6)

        plt.figure(figsize=figsize)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_mean, display_labels=self.classes
        )
        disp.plot(include_values=False, cmap=plt.cm.Blues, ax=ax, colorbar=not minimal)

        for i in range(cm_mean.shape[0]):
            for j in range(cm_mean.shape[1]):
                text_color = "white" if cm_mean[i, j] > np.max(cm_mean) / 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm_mean[i, j]:.2f}\n±{cm_std[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fontsize,
                )

        if minimal:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            fig.tight_layout(pad=0)
            ax.axis("off")
        else:
            plt.title("Confusion Matrix CV")
            plt.tight_layout()

        if save_path:
            pad_inches = 0 if minimal else None
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
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

        filename = os.path.join(self.save_dir, "confusion_matrix")
        for fold, cm in enumerate(self.confusion_matrices):
            self.plot_confusion_matrix(
                cm,
                self.classes,
                save_path=f"{filename}-fold_{fold}.png",
            )

        filename = os.path.join(self.save_dir, "confusion_matrix-cv.png")
        self.plot_confusion_matrix_cv(save_path=filename)

        filename = os.path.join(self.save_dir, "confusion_matrix-cv.minimal.png")
        self.plot_confusion_matrix_cv(minimal=True, save_path=filename)

    def run(self):
        self.compile()
        self.evaluate(
            return_train_score=True,
            return_estimator=False,
            show_plots=False,
            n_jobs=-1,
            verbose=True,
        )
        self.save()
