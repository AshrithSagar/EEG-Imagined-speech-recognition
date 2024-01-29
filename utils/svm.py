"""
svm.py
Support Vector Machine Classifier Utility scripts
"""
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(
        self,
        kernel="linear",
        test_size=0.2,
        random_state=42,
        verbose=False,
        console=None,
    ):
        self.kernel = kernel
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.verbose = verbose
        self.console = console if console else Console()

    def train(self, X, y, verbose=None):
        verbose = verbose if verbose is not None else self.verbose

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model = SVC(kernel=self.kernel)
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def evaluate(self, X_test=None, y_test=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test

        if self.model is None:
            self.console.print(
                "Model has not been trained yet. Please train the model first."
            )
            return

        self.y_pred = self.model.predict(X_test)
        self.get_metrics(verbose=verbose)

    def get_metrics(self, y_test=None, y_pred=None, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        y_test = y_test if y_test is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

        if verbose:
            self.metrics_info()

    def metrics_info(self, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            table = Table(title="Evaluation Metrics")
            table.add_column("Metric")
            table.add_column("Value")
            for metric, value in self.metrics.items():
                table.add_row(metric, str(value))
            self.console.print(table)
