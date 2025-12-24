"""
models.py
Model classes
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class HanmanClassifier(BaseEstimator, ClassifierMixin):
    """Hanman Classifier.

    For more information, refer here
    - https://doi.org/10.1016/j.eswa.2014.03.040
    - https://doi.org/10.1007/s00500-019-04277-9

    Parameters
    ----------
    alpha : float, optional (default=None)
        The alpha parameter.
    beta : float, optional (default=None)
        The beta parameter.
    a : float, optional (default=None)
        The a parameter.
    b : float, optional (default=None)
        The b parameter.
    q : float, optional (default=None)
        The q parameter.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for `predict`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during `fit`.
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values (class labels) associated with the training input samples.
    n_features_in_ : int
        The number of features seen during `fit`.
    X_cls : list of ndarrays, each of shape (n_samples_cls, n_features)
        The normalized training data for each class.

    """

    def __init__(
        self, *, alpha=None, beta=None, a=None, b=None, q=None, n_jobs=1, verbose=None
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.q = q
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __repr__(self):
        return (
            f"HanmanClassifier("
            f"alpha={self.alpha}, beta={self.beta}, a={self.a}, b={self.b}, q={self.q}"
            f")"
        )

    @staticmethod
    def frank_t_norm(a, b, q):
        """Compute the Frank t-norm.

        Parameters
        ----------
        a : float
            The first parameter.
        b : float
            The second parameter.
        q : float
            The q parameter.

        Returns
        -------
        float
            The computed Frank t-norm.

        """

        numerator = (q**a - 1) * (q**b - 1)

        denominator = q - 1
        if denominator == 0:
            return 0  # Handle division by zero

        return np.log1p(numerator / denominator) / np.log(q)

    def fit(self, X_train, y_train):
        """Fit the classifier to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels) associated with the input samples.

        Returns
        -------
        self
            Returns an instance of the estimator.

        """

        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.X_, self.y_ = X_train, y_train
        self.n_features_in_ = X_train.shape[1]

        # Pre-compute the normalized training data for each class
        self.X_cls = [
            minmax_scale(self.X_, axis=1)[self.y_ == cls] for cls in self.classes_
        ]

        self.is_fitted_ = True
        return self

    def predict(self, X_test):
        """Predict class labels for samples in X_test.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted class labels for the input samples.

        """

        check_is_fitted(self, ["X_", "y_"])
        X_test = check_array(X_test)

        X_test = minmax_scale(X_test, axis=1)

        y_pred = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_sample)(sample) for sample in X_test
        )

        return np.array(y_pred)

    def _predict_sample(self, sample):
        """Predict the class label for a single sample.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            The input sample, assuming MinMax scaled along features.

        Returns
        -------
        int
            The predicted class label for the input sample.

        """

        entropies = np.zeros(len(self.classes_))
        for cls_idx, X_cls in enumerate(self.X_cls):
            error = np.abs(X_cls - sample)

            norm_error = self.frank_t_norm(error[:, None], error[None, :], self.q)
            min_norm_error = np.min(norm_error, axis=(0, 1))

            possibilistic_uncertainty = np.sum(
                min_norm_error**self.alpha
                * np.exp(-((self.a * min_norm_error + self.b) ** self.beta))
            )
            entropies[cls_idx] = possibilistic_uncertainty

        return np.argmin(entropies, axis=0)
