"""
models.py
Model classes
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HanmanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, alpha=None, beta=None, a=None, b=None, q=None):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.q = q

    def __repr__(self):
        return (
            f"HanmanClassifier("
            f"alpha={self.alpha}, beta={self.beta}, a={self.a}, b={self.b}, q={self.q}"
            f")"
        )

    @staticmethod
    def frank_t_norm(a, b, q):
        numerator = (q**a - 1) * (q**b - 1)
        denominator = q - 1
        if denominator == 0:
            return 0  # Handle division by zero
        return np.log1p(numerator / denominator) / np.log(q)

    def fit(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.X_, self.y_ = X_train, y_train
        self.n_features_in_ = X_train.shape[1]

        self.is_fitted_ = True
        return self

    def predict(self, X_test):
        check_is_fitted(self, ["X_", "y_"])
        X_test = check_array(X_test)

        X_train, y_train = self.X_, self.y_
        X_train = minmax_scale(X_train, axis=1)
        X_test = minmax_scale(X_test, axis=1)

        n_test = X_test.shape[0]
        entropies = np.zeros((len(self.classes_), n_test))
        for l, cls in enumerate(self.classes_):
            X_cls = X_train[y_train == cls]

            n_cls = X_cls.shape[0]
            for t, x in enumerate(X_test):
                error = np.abs(X_cls - x)

                norm_error = np.ones((n_cls, n_cls, self.n_features_in_))
                for i in range(n_cls):
                    for j in range(n_cls):
                        if i == j:
                            continue
                        norm_error[i][j] = self.frank_t_norm(error[i], error[j], self.q)

                min_norm_error = np.min(norm_error, axis=(0, 1))

                possibilistic_uncertainty = np.sum(
                    min_norm_error**self.alpha
                    * np.exp(-((self.a * min_norm_error + self.b) ** self.beta))
                )
                entropies[l][t] = possibilistic_uncertainty

        y_pred = self.classes_[np.argmin(entropies, axis=0)]
        return y_pred
