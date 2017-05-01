from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np


class FDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.w = None

    def get_params(self, deep=True):
        return dict({
            'w': self.w
        })

    def set_params(self, **params):
        self.w = params['w']
        return self

    def score(self, X, y, sample_weight=None):
        if X is None or y is None:
            raise ValueError('Arguments must not be None')

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError('Arguments must not be None')
        if len(y.shape) > 1:
            raise ValueError('Expect y to be 1-dimensional')
        if X.shape[0] != y.shape[0]:
            raise ValueError('#Rows in X does not match #rows in y')
        if len(np.unique(y)) > 2:
            raise ValueError('y must not contain more than two labels')
        if 0 not in y or 1 not in y:    # ATTENTION: Each class needs one sample at least!
            raise ValueError('y  must not contain different labels than 0 and 1')

        # Fit
        cov_mat = np.cov(np.transpose(X[y == 0, :])) +\
                  np.cov(np.transpose(X[y == 1, :]))

        mean_a = np.mean(X[y == 0, :], axis=0)
        mean_b = np.mean(X[y == 1, :], axis=0)

        self.w = np.dot(np.linalg.inv(cov_mat), (mean_b - mean_a))

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if X is None:
            raise ValueError('X must not be None')

        pred = np.dot(X, self.w)
        return np.transpose(np.vstack((pred <= 0, pred > 0)).astype(np.float32))
