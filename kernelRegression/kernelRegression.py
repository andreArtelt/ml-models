import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin


class KernelRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='gauss', sigma=0.1):
        self.X = None
        self.y = None
        self.n_samples = None
        if kernel == 'gauss':
            self.kernel = self.__get_kernel(kernel, sigma)
        else:
            self.kernel = kernel

    def __get_kernel(self, kernel_desc, sigma):
        return lambda x, y: np.exp((-1.0 / (2.0 * sigma**2)) * np.square(np.linalg.norm(x - y)))

    def get_params(self, deep=True):
        return dict({
            'X': self.X,
            'y': self.y,
            'n_samples': self.n_samples,
            'kernel': self.kernel
        })

    def set_params(self, **params):
        self.X = params['X']
        self.y = params['y']
        self.n_samples = params['n_samples']
        self.kernel = params['kernel']

        return self

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]

        return self

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)

        return r2_score(y, y_pred, sample_weight)

    def predict(self, X):
        r = []

        for i in range(X.shape[0]):
            w = []
            y = 0.0
            for j in range(self.n_samples):
                t = self.kernel(X[i], self.X[j])

                y += t * self.y[j]
                w.append(t)
            y /= np.sum(w)

            r.append(y)

        return np.array(r)

