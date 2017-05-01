from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np


class RDA(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.cov_mat_a = None
        self.cov_mat_b = None
        self.mean_a = None
        self.mean_b = None
        self.log_prob_a = None
        self.log_prob_b = None
        self.cov_mat_a_comp_u = None
        self.cov_mat_a_comp_d = None
        self.cov_mat_a_det = None
        self.cov_mat_b_comp_u = None
        self.cov_mat_b_comp_d = None
        self.cov_mat_b_det = None
        self.add_prob_const = None

    def get_params(self, deep=True):
        return dict({
            'alpha': self.alpha,
            'cov_a': self.cov_mat_a,
            'mean_a': self.mean_a,
            'log_prob_a': self.log_prob_a,
            'cov_mat_a_comp_u': self.cov_mat_a_comp_u,
            'cov_mat_a_comp_d': self.cov_mat_a_comp_d,
            'cov_mat_a_det': self.cov_mat_a_det,
            'cov_b': self.cov_mat_b,
            'mean_b': self.mean_b,
            'log_prob_b': self.log_prob_b,
            'cov_mat_b_comp_u': self.cov_mat_b_comp_u,
            'cov_mat_b_comp_d': self.cov_mat_b_comp_d,
            'cov_mat_b_det': self.cov_mat_b_det,
            'add_prob_const': self.add_prob_const
        })

    def set_params(self, **params):
        self.alpha = params['alpha']
        self.cov_mat_a = params['cov_mat_a']
        self.cov_mat_b = params['cov_mat_b']
        self.mean_a = params['mean_a']
        self.mean_b = params['mean_b']
        self.log_prob_a = params['log_prob_a']
        self.log_prob_b = params['log_prob_b']
        self.cov_mat_a_comp_u = params['cov_mat_a_comp_u']
        self.cov_mat_a_comp_d = params['cov_mat_a_comp_d']
        self.cov_mat_b_comp_u = params['cov_mat_b_comp_u']
        self.cov_mat_b_comp_d = params['cov_mat_b_comp_d']
        self.cov_mat_a_det = params['cov_mat_a_det']
        self.cov_mat_b_det = params['cov_mat_b_det']
        self.add_prob_const = params['add_prob_const']
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
        self.mean_a = np.mean(X[y == 0, :], axis=0)
        self.mean_b = np.mean(X[y == 1, :], axis=0)

        y_count = np.bincount(y)
        self.log_prob_a = np.log((1.0 * y_count[0]) / len(y))
        self.log_prob_b = np.log((1.0 * y_count[1]) / len(y))

        dim = X.shape[1]
        self.add_prob_const = - (dim / 2.0) * np.log((2.0*np.pi))

        cov_mat = np.cov(np.transpose(X))
        self.cov_mat_a = self.alpha * np.cov(np.transpose(X[y == 0])) \
                         + (1 - self.alpha) * cov_mat
        self.cov_mat_b = self.alpha * np.cov(np.transpose(X[y == 1])) \
                         + (1 - self.alpha) * cov_mat

        self.cov_mat_a_comp_d, self.cov_mat_a_comp_u = np.linalg.eig(self.cov_mat_a)
        self.cov_mat_a_det = np.sum(np.log(self.cov_mat_a_comp_d))
        self.cov_mat_a_comp_d = np.diag(1.0 / self.cov_mat_a_comp_d)
        self.cov_mat_a_comp_u = np.transpose(self.cov_mat_a_comp_u)

        self.cov_mat_b_comp_d, self.cov_mat_b_comp_u = np.linalg.eig(self.cov_mat_b)
        self.cov_mat_b_det = np.sum(np.log(self.cov_mat_b_comp_d))
        self.cov_mat_b_comp_d = np.diag(1.0 / self.cov_mat_b_comp_d)
        self.cov_mat_b_comp_u = np.transpose(self.cov_mat_b_comp_u)

        return self

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        if X is None:
            raise ValueError('X must not be None')

        axis = len(X.shape) - 1 # Axis to sum over when computing the dot product

        t_a = np.dot(X - self.mean_a, self.cov_mat_a_comp_u)
        lp_a = -0.5 * np.sum((t_a * np.dot(t_a, np.transpose(self.cov_mat_a_comp_d))), axis=axis) \
               + self.log_prob_a - 0.5 * self.cov_mat_a_det + self.add_prob_const

        t_b = np.dot((X - self.mean_b), self.cov_mat_b_comp_u)
        lp_b = -0.5 * np.sum((t_b * np.dot(t_b, np.transpose(self.cov_mat_b_comp_d))), axis=axis) \
              + self.log_prob_b - 0.5 * self.cov_mat_b_det + self.add_prob_const

        return np.transpose(np.vstack((lp_a, lp_b)))
