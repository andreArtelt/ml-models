import numpy as np
from scipy.stats import pearsonr


class IFSR:
    """IFSR
    Incremental forward stagewise regression (IFSR).
    Parameters
    ----------
    input_dim : int
        Dimensions of input.
    max_steps : int, default=100, optional
        Maximum number of steps/iterations.
    epsilon : float, default=0.01
        Parameter of the algorithm.
    tol : float, default=Negative infinity
        Controlls early stopping.
    callback : function, default=None
        Callback function will be called (with current weight vector) after every iteration.
    """

    def __init__(self, input_dim, max_steps=100, epsilon=0.01, tol=-1*float('inf'),
                 callback=None):
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.w = np.zeros(input_dim)
        self.tol = tol
        self.callback = callback

    def fit(self, x, y):
        # Init residuals
        r = y.copy()

        for i in range(self.max_steps):
            # Find feature with maximum correlation with residuals
            c_max = float('inf')
            feature_idx = -1
            for j in range(x.shape[1]):
                c, _ = pearsonr(x[:, j], r)

                if np.abs(c) > np.abs(c_max) or c_max == float('inf'):
                    c_max = c
                    feature_idx = j

            # Update
            delta = self.epsilon * np.sign(np.dot(x[:, feature_idx], r))
            self.w[feature_idx] = self.w[feature_idx] + delta
            r = r - delta * x[:, feature_idx]

            # Call callback
            if self.callback is not None:
                self.callback(self.w.copy())

            # Stopping criterion fulfilled?
            if c_max < self.tol:
                break

        return self

    def predict(self, x):
        return np.dot(x, self.w)
