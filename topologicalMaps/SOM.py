from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import operator


class SOM(BaseEstimator, TransformerMixin):
    def __init__(self, learning_rate=0.9, sigma=1.0, learning_rate_final=0.1,
                 sigma_final=0.01, learning_rate_decay=0.9, sigma_decay=0.9,
                 norm=2, grid_shape=(10, 10), max_iter=1000):
        self.learning_rate = learning_rate
        self.learning_rate_final = learning_rate_final
        self.learning_rate_decay = learning_rate_decay
        self.sigma = sigma
        self.sigma_final = sigma_final
        self.sigma_decay = sigma_decay
        self.norm = norm
        self.grid_shape = grid_shape
        self.max_iter = max_iter
        self.w = None
        self.indices = []

    def get_params(self, deep=True):
        return dict({
            'learning_rate': self.learning_rate,
            'learning_rate_final': self.learning_rate_final,
            'learning_rate_decay': self.learning_rate_decay,
            'sigma': self.sigma,
            'sigma_final': self.sigma_final,
            'sigma_decay': self.sigma_decay,
            'norm': self.norm,
            'grid_shape': self.grid_shape,
            'max_iter': self.max_iter,
            'w': self.w,
            'indices': self.indices
        })

    def set_params(self, **params):
        self.learning_rate = params['learning_rate']
        self.learning_rate_final = params['learning_rate_final']
        self.learning_rate_decay = params['learning_rate_decay']
        self.sigma = params['sigma']
        self.sigma_final = params['sigma_final']
        self.sigma_decay = params['sigma_decay']
        self.norm = params['norm']
        self.grid_shape = params['grid_shape']
        self.max_iter = params['max_iter']
        self.w = params['w']
        self.indices = params['indices']

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if X is None:
            raise ValueError('Argument must not be None')

        r = []
        for i in range(X.shape[0]):
            _, _, index = self.get_winner(X[i,:])
            r.append(index)

        return np.array(r)

    def get_winner(self, x):
        winner = (None, -1, None)

        for index in self.indices:
            # Compute distance to current neuron
            d = np.linalg.norm(self.w[index] - x, ord=self.norm)

            # Check if it is closer than the previous ones
            if winner[0] is None:
                winner = (self.w[index], d, index)
            else:
                if d < winner[1]:
                    winner = (self.w[index], d, index)

        return winner

    def get_top_k_winner(self, x, k=1):
        n = []
        for index in self.indices:
            d = np.linalg.norm(self.w[index] - x, ord=self.norm)
            n.append((self.w[index], d, index))

        n.sort(key=lambda o: o[1])

        return n[:k]

    def fit(self, X):
        if X is None:
            raise ValueError('Argument must not be None')

        # Create grid of neurons/prototypes
        dim = X.shape[1]
        n_neurons = reduce(operator.mul, self.grid_shape, 1)

        self.w = np.array([np.random.rand(dim) for _ in range(n_neurons)]).reshape(self.grid_shape + (dim,))

        # Precompute list with all indices
        self.indices = []

        def get_all_indices(a, xs, r):  # Helper function for computing a list of all indices
            if len(xs) != 0:
                for i in xs[0]:
                    get_all_indices(a + [i], xs[1:], r)
            else:
                r.append(tuple(a))

        get_all_indices([], map(lambda x: range(x), self.grid_shape), self.indices)

        # Fit model
        n_samples = X.shape[0]
        s = self.sigma
        lr = self.learning_rate

        def dist(a, b): # Helper function for computing the distance between two indices
            return np.linalg.norm(np.array(a) - np.array(b))**2

        for _ in range(self.max_iter):
            # Select random sample and compute winner neuron
            i = np.random.randint(0, n_samples)
            x = X[i]
            _, _, w_index = self.get_winner(x)

            # Adapt neurons
            for index in self.indices:
                d = dist(index, w_index)
                if d <= s:
                    l = lr * np.exp(-d / (2.0 * s**2))
                    self.w[index] += l * (x - self.w[index])

            # Decay params
            if s > self.sigma_final:
                s *= self.sigma_decay
            if lr > self.learning_rate_final:
                lr *= self.learning_rate_decay

