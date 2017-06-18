from sklearn.base import BaseEstimator
import numpy as np


class NG(BaseEstimator):
    def __init__(self, n_neurons=100, learning_rate=0.9, epsilon=0.9,
                 learning_rate_final=0.1, epsilon_final=0.01, norm=2,
                 max_iter=1000):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.learning_rate_final = learning_rate_final
        self.epsilon_final = epsilon_final
        self.norm = norm
        self.max_iter = max_iter
        self.w = None

    def get_params(self, deep=True):
        return dict({
            'n_neurons': self.n_neurons,
            'learning_rate': self.learning_rate,
            'learning_rate_final': self.learning_rate_final,
            'epsilon': self.epsilon,
            'epsilon_final': self.epsilon_final,
            'norm': self.norm,
            'w': self.w
        })

    def set_params(self, **params):
        self.n_neurons = params['n_neurons']
        self.learning_rate = params['learning_rate']
        self.learning_rate_final = params['learning_rate_final']
        self.epsilon = params['epsilon']
        self.epsilon_final = params['epsilon_final']
        self.norm = params['norm']
        self.w = params['w']

    def get_winner(self, x):
        if X is None:
            raise ValueError('Argument must not be None')

        best_w = None
        best_dist = float('Inf')

        for i in range(self.n_neurons):
            d = np.linalg.norm(x - self.w[i], ord=self.norm)
            if d < best_dist:
                best_w = self.w[i]
                best_dist = d

        return best_w, best_dist

    def fit(self, X):
        if X is None:
            raise ValueError('Argument must not be None')

        # Create neurons/prototypes
        dim = X.shape[1]
        self.w = np.array([np.random.rand(dim) for _ in range(self.n_neurons)])

        # Fit
        n_samples = X.shape[0]
        for t in range(self.max_iter):
            # Select random sample
            x = X[np.random.randint(0, n_samples)]

            # Compute distance to each prototype and sort them according to their distance
            neuron_dist = [(i, np.linalg.norm(x - self.w[i], ord=self.norm)) for i in range(self.n_neurons)]
            neuron_dist.sort(key=lambda k: k[1])

            # Adapt neurons
            for i in range(self.n_neurons):
                lr = self.learning_rate * (self.learning_rate_final / self.learning_rate)**(t / self.max_iter)
                ep = self.epsilon * (self.epsilon_final / self.epsilon)**(t / self.max_iter)
                h = np.exp((1.0 - i) / ep)
                self.w[i] += lr * h * (x - self.w[i])

