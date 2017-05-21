# Kernel regression
Kernel regression in python.

## About
This folder contains a **python** implementation of *kernel regression* (including a predefined gaussian kernel).

The model implements the **sklearn regressor and estimator interfaces**.

## Requirements
- python 2.7
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)

## Usage
```python
from KernelRegression import KernelRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.spatial.distance

# Create data
N = 1000
X = np.linspace(0, 3 * np.pi, N)
y = np.sin(X) + 0.25 * np.random.randn(N)  # Generate noisy samples

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

x_train = x_train.reshape(x_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Estimate sigma by computing all pairwise distances
pairwise_dist = scipy.spatial.distance.pdist(x_train)
print np.min(pairwise_dist)
print np.max(pairwise_dist)
print np.mean(pairwise_dist)
# Estimate sigma=0.5 * sqrt(Mean_dist / 2)

# Fit model
model = KernelRegression(kernel='gauss', sigma=0.35)  # NOTE: kernel= could be set to any custom function implementing a valid kernel
model.fit(x_train, y_train)

# Evaluate
y_pred_train = model.predict(x_train)
print mean_squared_error(y_train, y_pred_train)

y_pred = model.predict(x_test)
print mean_squared_error(y_test, y_pred)

def plot(x, y, y_pred):
    plt.plot(x, y, 'o')
    plt.plot(x, y_pred, 'or')
    plt.show()
plot(x_train, y_train, y_pred_train)
plot(x_test, y_test, y_pred)
```
