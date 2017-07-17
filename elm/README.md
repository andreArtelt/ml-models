# ELM
*E*xtreme *L*earning *M*achine (**ELM**).

## About
This folder contains a **python** implementation of the *ELM*.

Solving the *least squares problem* can be done by using a **direct solver** (compute the inverse of the design matrix) or an **iterative optimization algorithm** like Adam, SGD, etc. (makes it suiteable for very large datasets).

The ELM can be used as a **regressor**, **classifier** or as an **autoencoder**.

The model implements the **sklearn regressor interface** and builds upon sklearn, numpy and tensorflow.

### About ELM
If you want to know more or "dig deeper", you might want to visit [www.ntu.edu.sg/home/egbhuang/](https://www.ntu.edu.sg/home/egbhuang/) where you can find a lot of information (including current research) about ELM.

## Requirements
- python 2.7
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)
- [tensorflow](https://github.com/tensorflow/tensorflow)

## Usage
**Regressor**
```python
from elm import ELM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Create data
N = 1000
X = np.linspace(0, 3 * np.pi, N)
y = np.sin(X) + 0.25*np.random.randn(N)  # Generate noisy samples

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

x_train = x_train.reshape(x_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Create and fit ELM
model = ELM(n_hidden=200, n_output=1, act='tanh', solver='direct',
            max_iter=10000, tol=0.00001, early_stopping=False,
            batch_size=100, shuffle=True,
            solver_param=dict({'learning_rate': 0.001}), verbose=True,
            loss='mse', verbose_eval=100, l2_penalty=0.01, l1_penalty=0.0,
            rl_patience=100, reduce_learning_rate=lambda x: 0.9*x)
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

**Autoencoder**
```python
from elm import ELM
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data (digit dataset)
data = datasets.load_digits()
digits = data.images

digits = digits.reshape(digits.shape[0], digits.shape[1]*digits.shape[2])
X = digits

# Preprocessing
X /= 255

# Train test split
x_train, x_test, y_train, y_test = train_test_split(X, X, test_size=0.3)

# Build and fit autoencoder (64 dims -> 50 dims)
model = ELM(n_hidden=50, n_output=64, act='sigmoid', solver='direct',
            max_iter=5000, tol=0.00001, early_stopping=False,
            batch_size=100, shuffle=True,
            solver_param=dict({'learning_rate': 0.0001}), verbose=True,
            loss='mse', verbose_eval=100, l2_penalty=0.0001, l1_penalty=0.0)

model.fit(x_train, x_train)

# Transform data
x_train_transformed = model.transform_ae(x_train)
print x_train_transformed.shape
print x_train.shape

# Evaluate model
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print mean_squared_error(x_train, y_train_pred)
print mean_squared_error(x_test, y_test_pred)

x_train = x_train.reshape(x_train.shape[0], 8, 8)
x_test = x_test.reshape(x_test.shape[0], 8, 8)
y_train_pred = y_train_pred.reshape(y_train_pred.shape[0], 8, 8)
y_test_pred = y_test_pred.reshape(y_test_pred.shape[0], 8, 8)

# Plot random image (original & decoded)
def plot_img(x):
    plt.imshow(x, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
i = np.random.randint(0, x_train.shape[0])
plot_img(x_train[i])
plot_img(y_train_pred[i])
```
