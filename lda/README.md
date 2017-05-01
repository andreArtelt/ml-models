# Linear Discriminant Analysis
**Regularized/Fisher Discriminant Analysis (RDA/FDA)** in **python**.

## About
This folder contains a **python** implementation of *Regularized Discriminant Analysis* (**RDA**)
and *Fisher Discriminant Analysis* (**FDA**).

Both models implements the **sklearn classifier & estimator interface**.

## Requirements
- python 2.7
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)

## Usage
```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from rda import RDA     # Import RDA
from fisher import FDA  # Import FDA

# Create some artifical data
X, y = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.5)
X = StandardScaler().fit_transform(X) # Standardize data

# Fit model
model = FDA()  # Fisher Discriminant Analysis (FDA)
#model = RDA(alpha=0.5)  # Regularized Discriminant Analysis (RDA)
model.fit(X, y)

# Evaluate model
print model.score(X, y)

# Plot contours of decision boundaries in scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict_proba(grid)[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.show()
```
