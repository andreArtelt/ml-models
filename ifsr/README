# IFSR
Implementation of * **I**ncremental **S**tagewise **F**orward **R**egression
(**IFSR**)* in python.

# About
This directory contains code for the *incremental stagewise forward regression algorithm*.

Correlation is measured by the [pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient).

# Usage
```python
from ifsr import IFSR

# Load data
data = ...
x = data['x_train']
y = data['y_train']

# Make sure data is standardized!
#from sklearn.preprocessing import StandardScaler
#x = StandardScaler().fit_transform(x)

# Fit model
x_axis = []
y_axis = []
# Callback is called after each iteration with the current weight vector
callback = lambda w: y_axis.append(w)  # Save current weight vector

model = IFSR(x.shape[1], max_steps=150, epsilon=0.01, callback=callback)
model.fit(x, y)
y_pred = model.predict(x)

# Print weight vector and plot feature coefficients over time
x_axis = np.array(range(len(y_axis)))
y_axis = np.array(y_axis)

print model.w

import matplotlib.pyplot as plt
labels = []
for k in range(y_axis.shape[1]):
    l,  = plt.plot(x_axis, y_axis[:, k], label='Feature ' + str(k))
    labels.append(l)
plt.legend(handles=labels)
plt.show()
```

# Requirements
- python 2.7
- [scipy](https://github.com/scipy/scipy)
- [numpy](https://github.com/numpy/numpy)
