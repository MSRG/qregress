from classical import classical_regressor
from classical import run_models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate data set
dataset_size = 50
x_values = np.linspace(-10, 10, dataset_size)
y_values = np.sin(x_values)

# Combine x and y values into a dataset
dataset = np.column_stack((x_values, y_values))

# Shuffle the dataset randomly
np.random.shuffle(dataset)


X_tr, X_te, y_tr, y_te = train_test_split(x_values, y_values, test_size=0.2)


"""
plt.scatter(X_tr, y_tr, color='b', label='train')
plt.scatter(X_te, y_te, color='r', label='test')
plt.legend()
plt.show()
"""

X_tr = X_tr.reshape(-1, 1)
X_te = X_te.reshape(-1, 1)


run_models(X_tr, y_tr, X_te, y_te)
