import numpy as np
from redneuro.losses import relu, relu_derivative, sigmoid


def init_params(n_x=8, n_h=16, n_y=1, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_h, n_x)) * np.sqrt(2 / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = rng.standard_normal((n_y, n_h)) * np.sqrt(2 / n_h)
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2


def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (W2.T @ dZ2) * relu_derivative(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
