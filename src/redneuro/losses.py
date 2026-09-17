import numpy as np


def relu(Z):
    return np.maximum(Z, 0)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def binary_cross_entropy(A2, Y, eps=1e-12):
    m = Y.shape[1]
    A2 = np.clip(A2, eps, 1 - eps)
    return -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m


def relu_derivative(Z):
    return (Z > 0).astype(float)
