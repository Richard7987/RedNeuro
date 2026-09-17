import numpy as np

from redneuro.losses import binary_cross_entropy
from redneuro.model import backward_prop, forward_prop, init_params


def loss_for(params, X, Y):
    W1, b1, W2, b2 = params
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return binary_cross_entropy(A2, Y)


def numerical_gradient(params, idx, X, Y, eps=1e-6):
    p = params[idx]
    grad = np.zeros_like(p)
    for i in np.ndindex(p.shape):
        original = p[i]
        p[i] = original + eps
        loss_plus = loss_for(params, X, Y)
        p[i] = original - eps
        loss_minus = loss_for(params, X, Y)
        p[i] = original
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def test_gradient_check():
    rng = np.random.default_rng(42)
    n_x, n_h, m = 8, 5, 20
    X = rng.standard_normal((n_x, m))
    Y = (rng.random((1, m)) < 0.3).astype(float)
    params = list(init_params(n_x=n_x, n_h=n_h, seed=1))
    W1, b1, W2, b2 = params

    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    analytic = backward_prop(Z1, A1, Z2, A2, W2, X, Y)

    for idx, name in enumerate(["W1", "b1", "W2", "b2"]):
        numeric = numerical_gradient(params, idx, X, Y)
        rel_error = np.linalg.norm(numeric - analytic[idx]) / (
            np.linalg.norm(numeric) + np.linalg.norm(analytic[idx])
        )
        print(f"{name}: error relativo {rel_error:.2e}")
        assert rel_error < 1e-7, f"{name}: error relativo {rel_error:.2e}"
