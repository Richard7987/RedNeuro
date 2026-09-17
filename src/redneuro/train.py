import numpy as np

from redneuro.losses import binary_cross_entropy
from redneuro.model import backward_prop, forward_prop, init_params, update_params


def gradient_descent(X, Y, n_h=16, alpha=0.1, iterations=1000, log_every=100, seed=0,
                     on_iteration=None):
    """Entrena con gradient descent full-batch.

    on_iteration: callback opcional que se llama al final de cada iteración con
    (i, (W1, b1, W2, b2), loss). Sirve para visualizar sin acoplar este módulo a matplotlib.
    """
    W1, b1, W2, b2 = init_params(n_x=X.shape[0], n_h=n_h, seed=seed)
    history = []
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        loss = binary_cross_entropy(A2, Y)
        history.append(loss)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % log_every == 0 or i == iterations - 1:
            accuracy = ((A2 > 0.5) == Y).mean()
            print(f"iter {i:5d}  loss {loss:.4f}  accuracy {accuracy:.4f}")
        if on_iteration is not None:
            on_iteration(i, (W1, b1, W2, b2), loss)
    return (W1, b1, W2, b2), np.array(history)
