---
tags: [redneuro, testing, gradiente]
---
# 05 Gradient check

Anterior: [[04 La red]]. Siguiente: [[06 Entrenamiento]]. Índice: [[index|Red neuronal from scratch]].

Es la prueba que el notebook original no hace, y la razón principal por la que el proyecto está en módulos y no en un notebook.

## La idea

La derivada tiene una definición que no necesita regla de la cadena ni saber nada de la red:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$

Tomas **un solo peso**, lo mueves un poquito hacia arriba, calculas la pérdida completa con forward prop, lo mueves hacia abajo, calculas otra vez, divides. Eso es la pendiente. Se repite para cada peso de cada matriz.

Es carísimo: dos forward props por parámetro. Por eso el test usa una red chica (5 ocultas) y 20 ejemplos **aleatorios**. No importa que sean inventados: las derivadas son correctas o no lo son, sin importar los datos. Y así el test no depende de haber descargado el CSV.

Se usa la diferencia centrada (`+ε` y `−ε`) en vez de solo `+ε` porque su error es proporcional a `ε²`, mucho más preciso.

## Cómo comparar

$$\text{error} = \frac{\|g_{num} - g_{ana}\|}{\|g_{num}\| + \|g_{ana}\|}$$

Se divide por las normas para que no dependa de la escala. En float64 con `ε = 1e-6`, si las fórmulas están bien el error queda cerca de `1e-9`. Si hay un bug, sale del orden de 1. No hay zona gris.

## El test

`tests/test_model.py`:

```python
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
```

`np.ndindex(p.shape)` recorre todos los índices de una matriz, sirve igual para `(5, 8)` que para `(1, 1)`. `p[i] = original` restaura el peso al final de cada vuelta; como `params[idx]` es el mismo objeto que `W1`, modificarlo modifica lo que usa `loss_for`, y eso es intencional.

## Resultado

```
pytest tests/ -v -s

W1: error relativo 6.18e-10
b1: error relativo 7.49e-10
W2: error relativo 3.50e-10
b2: error relativo 1.48e-10
PASSED
```

Cuatro errores del orden de `1e-10`. El backward prop está **demostrado** correcto, no "parece que funciona".

> [!tip] Para ver qué atrapa
> Cambiar `dZ2 = A2 - Y` por `dZ2 = Y - A2` en `model.py`, correr pytest, ver el error saltar a `1.00e+00`, y deshacer. Gradiente exactamente al revés, shapes perfectos, y el test lo pilla.

## Por qué esto importa más de lo que parece

En [[06 Entrenamiento]] la pérdida baja bien a la primera. Sin el gradient check, que baje no dice mucho: una red con un gradiente medio mal (por ejemplo `db1` con el eje equivocado) muchas veces también baja, solo que peor y sin que sepas por qué. Con el test pasando, cualquier problema que aparezca después ya no es de las derivadas, y eso descarta la mitad de los sospechosos.
