---
tags: [redneuro, forward, backward, matematica]
---
# 04 La red

Anterior: [[03 Preprocesamiento]]. Siguiente: [[05 Gradient check]]. Índice: [[index|Red neuronal from scratch]].

Esta es la nota larga. Aquí está la matemática de cada función y, en cada paso, qué es igual al notebook de MNIST y qué cambia por ser clasificación binaria.

## Arquitectura

```
X (8, m) → [W1, b1] → Z1 (n_h, m) → ReLU → A1 (n_h, m)
         → [W2, b2] → Z2 (1, m)   → sigmoid → A2 (1, m) = P(púlsar | x)
```

El original es `784 → 10 → 10`. Este es `8 → 16 → 1`. Una sola salida porque con dos clases basta un número: la probabilidad de púlsar. Ese cambio en la capa final es lo que aparece en forward, en la pérdida y en backward. Todo lo demás es idéntico.

`n_h = 16` es un hiperparámetro. Con 8 features no hace falta más.

## init_params

| Parámetro | Shape | Por qué |
|---|---|---|
| W1 | (n_h, 8) | `W1 · X` con X de (8, m) da (n_h, m) |
| b1 | (n_h, 1) | se suma a cada columna por broadcasting |
| W2 | (1, n_h) | `W2 · A1` con A1 de (n_h, m) da (1, m) |
| b2 | (1, 1) | |

**Por qué no ceros.** Si todas las filas de `W1` son iguales, las 16 neuronas ocultas calculan lo mismo, reciben el mismo gradiente y se actualizan igual. Siguen idénticas para siempre: una red de 16 neuronas que se comporta como una de 1. Los valores aleatorios "rompen la simetría".

**Por qué no cualquier aleatorio.** El original usa `rand - 0.5`. Funciona ahí, pero hay una forma con fundamento. Con `X` normalizada (varianza 1), `z = Σ w_j x_j` sobre `n_x` entradas tiene varianza `n_x · Var(w)`. Para que `z` también tenga varianza ~1, hace falta `Var(w) ≈ 1/n_x`. ReLU mata la mitad de las activaciones, así que se compensa con el doble: `Var(w) = 2/n_x`. Eso es la **inicialización de He**: normal estándar por `sqrt(2/n_x)`.

```python
def init_params(n_x=8, n_h=16, n_y=1, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_h, n_x)) * np.sqrt(2 / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = rng.standard_normal((n_y, n_h)) * np.sqrt(2 / n_h)
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2
```

Verificado: `W1 (16, 8)` con std 0.476 (`sqrt(2/8) = 0.5`), `W2 (1, 16)` con std 0.351 (`sqrt(2/16) = 0.354`). Los biases pueden ir en cero; la simetría ya la rompen los pesos.

## forward_prop

```
Z1 = W1 · X + b1        (n_h, m)
A1 = ReLU(Z1)           ReLU(z) = max(0, z)
Z2 = W2 · A1 + b2       (1, m)
A2 = sigmoid(Z2)        sigmoid(z) = 1 / (1 + e^(−z))
```

ReLU es idéntica al original. Sin ella las dos capas lineales colapsan en una (`W2·(W1·X)` es otra matriz por `X`) y la red solo podría aprender rectas.

Sigmoid es el cambio. Aplasta cualquier real a (0, 1): `z = 0` da 0.5, `z = 3` da 0.95, `z = −4` da 0.018. `A2` se interpreta como probabilidad de púlsar.

### Por qué sigmoid y no softmax con dos salidas

Softmax con dos entradas:

$$
p_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}} = \text{sigmoid}(z_1 - z_2)
$$
Softmax de dos clases **es** sigmoid de la diferencia. Las dos neuronas solo importan a través de su resta, así que una sobra. Con una sola salida hay la mitad de parámetros en `W2` y `b2`, la probabilidad de la otra clase es `1 − A2`, y no se pierde nada.

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
```

Misma estructura y mismos cuatro retornos que el original. `@` es lo mismo que `.dot`. La única línea distinta es la última. Se devuelven `Z1` y `A1` porque backward los necesita.

Con la red sin entrenar: `A2` con media 0.61, 49% de `A1` en cero (ReLU apagando la mitad, el "mitad muere" que justificó el factor 2 en He), y 72% de candidatos clasificados como púlsar cuando el real es 9%.

## Pérdida: binary cross-entropy

### De dónde sale

Si el modelo dice que la probabilidad de púlsar es `a`, para un ejemplo con etiqueta `y` la probabilidad que asigna a lo que realmente pasó es:

$$
P(y \mid a) = a^y (1-a)^{1-y}
$$
Si `y = 1` queda `a`; si `y = 0` queda `1 − a`. Un buen modelo asigna alta probabilidad a lo que sí pasó. Logaritmo (productos en sumas, no cambia el máximo) y signo cambiado (minimizar en vez de maximizar):

$$
L = -\frac{1}{m} \sum_i \left[ y_i \log a_i + (1 - y_i) \log(1 - a_i) \right]
$$
### Qué hace en números

Con `y = 1`:

| a | pérdida = −log(a) |
|---|---|
| 0.99 | 0.01 |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

Acertar con confianza cuesta casi nada. **Equivocarse con confianza cuesta muchísimo.** Un ejemplo con `y = 0` y `a = 0.99` cuesta 4.6, cuarenta y cuatro veces más que uno bien clasificado con 0.9. Esa asimetría es lo que empuja a la red.

### Dos números de referencia

- Predecir 0.5 para todo: `−log(0.5) = 0.693`. El "no sé nada".
- Predecir el 9.16% para todo: `−(0.0916·log 0.0916 + 0.9084·log 0.9084) ≈ 0.306`.

El segundo importa. Un modelo baja de 0.693 a 0.306 **solo aprendiendo que los púlsares son raros**, sin mirar las features. Si la pérdida se estanca cerca de 0.3, la red aprendió el prior y nada más. En [[06 Entrenamiento]] esa línea aparece como referencia gris.

### Relación con el cross-entropy del original

La cross-entropy general es `−Σ_k y_k log(a_k)` con `y` one-hot. Con dos clases, `y_2 = 1 − y_1` y `a_2 = 1 − a_1`, y queda exactamente BCE. **No es otra pérdida, es la misma escrita para una sola salida.** Por eso, como se ve abajo, `dZ2 = A2 − Y` sigue siendo idéntico.

Detalle: el notebook de Kaggle **nunca calcula la pérdida**, solo imprime accuracy. Aquí sí, porque quiero ver bajar el número que de verdad se optimiza, y porque accuracy no sirve con este desbalance ([[07 Métricas con desbalance]]).

```python
def binary_cross_entropy(A2, Y, eps=1e-12):
    m = Y.shape[1]
    A2 = np.clip(A2, eps, 1 - eps)
    return -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
```

El `clip` es porque `log(0)` es `−inf`. Con la red sin entrenar `A2` ya tocó `1.0` por redondeo de float. Solo afecta al cálculo de la pérdida, la `A2` real no se toca.

Red sin entrenar: 1.1158. Peor que tirar una moneda, porque grita "púlsar" con confianza a miles de no púlsares.

## backward_prop

Cuatro gradientes, de atrás hacia adelante con la regla de la cadena.

### Paso 1: dZ2, el único que cambia y sin embargo da lo mismo

Para un ejemplo, con `a = sigmoid(z)`:

$$
\frac{\partial L}{\partial a} = -\left[\frac{y}{a} - \frac{1-y}{1-a}\right] \qquad \frac{\partial a}{\partial z} = a(1-a)
$$
La derivada de sigmoid vale verla una vez: si `a = 1/(1+e^(−z))`, entonces `da/dz = e^(−z)/(1+e^(−z))² = a(1−a)`. Multiplicando:

$$
\frac{\partial L}{\partial z} = -\left[\frac{y}{a} - \frac{1-y}{1-a}\right] a(1-a) = -[y(1-a) - (1-y)a] = a - y
$$
Todo se cancela. Vectorizado: `dZ2 = A2 − Y`.

En el original es `dZ2 = A2 − one_hot_Y`, y la derivación con softmax + cross-entropy llega al mismo `a − y`. No es casualidad: BCE es CE de dos clases. Lo único que desaparece es el `one_hot`, porque `Y` ya tiene shape `(1, m)` con ceros y unos.

Para sentirlo: `y = 1`, `a = 0.2` da `dZ2 = −0.8` (hay que subir `z`). `y = 0`, `a = 0.99` da `dZ2 = +0.99` (hay que bajar `z`, mucho).

### Paso 2: dW2 y db2

`Z2 = W2 · A1 + b2`. Derivando respecto a `W2`, la entrada `A1` aparece como factor:

$$
dW_2 = \frac{1}{m} dZ_2 A_1^T \quad (1, m)(m, n_h) \to (1, n_h) \qquad db_2 = \frac{1}{m} \sum_{\text{cols}} dZ_2
$$
El `1/m` viene de que la pérdida es un promedio. Idéntico al original.

### Paso 3: dZ1, propagar el error hacia atrás

$$
dA_1 = W_2^T dZ_2 \quad (n_h, 1)(1, m) \to (n_h, m) \qquad dZ_1 = dA_1 * \text{ReLU}'(Z_1)
$$
`ReLU'(z)` es 1 donde `z > 0` y 0 donde no. Una neurona que estaba apagada en forward **recibe gradiente cero** para ese ejemplo: no contribuyó a la salida, no tiene nada que corregir. El error pasa de largo y solo llega a las que estaban activas. Idéntico al original.

### Paso 4: dW1 y db1

$$
dW_1 = \frac{1}{m} dZ_1 X^T \quad (n_h, m)(m, 8) \to (n_h, 8) \qquad db_1 = \frac{1}{m} \sum_{\text{cols}} dZ_1
$$
Ahí está la `X^T` de [[03 Preprocesamiento#Por qué los rangos dispares dañan el gradiente]]. Idéntico al original.

### Resumen

| Línea | Original (multiclase) | Aquí (binario) |
|---|---|---|
| dZ2 | `A2 − one_hot(Y)` | `A2 − Y` |
| dW2, db2, dZ1, dW1, db1 | igual | igual |

Nunca se escribe la derivada de sigmoid como función aparte: se canceló dentro de `dZ2 = A2 − Y`. La de ReLU sí hace falta.

```python
def relu_derivative(Z):
    return (Z > 0).astype(float)


def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (W2.T @ dZ2) * relu_derivative(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2
```

Dos diferencias de forma con el original. Se quitó `W1` de los argumentos porque el original lo recibe y no lo usa. Y `keepdims=True` en los biases: el original hace `np.sum(dZ2)` sin eje, que devuelve un escalar en vez de `(10, 1)`. Ahí les funciona por broadcasting pero es un bug latente. Con `keepdims` los shapes de `db1`, `db2` coinciden con `b1`, `b2`.

Verificado: los cuatro gradientes tienen el shape de su parámetro, y `db2 = 0.5193 = (A2 − Y).mean()`, positivo, porque la red predice de más y el bias de salida tiene que bajar.

> [!warning] Los shapes correctos no demuestran nada
> Si escribes `Y − A2` en vez de `A2 − Y`, todos los shapes coinciden y la red sube la pérdida en vez de bajarla. Por eso el siguiente paso es el [[05 Gradient check]].

## update_params

$$
\theta \leftarrow \theta - \alpha \, d\theta
$$
para los cuatro parámetros. Si el gradiente dice "la pérdida sube si aumento este peso", el signo menos lo baja. Idéntico al original.

```python
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
```

Qué pasa con `alpha` grande o chico está medido en [[06 Entrenamiento#Experimentos con el learning rate]].
