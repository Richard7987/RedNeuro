---
title: 03 Preprocesamiento
---

# 03 Preprocesamiento

Anterior: [02 Datos HTRU2](02-datos-htru2.md). Siguiente: [04 La red](04-la-red.md). Índice: [RedNeuro: cómo se hizo](index.md).

Tres pasos en este orden: **split, normalizar, transponer**. El orden importa.

## Split estratificado, y por qué va primero

El notebook original hace `np.random.shuffle(data)` y corta las primeras 1000 filas como dev. Con MNIST funciona porque las diez clases están balanceadas. Con 9% de positivos, un corte al azar puede dejar el test con pocos púlsares, y entonces precision y recall saltan un montón entre corridas.

Estratificar es hacer el shuffle y el corte **dentro de cada clase** por separado, y luego juntar. Así train y test tienen los dos ~9.2% de positivos.

Y va antes de normalizar porque la media y desviación que se usan para normalizar deben calcularse solo con train. Si las calculas con todo el dataset, el test "filtra" información al entrenamiento. En producción los datos nuevos no estaban cuando calculaste la media; el test tiene que simular eso.

```python
def stratified_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
```

`np.flatnonzero(y == cls)` devuelve los índices donde la condición es verdadera. La `permutation` final vuelve a mezclar las dos clases; sin ella train quedaría con todos los ceros primero y todos los unos al final.

Resultado: train 14,320 filas con 0.0916 de positivos, test 3,578 con 0.0914. Que coincidan es la prueba de que la estratificación funciona.

## Por qué los rangos dispares dañan el gradiente

En la primera capa se calcula `Z1 = W1 · X + b1`. Cuando llegue el backward prop ([Paso 4: dW1 y db1](04-la-red.md#paso-4-dw1-y-db1)), el gradiente de esos pesos es:

$$dW_1 = \frac{1}{m} \, dZ_1 \, X^T$$

La `X` está ahí al final. **El gradiente de cada peso es proporcional al valor de la feature que multiplica.** Si `dm_skewness` vale ~1000 e `ip_kurtosis` vale ~1, el peso de la primera recibe un gradiente mil veces más grande.

Ejemplo con dos features y pesos iniciales de 0.01:

```
x1 = 1000   w1 = 0.01   →  w1·x1 = 10
x2 = 1      w2 = 0.01   →  w2·x2 = 0.01
```

`z = 10.01`. La segunda feature aporta el 0.1%. La red no la ve. Y si el error `dz` vale 0.5:

```
dw1 = 0.5 · 1000 = 500
dw2 = 0.5 · 1    = 0.5
```

Con un solo learning rate no puedes contentar a los dos. Si es 0.01, `w1` salta 5 unidades por paso y diverge. Si es 0.00001 para que `w1` se porte bien, `w2` avanza a paso de hormiga. Geométricamente la superficie de la pérdida es un valle largo y angosto, y el gradient descent rebota entre las paredes. Normalizar convierte el valle en un tazón.

Si los dos valieran alrededor de 1, `dw1 ≈ dw2 ≈ 0.5`, y un solo learning rate les sirve. Eso es todo lo que hace la normalización.

## Z-score

Para cada feature $j$, con $\mu_j$ y $\sigma_j$ calculadas sobre train:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

Restar la media centra en 0, dividir por la desviación deja dispersión 1. Después de esto todas las features viven en "cuántas desviaciones estándar me alejo del promedio".

```python
def normalize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma
```

`axis=0` colapsa las filas, así que da un valor por columna. Al test se le aplican **las mismas** `mu` y `sigma`, sin recalcular. Se devuelven porque si algún día hay que clasificar un candidato nuevo, hay que normalizarlo con las estadísticas de train.

Diferencia con el original: ahí hacen `X / 255`. Es una constante conocida de antemano (los píxeles van de 0 a 255), no hay nada que aprender de train ni riesgo de filtrar test. Con HTRU2 no conozco los rangos a priori y varían por feature.

## Shapes (features, m)

Pandas entrega `(m, 8)`: una fila por ejemplo. La red quiere `(8, m)`: una **columna** por ejemplo. Es la convención del notebook original (`data.T`, y `X_train` termina siendo `(784, m)`), y la razón es la multiplicación de la primera capa:

```
W1 · X + b1
(n_h, 8) · (8, m) → (n_h, m)
```

Sale una columna de activaciones por ejemplo, y `b1` de shape `(n_h, 1)` se suma a cada columna por broadcasting. Si `X` quedara como `(m, 8)` habría que voltear todas las fórmulas. `Y` pasa de `(m,)` a `(1, m)` con `reshape(1, -1)` para que `A2 - Y` no dé sorpresas.

```python
def load_data(test_frac=0.2, seed=0):
    X, y = load_raw()
    Xtr, ytr, Xte, yte = stratified_split(X, y, test_frac, seed)
    Xtr, Xte, mu, sigma = normalize(Xtr, Xte)
    return Xtr.T, ytr.reshape(1, -1), Xte.T, yte.reshape(1, -1)
```

## Verificación

```
X_train (8, 14320) Y_train (1, 14320)
X_test  (8, 3578)  Y_test  (1, 3578)
media train por feature  [-0. -0. -0. -0. -0. -0. -0. -0.]
std   train por feature  [1. 1. 1. 1. 1. 1. 1. 1.]
media test  por feature  [ 0.019  0.028 -0.017 -0.015 -0.013 -0.001  0.012  0.021]
```

> **💡 La media de test NO es cero**
>
> Y está bien que así sea. Si saliera exactamente cero significaría que normalicé test con sus propias estadísticas, que es el error que quería evitar. Que quede en ±0.03 es la señal de que se hizo con las de train.
