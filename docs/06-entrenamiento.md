---
title: 06 Entrenamiento
---

# 06 Entrenamiento

Anterior: [05 Gradient check](05-gradient-check.md). Siguiente: [07 Métricas con desbalance](07-metricas-con-desbalance.md). Índice: [RedNeuro: cómo se hizo](index.md).

## El loop

Cada iteración: forward, medir pérdida, backward, actualizar. Se usa todo el train en cada paso (full batch), igual que el original. Con 14,320 ejemplos y 161 parámetros cabe de sobra y cada iteración es una multiplicación de matrices barata.

`src/redneuro/train.py`:

```python
def gradient_descent(X, Y, n_h=16, alpha=0.1, iterations=1000, log_every=100, seed=0,
                     on_iteration=None):
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
```

Misma estructura que el `gradient_descent` del original. Diferencias: se calcula la pérdida (el original no), la predicción es `A2 > 0.5` en vez de `argmax`, se devuelve el historial, y el `on_iteration` es un callback opcional que se agregó después para la [08 Visualización en vivo](08-visualizacion-en-vivo.md). Sin él, se comporta igual.

Se loguea accuracy a propósito, no porque sirva sino para ver la trampa con los propios ojos.

## Primera corrida, alpha = 0.1

```
iter     0  loss 1.1158  accuracy 0.2682
iter   100  loss 0.0977  accuracy 0.9704
iter   200  loss 0.0843  accuracy 0.9745
iter   500  loss 0.0781  accuracy 0.9770
iter   999  loss 0.0751  accuracy 0.9784
```

En 100 iteraciones pasa de 1.12 a 0.098, ya muy por debajo del 0.306 del prior ([Dos números de referencia](04-la-red.md#dos-números-de-referencia)). Está usando las features de verdad, no solo "los púlsares son raros". Después sigue bajando lento.

Primera línea: accuracy 0.27 con pérdida 1.12, la red diciendo púlsar al 72% de todo. Última: 0.978. Suena espectacular. Pero 0.908 lo consigue un modelo que no mira nada, y la pregunta que accuracy no responde es cuántos de los 1,311 púlsares de train está encontrando. Eso está en [07 Métricas con desbalance](07-metricas-con-desbalance.md).

## Experimentos con el learning rate

Mismo código, tres valores:

| alpha | iter 250 | iter 999 | qué pasó |
|---|---|---|---|
| 0.001 | 0.685 | 0.322 | se arrastra; después de 1000 iteraciones sigue **por encima** del prior |
| 0.1 | ~0.083 | 0.075 | bien |
| 3.0 | 0.066 | 0.063 | mejor que 0.1, y **no divergió** |

**0.001** es la misma superficie y la misma dirección, solo que cada paso es cien veces más corto. Llegaría al mismo lugar con unas 100,000 iteraciones.

**3.0** contradijo lo que esperaba. Un learning rate grande debería pasarse del mínimo, subir la pérdida, pasarse hacia el otro lado y explotar. No pasó. La razón está en [03 Preprocesamiento](03-preprocesamiento.md): con las features normalizadas la superficie es un tazón bastante redondo, y en un tazón redondo puedes dar pasos grandes sin rebotar. Con los datos crudos (rangos de 1 a 1191), 3.0 explotaría en las primeras iteraciones. Es la mejor demostración de para qué sirvió normalizar. El punto de quiebre está más arriba; pendiente probar 30.

## Guardar el modelo

`scripts/train.py` es el punto de entrada. Además de entrenar y evaluar, guarda `data/processed/loss_history.npy` y los pesos en `data/processed/params.npz` con `np.savez`, para que [08 Visualización en vivo](08-visualizacion-en-vivo.md) pueda dibujar sin reentrenar. Las semillas están fijas, así que cualquiera que clone el repo obtiene exactamente estos números. La secuencia completa está en [Comandos](comandos.md).

Un tropiezo tonto en esta fase: el archivo `train.py` del paquete se guardó como `trein.py`. Ver [Fase 5](errores-y-lecciones.md#fase-5).
