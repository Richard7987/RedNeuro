---
tags: [redneuro, visualizacion, matplotlib]
---

Anterior: [[07 Métricas con desbalance]]. Índice: [[index|Red neuronal from scratch]].

Una ventana donde se ve la red entrenando y después procesando candidatos uno por uno. Todo con matplotlib, que ya estaba en el flake.

## Qué codifica cada elemento

La red se dibuja como tres columnas de círculos con líneas entre ellas. Cada cosa visual es un número concreto:

- **Nodos de entrada:** el valor normalizado de cada feature para ese candidato. Como son z-scores van de negativo a positivo, así que color divergente: azul negativo, rojo positivo, blanco en cero.
- **Nodos ocultos:** `A1`, la salida de ReLU. Blanco es apagado. Escala de un solo color hasta azul oscuro.
- **Nodo de salida:** `A2`, la probabilidad, con el número al lado.
- **Líneas:** al principio eran los pesos (grosor por magnitud, color por signo). Después cambió a algo mejor, ver abajo.

## Las líneas se encienden por uso, no por peso

La primera versión dibujaba `|W|` como grosor. Se ve bien pero no cuenta lo que pasa con **este** candidato: un peso grande conectado a una entrada que vale cero no lleva nada.

Lo que se dibuja ahora es la **contribución**: `W[j, i] · a_i`, el peso por la activación que entra. Grosor y opacidad crecen con su magnitud, el color es su signo (azul empuja hacia púlsar, naranja hacia no púlsar). Una feature cercana a cero o una neurona apagada por ReLU no envía nada y sus líneas quedan casi invisibles. En la capa de salida se ve clarísimo qué dos o tres neuronas ocultas están decidiendo.

```python
def draw_edges(ax, p_from, p_to, W, a_from, activo):
    segs = [(p_from[i], p_to[j]) for j in range(W.shape[0]) for i in range(W.shape[1])]
    if activo:
        contrib = (W * a_from.reshape(1, -1)).ravel()
        fuerza = np.abs(contrib) / (np.abs(contrib).max() + 1e-12)
        base = np.where(contrib > 0, POSITIVO, NEGATIVO)
        widths = 0.2 + 4.0 * fuerza
        alphas = 0.05 + 0.95 * fuerza
    else:
        w = W.ravel()
        base = np.where(w > 0, POSITIVO, NEGATIVO)
        widths = 0.3 + 2.0 * np.abs(w) / (np.abs(W).max() + 1e-12)
        alphas = np.full(w.size, 0.08)
    colors = [to_rgba(c, alpha=float(a)) for c, a in zip(base, alphas)]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=1))
```

El orden de `segs` y de `W.ravel()` coincide a propósito: `j` externo, `i` interno, igual que `ravel` recorre una matriz `(n_to, n_from)`. Si se cruzan, cada línea tendría el color de otro peso.

## Las cuatro fases

`draw_network(ax, params, x, y_true, fase, threshold)` dibuja un solo ejemplo `x` de shape `(8, 1)`. El `fase` controla cuánto se ha "revelado":

1. Entra el dato: solo las entradas coloreadas, todas las líneas tenues.
2. Capa oculta: se encienden las líneas entrada→oculta por contribución y las neuronas ocultas.
3. Salida: se encienden las líneas oculta→salida y el nodo de salida con `A2`.
4. Veredicto: "PÚLSAR" o "NO PÚLSAR", la etiqueta real, y correcto en verde o incorrecto en rojo.

Un detalle: `X_test[:, k:k + 1]` y no `X_test[:, k]`. El primero da `(8, 1)`, una columna, y `forward_prop` funciona igual que con `m` ejemplos. El segundo daría `(8,)` y rompería la multiplicación.

## Cómo se engancha con el entrenamiento

`gradient_descent` no debe saber que existe matplotlib. Si metes `plt.pause` dentro del loop, `scripts/train.py` se vuelve lento y dependiente de una ventana. La solución es un **callback**: `gradient_descent` recibe `on_iteration=None` y, si se lo dan, lo llama al final de cada iteración con `(i, params, loss)`. Sin callback, el comportamiento es idéntico al de antes.

`scripts/train_live.py` define el callback dentro de `main`, así tiene acceso a la figura y a la lista de pérdidas sin pasarlas como argumentos (el estado vive en el closure). Cada 5 iteraciones redibuja la red con los pesos actuales, y cada 4 cuadros cambia de ejemplo alternando un púlsar y un no púlsar. A la derecha, la curva de pérdida con la línea gris del prior en 0.306.

`plt.ion()` pone matplotlib en modo interactivo: la ventana abre de inmediato y no bloquea. `plt.pause` es lo que de verdad la refresca. `ax.relim()` y `autoscale_view()` recalculan los límites porque la curva crece en cada cuadro.

## Modo inferencia

Al terminar de entrenar, 40 candidatos del test (mitad púlsares, mitad no, mezclados) pasan uno por uno: cuatro fases a 0.7 segundos cada una, y después del veredicto la pantalla se queda quieta 3 segundos para poder leerlo. El panel derecho cambia a cuatro barras acumuladas: púlsar bien (TP), falsa alarma (FP), púlsar perdido (FN), no púlsar bien (TN). Verde aciertos, rojo errores.

```bash
python scripts/train_live.py               # entrena en vivo y luego inferencia
python scripts/train_live.py --inferencia  # salta el entrenamiento, carga params.npz
```

Perillas al inicio del archivo: `INFER_PAUSE` (tiempo por fase), `INFER_HOLD` (tiempo con el veredicto en pantalla), `INFER_EXAMPLES` (cuántos candidatos), `TRAIN_DRAW_EVERY` y `TRAIN_PAUSE` para la etapa de entrenamiento.

> [!warning] Nunca una pausa en cero
> `plt.pause(0)` con un backend sin ventana significa "espera indefinidamente". Lo descubrí en una prueba automatizada que se quedó colgada seis minutos a 0% de CPU. Detalles en [[Errores y lecciones#Fase 8]].

## Backend

En NixOS el matplotlib del flake trae `TkAgg`, que es el que abre la ventana. Se puede confirmar con:

```bash
python -c "import matplotlib; print(matplotlib.get_backend())"
```

Para probar el script sin pantalla (por ejemplo en un test), `MPLBACKEND=Agg python scripts/train_live.py` corre todo y solo avisa que no puede mostrar la ventana.

## Imagen estática

`scripts/animate.py` carga `params.npz`, dibuja el primer púlsar del test en fase 4 y lo guarda en `data/processed/red_ejemplo.png`. Sirve para una nota o una presentación sin abrir la ventana.
