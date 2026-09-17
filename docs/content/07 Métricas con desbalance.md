---
tags: [redneuro, metricas, desbalance]
---

Anterior: [[06 Entrenamiento]]. Siguiente: [[08 Visualización en vivo]]. Índice: [[index|Red neuronal from scratch]].

## La matriz de confusión

Todo sale de cuatro conteos sobre el test, comparando la predicción (`A2 > umbral`) contra la etiqueta real:

| | Predijo púlsar | Predijo no púlsar |
|---|---|---|
| **Es púlsar** | TP | FN |
| **No es púlsar** | FP | TN |

Accuracy es `(TP + TN) / total`. Con 3,251 negativos y 327 positivos en test, el modelo "siempre 0" tiene `TN = 3251`, `TP = 0`, y accuracy 0.909. Los 327 púlsares perdidos son FN y accuracy apenas los nota, porque están ahogados entre los TN.

## Las tres métricas de la clase positiva

$$
\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN} \qquad F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$
Precision: de los que llamé púlsar, ¿cuántos lo eran? Recall: de los púlsares reales, ¿cuántos encontré? Tiran en direcciones opuestas: si bajas el umbral, subes recall y bajas precision. El modelo "siempre 0" tiene recall 0 y ahí se delata.

F1 es la media **armónica**, no la aritmética, a propósito. Castiga el desbalance entre las dos: precision 1.0 y recall 0.1 dan F1 de 0.18, no de 0.55. Para que F1 sea alto, las dos tienen que serlo.

`src/redneuro/metrics.py`, todo en NumPy:

```python
def predict(A2, threshold=0.5):
    return (A2 > threshold).astype(int)


def confusion_matrix(Y, Y_pred):
    Y = Y.astype(bool)
    Y_pred = Y_pred.astype(bool)
    tp = int(np.sum(Y & Y_pred))
    fp = int(np.sum(~Y & Y_pred))
    fn = int(np.sum(Y & ~Y_pred))
    tn = int(np.sum(~Y & ~Y_pred))
    return tp, fp, fn, tn


def precision_recall_f1(Y, Y_pred):
    tp, fp, fn, tn = confusion_matrix(Y, Y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
```

Los `if ... > 0` evitan dividir por cero con el modelo "siempre 0", que tiene `TP + FP = 0`. Se cotejó contra `sklearn.metrics` y coincide al cuarto decimal, que es la única razón por la que scikit-learn está en el flake.

## Resultado en test, umbral 0.5

```
TP=256 FP=19 FN=71 TN=3232
accuracy : 0.9748
nuestro  : precision=0.9309 recall=0.7829 f1=0.8505
sklearn  : precision=0.9309 recall=0.7829 f1=0.8505
siempre 0: accuracy=0.9086 precision=0.0000 recall=0.0000 f1=0.0000
```

Accuracy dice que la red es apenas 7 puntos mejor que no hacer nada. Recall dice que encuentra 256 de 327 púlsares contra cero. Pero `FN = 71`: uno de cada cuatro púlsares reales se pierde. Con precision 0.93 la red es muy conservadora, casi nunca se equivoca cuando dice "púlsar", a costa de callarse demasiado.

## Qué error es más caro

Un falso positivo cuesta unas horas de telescopio verificando un candidato que no era. Un falso negativo es un púlsar real que nadie va a volver a mirar, porque el modelo es justamente el filtro que decide qué se mira. El FN es más caro. Conviene **subir recall**, aceptando algo menos de precision.

## El umbral no tiene que ser 0.5

`A2` es una probabilidad y el 0.5 era arbitrario. Bajarlo a 0.3 hace que un candidato con `A2 = 0.4` pase a ser púlsar: más TP, más FP. Es una perilla gratis, no requiere reentrenar.

> [!warning] Elegir el umbral mirando el test es hacer trampa
> Sería ajustar una decisión con los datos que se supone que no viste. Se elige en train (o en un split de validación) y se reporta en test con el umbral ya fijado.

```python
def threshold_sweep(A2, Y, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        p, r, f = precision_recall_f1(Y, predict(A2, t))
        rows.append((t, p, r, f))
    return np.array(rows)
```

Barrido en train (recortado):

| umbral | precision | recall | F1 |
|---|---|---|---|
| 0.05 | 0.612 | 0.927 | 0.737 |
| 0.10 | 0.769 | 0.906 | 0.832 |
| 0.20 | 0.873 | 0.878 | 0.876 |
| **0.35** | 0.920 | 0.845 | **0.881** |
| 0.50 | 0.940 | 0.816 | 0.874 |
| 0.70 | 0.964 | 0.759 | 0.850 |
| 0.95 | 0.983 | 0.604 | 0.749 |

Máximo F1 en 0.35. Aplicado al test:

| umbral | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.50 | 256 | 19 | 71 | 0.931 | 0.783 | 0.851 |
| 0.35 | 270 | 29 | 57 | 0.903 | 0.826 | 0.863 |

Léelo como una negociación: 14 púlsares más recuperados a cambio de 10 falsos positivos más. Para un astrónomo, 14 descubrimientos por 10 horas de telescopio es un trato fácil. Y si el criterio fuera "recall de al menos 0.9 cueste lo que cueste", la tabla dice que el umbral tendría que bajar a 0.10, pagando con precision de 0.77: uno de cada cuatro candidatos que mires será falso. Esa decisión ya no es matemática, es de quien opera el telescopio.

## Lo que falta

- **Split de validación.** El umbral se elige en train, que está sesgado a favor del modelo. Un tercer split es lo correcto.
- **Pesos de clase en la BCE.** Multiplicar el término de los positivos por un factor es otra forma de subir recall, cambiando lo que la red aprende y no solo el umbral. Toca `binary_cross_entropy` y `dZ2`, y el [[05 Gradient check]] diría si quedó bien.
- **Curva precision-recall completa** con el área bajo la curva como métrica única que no depende del umbral.
