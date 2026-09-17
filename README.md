# RedNeuro

Red neuronal desde cero con NumPy para clasificar candidatos a púlsar (dataset HTRU2).
Sin TensorFlow ni PyTorch: init, forward, backward y gradient descent escritos a mano.

## Correr

```bash
nix develop
python scripts/download_data.py
python scripts/train.py
```

Para verla trabajar candidato por candidato:

```bash
python scripts/train_live.py
```

## Resultado

Red 8→16→1, 1000 iteraciones, test de 3,578 candidatos.

| umbral | precision | recall | F1 |
|---|---|---|---|
| 0.50 | 0.931 | 0.783 | 0.851 |
| 0.35 | 0.903 | 0.826 | 0.863 |

Accuracy 0.976, pero un modelo que dice "no púlsar" a todo saca 0.909. Por eso las métricas de arriba.

## Cómo se hizo

Paso a paso, con la matemática y los errores del camino:
[richard7987.github.io/RedNeuro](https://richard7987.github.io/RedNeuro/) (fuente en [docs/](docs/index.md)).
