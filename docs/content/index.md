---
title: Red neuronal from scratch
tags: [redneuro, numpy, pulsares, indice]
---
## Idea
Crear una red neuronal desde 0 usando como base el [blog](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) y en lugar de los datos que propone el, usar los datos de **HTRU Pulsar candidates**, de los cuales la red identificara si es un pulsar o no.

El repo vive en `~/Projects/Redneuro`. Esta nota es el índice; cada fase tiene su propia nota con el porqué de cada decisión, los números que salieron y lo que se rompió en el camino.

## Guía paso a paso

1. [[01 Entorno con Nix]] — flake.nix, `nix develop`, y el detalle de git que hace que el flake no vea archivos nuevos.
2. [[02 Datos HTRU2]] — descarga, exploración, y el archivo con fines de línea raros.
3. [[03 Preprocesamiento]] — split estratificado, z-score con estadísticas de train, y por qué X tiene shape (8, m).
4. [[04 La red]] — init, forward, pérdida y backward. Qué es igual al notebook original y qué cambia por ser binario.
5. [[05 Gradient check]] — la prueba que demuestra que las derivadas están bien.
6. [[06 Entrenamiento]] — el loop, la pérdida bajando, y los experimentos con el learning rate.
7. [[07 Métricas con desbalance]] — por qué accuracy engaña, precision, recall, F1 y el umbral.
8. [[08 Visualización en vivo]] — ver la red trabajando candidato por candidato.

Aparte:
- [[Errores y lecciones]] — todo lo que falló y qué significaba cada error.
- [[Glosario RedNeuro]] — los términos, explicados corto.
- [[Comandos RedNeuro]] — chuleta para correr todo.

## Resultado final

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Red, umbral 0.5 | 0.975 | 0.931 | 0.783 | 0.851 |
| Red, umbral 0.35 | 0.976 | 0.903 | 0.826 | 0.863 |
| Siempre "no púlsar" | 0.909 | 0 | 0 | 0 |

La pérdida (binary cross-entropy) baja de 1.12 a 0.075 en 1000 iteraciones. Con umbral 0.35 la red encuentra 270 de los 327 púlsares del test. Detalles en [[07 Métricas con desbalance]].

## Organización  
``` 
Redneuro/
├── flake.nix             # nix develop → python + numpy, pandas, matplotlib, sklearn, jupyter
├── flake.lock            # lo genera nix, se commitea
├── .gitignore            # data/, __pycache__, .ipynb_checkpoints, .pytest_cache, result
├── README.md
├── data/
│   ├── raw/              # HTRU_2.csv tal cual se bajó, nunca se toca
│   └── processed/        # params.npz, loss_history.npy, imágenes generadas
├── notebooks/
│   └── 01_explore.ipynb  # solo mirar los datos, cero lógica del modelo
├── src/redneuro/
│   ├── __init__.py
│   ├── data.py           # cargar, normalizar, split, transponer a (features, m)
│   ├── model.py          # init_params, forward_prop, backward_prop, update_params
│   ├── losses.py         # sigmoid, ReLU, su derivada, binary cross-entropy
│   ├── metrics.py        # precision, recall, F1, matriz de confusión, barrido de umbral
│   ├── train.py          # loop de gradient descent (con callback opcional)
│   └── viz.py            # dibujo de la red con activaciones de un ejemplo
├── scripts/
│   ├── download_data.py  # baja HTRU2 de UCI a data/raw
│   ├── train.py          # punto de entrada: entrena, evalúa, guarda pesos
│   ├── animate.py        # imagen estática de la red con un ejemplo
│   └── train_live.py     # ventana en vivo: entrenamiento y luego inferencia uno por uno
└── tests/
    └── test_model.py     # gradient check
```

### Qué es el dataset

HTRU2 viene del survey *High Time Resolution Universe* (Lyon et al., 2016). Cada fila es un **candidato** a púlsar detectado por el radiotelescopio. La mayoría son interferencia de radio o ruido. Los astrónomos etiquetaron a mano cuáles son púlsares reales.

Las 8 features son estadísticas de dos curvas que describen cada candidato:

- **Perfil integrado** (la forma del pulso promediado): media, desviación estándar, curtosis en exceso y asimetría (skewness).
- **Curva DM-SNR** (cómo cambia la señal con la dispersión): las mismas cuatro estadísticas.

La columna 9 es la clase: 1 púlsar, 0 no púlsar. El CSV **no tiene encabezado**, así que los nombres los ponemos nosotros.

| Dato | Valor |
|---|---|
| Filas | 17,898 |
| Púlsares reales | 1,639 |
| Proporción positiva | ~9.2% |

Ese 9.2% es lo que vamos a tener en la cabeza durante todo el proyecto. Un modelo que diga "no púlsar" a todo acierta el 90.8% de las veces sin aprender nada.

### Por qué un script de descarga y no bajarlo a mano

Tres razones. El archivo queda **en `.gitignore`**, así que el repo no pesa y cualquiera lo regenera. El script documenta **de dónde salió** el dato, con la URL exacta. Y si UCI cambia algo, el script falla ruidosamente en vez de que tú trabajes con un CSV distinto sin saberlo.

Usamos solo la librería estándar (`urllib` y `zipfile`), nada que instalar.