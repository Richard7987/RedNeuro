---
tags: [redneuro, glosario]
---

Índice: [[index|Red neuronal from scratch]]. Cada término apunta a la nota donde se usa.

**Feature.** Una columna de entrada. HTRU2 tiene 8: cuatro estadísticas del perfil integrado (`ip_`) y cuatro de la curva DM-SNR (`dm_`). [[02 Datos HTRU2]].

**m.** Número de ejemplos. 14,320 en train, 3,578 en test. En las fórmulas aparece como el `1/m` de los promedios.

**Shape (features, m).** Convención donde cada columna es un ejemplo. `X` es `(8, m)`, `Y` es `(1, m)`. Viene de que `W1 · X` necesita las features como filas. [[03 Preprocesamiento#Shapes (features, m)]].

**Split estratificado.** Dividir train/test manteniendo la proporción de clases en ambos. Necesario con 9% de positivos. [[03 Preprocesamiento]].

**Z-score.** `(x − μ) / σ` por feature, con `μ` y `σ` de train. Deja cada feature con media 0 y desviación 1. [[03 Preprocesamiento#Z-score]].

**Fuga de datos (leakage).** Usar información del test durante el entrenamiento. Por ejemplo, calcular `μ` y `σ` con todo el dataset, o elegir el umbral mirando el test. [[03 Preprocesamiento]], [[07 Métricas con desbalance]].

**Broadcasting.** Regla de NumPy por la que `b1` de shape `(n_h, 1)` se suma a `Z1` de shape `(n_h, m)` copiándose en cada columna. También por la que `X − mu` funciona con `X` de `(m, 8)` y `mu` de `(8,)`.

**ReLU.** `max(0, z)`. La no linealidad de la capa oculta. Sin ella dos capas lineales colapsan en una. Su derivada es 1 donde `z > 0` y 0 donde no. [[04 La red#forward_prop]].

**Sigmoid.** `1 / (1 + e^(−z))`. Aplasta cualquier real a (0, 1). Es la salida de la red y se lee como probabilidad de púlsar. Con dos clases equivale a softmax. [[04 La red#Por qué sigmoid y no softmax con dos salidas]].

**Softmax.** La generalización de sigmoid a K clases. Es lo que usa el notebook original con 10 dígitos. [[04 La red]].

**One-hot.** Codificar la clase 3 de 10 como `[0,0,0,1,0,0,0,0,0,0]`. El original lo necesita; aquí no, porque `Y` ya es 0 o 1.

**Inicialización de He.** Pesos iniciales `normal × sqrt(2/n_entradas)`. Mantiene la varianza de las activaciones al pasar por capas ReLU. [[04 La red#init_params]].

**Romper la simetría.** Por qué los pesos no pueden empezar en cero: neuronas idénticas reciben gradientes idénticos y nunca se diferencian. [[04 La red#init_params]].

**Binary cross-entropy (BCE).** `−[y log a + (1−y) log(1−a)]` promediado. La pérdida. Sale de la verosimilitud de una Bernoulli. Castiga mucho los errores confiados. [[04 La red#Pérdida: binary cross-entropy]].

**Prior.** La proporción base de positivos, 9.16%. Un modelo que la predice para todo tiene BCE 0.306. Es la referencia para saber si la red aprendió algo de las features. [[04 La red#Dos números de referencia]].

**Gradiente.** La derivada de la pérdida respecto a cada parámetro: cuánto y en qué dirección cambia la pérdida si muevo ese peso. [[04 La red#backward_prop]].

**Backward prop.** Calcular los gradientes de atrás hacia adelante con la regla de la cadena, reutilizando cada paso en el siguiente. [[04 La red#backward_prop]].

**dZ2 = A2 − Y.** El resultado de derivar BCE∘sigmoid (o CE∘softmax). Todo se cancela y queda la diferencia entre predicción y etiqueta. [[04 La red#Paso 1: dZ2, el único que cambia y sin embargo da lo mismo]].

**Gradient check.** Comparar el gradiente analítico con `(L(θ+ε) − L(θ−ε)) / 2ε` para cada peso. Error relativo ~1e-10 si está bien, ~1 si no. [[05 Gradient check]].

**Learning rate (α).** Cuánto se mueve cada peso por paso: `θ ← θ − α dθ`. Chico se arrastra, grande puede divergir. Con datos normalizados aguanta valores grandes. [[06 Entrenamiento#Experimentos con el learning rate]].

**Full batch.** Usar todos los ejemplos de train en cada iteración. Lo opuesto son mini-batches. Aquí cabe de sobra.

**Iteración.** Una vuelta completa de forward, pérdida, backward, update. Con full batch, iteración y epoch son lo mismo.

**Callback.** Una función que se pasa como argumento para que otra la llame en cierto momento. `gradient_descent` llama a `on_iteration` al final de cada iteración, y así la visualización se engancha sin que el loop sepa de matplotlib. [[08 Visualización en vivo]].

**Matriz de confusión.** Los cuatro conteos TP, FP, FN, TN. Todo lo demás sale de ahí. [[07 Métricas con desbalance]].

**Precision.** `TP / (TP + FP)`. De los que llamé púlsar, cuántos lo eran.

**Recall.** `TP / (TP + FN)`. De los púlsares reales, cuántos encontré. La métrica que le importa al astrónomo.

**F1.** Media armónica de precision y recall. Alta solo si las dos lo son.

**Umbral.** El corte sobre `A2` para decidir púlsar o no. 0.5 es arbitrario; bajarlo sube recall y baja precision. Se elige en train, nunca en test. [[07 Métricas con desbalance#El umbral no tiene que ser 0.5]].

**Contribución.** `W[j, i] · a_i`, lo que de verdad viaja por una conexión para un ejemplo concreto. Es lo que dibuja el grosor de las líneas en la visualización. [[08 Visualización en vivo]].

**Flake.** El archivo `flake.nix` que define el entorno reproducible. `nix develop` lo lee y abre un shell con todo instalado. [[01 Entorno con Nix]].
