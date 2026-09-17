---
tags: [redneuro, errores, debugging]
---
# Errores y lecciones

Índice: [[index|Red neuronal from scratch]].

Todo lo que se rompió, en orden, con lo que significaba el error. Casi ninguno fue de la red; casi todos fueron de las herramientas alrededor. Eso también es una lección.

## Fase 1

**`bash: ~jupyter: orden no encontrada`.** Un typo: una virgulilla antes del comando. El shell interpreta `~nombre` como "la carpeta home del usuario nombre", así que estaba buscando un usuario llamado jupyter. Sin la virgulilla funcionó.

## Fase 2

**`wc -l` dijo 0 líneas en un archivo de 17,898 filas, y `head -3` mostró una sola línea con basura al final.** El CSV de UCI usa `\r` como fin de línea, no `\n`. `wc -l` cuenta `\n`. `head -3` no encontró tres saltos y volcó el archivo entero, pero cada `\r` devuelve el cursor al inicio de la línea en la terminal y todas las filas se dibujaron una encima de otra. Se confirmó con `head -c 200 archivo | od -c`. Pandas lo lee bien; no hubo que tocar nada. Detalle en [[02 Datos HTRU2#El archivo raro]].

**`NotJSONError: Notebook does not appear to be JSON`.** Escribí el `.ipynb` a mano con un editor de texto, como si fuera un `.py`. Un notebook es un JSON con celdas, metadatos y salidas. Se crea desde Jupyter (New → Python 3), no con un editor. Borré el archivo y lo creé desde la interfaz.

**`FileNotFoundError: data/raw/HTRU_2.csv`** desde el notebook. El directorio de trabajo de un notebook es la carpeta donde vive el notebook, `notebooks/`, no la raíz del repo. La ruta tenía que ser `Path("..") / "data" / "raw" / "HTRU_2.csv"`. Se puede confirmar con `import os; os.getcwd()` en una celda. Es una de las razones por las que el notebook quedó solo para explorar: en `src/` la raíz se calcula desde `__file__` y el problema desaparece.

## Fase 5

**`ModuleNotFoundError: No module named 'redneuro.train'`.** El archivo se había guardado como `trein.py`. Un `ls src/redneuro/` lo mostró al instante. `mv` y listo. Lección barata: cuando Python dice que un módulo no existe, lo primero es un `ls`, no releer el código.

## Fase 6

Nada se rompió, pero hubo una **predicción equivocada** que vale registrar: esperaba que `alpha = 3.0` divergiera y no lo hizo, dio mejor resultado que 0.1. La explicación es la normalización: con la superficie de pérdida bien condicionada se pueden dar pasos grandes. Está en [[06 Entrenamiento#Experimentos con el learning rate]].

## Fase 8

**Aviso `No data for colormapping provided via 'c'`.** Cuando una capa estaba "apagada" pasaba el color blanco como texto junto con `cmap`, `vmin`, `vmax`, y matplotlib avisaba que los ignoraba. Inofensivo pero ruidoso. Se resolvió con un helper `_draw_nodes` que usa `color="white"` sin cmap cuando no hay valores, y `c=valores` con cmap cuando los hay.

**Prueba automatizada colgada seis minutos a 0.6% de CPU.** Para probar `train_live.py` sin ventana puse las pausas en `0.0` con el backend `Agg`. Resulta que `plt.pause(0)` en un backend sin GUI llama a `start_event_loop(0)`, y cero ahí significa "sin timeout": espera para siempre. No era un bug del script (con la ventana real y `0.001` no pasa), era del arnés de prueba. Regla: ninguna pausa en cero, nunca.

**`UserWarning: FigureCanvasAgg is non-interactive` convertido en excepción.** Segunda vez que mi propia prueba se rompió sola: había puesto `-W error::UserWarning` para que cualquier aviso fallara, y eso incluyó el aviso normal y esperado de "no hay pantalla". Intenté filtrar ese aviso con `filterwarnings("ignore", ...)`, pero lo puse **después** de `simplefilter("error")`, y matplotlib revisa los filtros en orden: el primero que coincide gana. Al invertir el orden funcionó. Dos errores del harness en fila; el script en sí nunca falló.

**Avisos de Pyright `Import "numpy" could not be resolved`.** El editor corre fuera de `nix develop`, así que no ve el Python del flake. No es un problema del código. Se arregla abriendo el editor desde dentro del `nix develop`, o apuntando el LSP al intérprete del store de Nix.

## Lo que se repite

- Cuando algo "se ve raro" en un archivo, `od -c`. Cuando un módulo "no existe", `ls`. Los dos comandos resuelven en segundos cosas que releyendo código tardan media hora.
- Los shapes correctos no demuestran nada. El único error real de la red que este proyecto podría haber tenido (un signo o una transpuesta en backward) lo habría atrapado solo el [[05 Gradient check]], y por eso está antes de entrenar.
- La mitad de los errores fueron del arnés de prueba, no de lo probado. Vale la pena leer el traceback completo antes de tocar el código bajo prueba.
