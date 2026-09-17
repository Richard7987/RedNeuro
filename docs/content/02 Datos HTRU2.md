---
tags: [redneuro, datos, htru2]
---

Anterior: [[01 Entorno con Nix]]. Siguiente: [[03 Preprocesamiento]]. Índice: [[index|Red neuronal from scratch]].

Qué es el dataset y por qué se baja con un script está en el índice: [[index#Qué es el dataset|Red neuronal from scratch]]. Aquí va lo que pasó al bajarlo y explorarlo.

## El script de descarga

`scripts/download_data.py`, solo librería estándar:

```python
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CSV_NAME = "HTRU_2.csv"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_NAME
    if csv_path.exists():
        print(f"Ya existe {csv_path}, no se descarga de nuevo.")
        return

    zip_path = RAW_DIR / "htru2.zip"
    print(f"Descargando {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(CSV_NAME, RAW_DIR)

    zip_path.unlink()
    print(f"Listo: {csv_path}")


if __name__ == "__main__":
    download()
```

Dos cosas que vale la pena copiar en otros proyectos. `Path(__file__).resolve().parent.parent` es la raíz del repo calculada desde donde está el script, así funciona sin importar desde qué carpeta lo lances. Y el `if csv_path.exists()` lo hace idempotente: correrlo dos veces no descarga dos veces.

## El archivo raro

Después de bajarlo, `head -3` mostró una sola línea con un final extraño, y `wc -l` dijo **0 líneas**. Un archivo de 17,898 filas con cero líneas.

La explicación está en los bytes. Con `od -c` se ve que el archivo separa filas con `\r` (retorno de carro, el formato de Mac clásico) y no con `\n`. `wc -l` cuenta `\n`, por eso da cero. Y `head -3` no encontró tres saltos de línea, así que imprimió el archivo completo, pero cada `\r` devuelve el cursor al inicio de la misma línea de la terminal y las 17,898 filas se dibujaron una encima de otra. Lo que quedó visible fue la última fila con residuos de filas anteriores más largas.

```bash
tr '\r' '\n' < data/raw/HTRU_2.csv | wc -l    # 17897, más la última sin terminador = 17898
```

Pandas lo lee sin problema porque reconoce `\r` como fin de línea. No hubo que convertir nada. Pero la lección quedó: cuando un archivo "se ve raro", `od -c | head` resuelve la duda en segundos. Más en [[Errores y lecciones#Fase 2]].

## Exploración

En `notebooks/01_explore.ipynb`. Dos tropiezos con el notebook en [[Errores y lecciones#Fase 2]]: uno por escribir el `.ipynb` a mano como si fuera texto, otro por la ruta relativa.

```python
import pandas as pd
from pathlib import Path

COLUMNS = [
    "ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",
    "pulsar",
]

csv_path = Path("..") / "data" / "raw" / "HTRU_2.csv"
df = pd.read_csv(csv_path, header=None, names=COLUMNS)

print("shape:", df.shape)
print(df["pulsar"].value_counts())
print("proporcion positiva:", df["pulsar"].mean().round(4))
print(df.describe().T[["mean", "std", "min", "max"]])
```

`ip_` es integrated profile y `dm_` es la curva DM-SNR. Salida:

```
shape: (17898, 9)
pulsar
0    16259
1     1639
proporcion positiva: 0.0916

                   mean         std        min          max
ip_mean      111.079968   25.652935   5.812500   192.617188
ip_std        46.549532    6.843189  24.772042    98.778911
ip_kurtosis    0.477857    1.064040  -1.876011     8.069522
ip_skewness    1.770279    6.167913  -1.791886    68.101622
dm_mean       12.614400   29.472897   0.213211   223.392141
dm_std        26.326515   19.470572   7.370432   110.642211
dm_kurtosis    8.303556    4.506092  -3.139270    34.539844
dm_skewness  104.857709  106.514540  -1.976976  1191.000837
```

## Lo que hay que ver en esa tabla

Mirar la columna `std` de arriba a abajo. `ip_kurtosis` tiene desviación 1. `dm_skewness` tiene 106 y llega a 1191. Entre la feature más chica y la más grande hay un factor de cien. Eso es el problema que resuelve la normalización, y el porqué matemático está en [[03 Preprocesamiento#Por qué los rangos dispares dañan el gradiente]].

También: `dm_skewness` tiene media 105 pero máximo 1191. Cola pesada. La normalización z-score no elimina eso, solo lo lleva a escala manejable. Si algún día estorba, un logaritmo antes de normalizar es la salida.

## Histogramas por clase

Esto es lo único para lo que sirve el notebook. Con `density=True`, porque hay diez veces más negativos que positivos y con conteos la clase púlsar se ve como una rayita aplastada.

```python
import matplotlib.pyplot as plt

features = COLUMNS[:-1]
colors = {0: "#2a78d6", 1: "#eb6834"}
labels = {0: "no púlsar", 1: "púlsar"}

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, features):
    for cls in (0, 1):
        ax.hist(df.loc[df["pulsar"] == cls, feat], bins=50, density=True,
                alpha=0.6, color=colors[cls], label=labels[cls])
    ax.set_title(feat)
axes[0, 0].legend()
fig.tight_layout()
plt.show()
```

`ip_kurtosis` e `ip_skewness` separan bastante bien las dos clases. Las `dm_` se solapan más. Eso adelanta qué features va a aprovechar la red, y después se ve confirmado en [[08 Visualización en vivo]]: las líneas más gruesas salen de esas entradas.
