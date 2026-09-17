---
tags: [redneuro, comandos, chuleta]
---

Índice: [[index|Red neuronal from scratch]]. Todo se corre desde la raíz del repo (`~/Projects/Redneuro`) y dentro de `nix develop`.

## De cero a resultado

```bash
cd ~/Projects/Redneuro
nix develop                          # la primera vez tarda, descarga todo
python scripts/download_data.py      # baja HTRU_2.csv a data/raw (idempotente)
pytest tests/                        # gradient check, debe dar 1 passed
python scripts/train.py              # entrena, evalúa en test, guarda params.npz y loss_history.npy
```

## Ver la red

```bash
python scripts/animate.py                   # imagen estática en data/processed/red_ejemplo.png
python scripts/train_live.py                # ventana: entrena en vivo, luego candidatos uno por uno
python scripts/train_live.py --inferencia   # solo la parte de candidatos, con los pesos guardados
```

Detalles y perillas en [[08 Visualización en vivo]].

## Exploración

```bash
jupyter notebook          # abre el navegador; el notebook está en notebooks/01_explore.ipynb
```

Recordar que dentro del notebook la ruta al CSV lleva `..` adelante ([[Errores y lecciones#Fase 2]]).

## Chequeos rápidos que sirvieron

```bash
# qué backend de matplotlib hay
python -c "import matplotlib; print(matplotlib.get_backend())"

# ver los bytes de un archivo sospechoso
head -c 200 data/raw/HTRU_2.csv | od -c | head

# contar filas de un archivo con \r
tr '\r' '\n' < data/raw/HTRU_2.csv | wc -l

# ver el gradient check con detalle
pytest tests/ -v -s

# correr la visualización sin ventana (para pruebas)
MPLBACKEND=Agg python scripts/train_live.py
```

## Git

Si el repo es git, el flake solo ve archivos que git conoce ([[01 Entorno con Nix]]):

```bash
git add flake.nix
```

Pendiente: README.md y primer commit. `.gitignore` ya excluye `data/`, `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/` y `result`.
