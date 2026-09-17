---
tags: [redneuro, nix, entorno]
---
# 01 Entorno con Nix

Volver al índice: [[index|Red neuronal from scratch]]. Siguiente: [[02 Datos HTRU2]].

## Por qué un flake y no pip

Estoy en NixOS y no hay `python3` en el PATH. Podría haber usado `uv` o un venv, pero la gracia de Nix es que la definición del entorno vive en el repo igual que el código. Cualquiera que clone el proyecto hace `nix develop` y tiene exactamente el mismo Python con las mismas librerías, sin instalar nada a mano.

## El flake.nix

```nix
{
  description = "Red neuronal desde cero con NumPy para HTRU2";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: with ps; [
        numpy
        pandas
        matplotlib
        scikit-learn
        jupyter
        pytest
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
        shellHook = ''
          export PYTHONPATH="$PWD/src:$PYTHONPATH"
          echo "Entorno Redneuro listo: $(python --version)"
        '';
      };
    };
}
```

Tres partes:

- `inputs`: de dónde salen los paquetes (nixpkgs unstable).
- `python3.withPackages`: arma un solo intérprete que ya trae las librerías. Es el equivalente de un venv con requirements.txt, pero declarativo.
- `shellHook`: se ejecuta al entrar. La línea del `PYTHONPATH` apuntando a `src/` es lo que permite hacer `from redneuro.model import ...` desde `scripts/` y `tests/` sin instalar el paquete.

Aclaración sobre las librerías: la red es NumPy puro. `scikit-learn` solo se usa para verificar que mis métricas dan lo mismo que las suyas ([[07 Métricas con desbalance]]), y `pytest` para el [[05 Gradient check]].

## Primer `nix develop`

La primera vez tarda porque descarga todo. Al final imprime `Entorno Redneuro listo: Python 3.14.7`. Se verifica con:

```bash
python -c "import numpy, pandas, matplotlib, sklearn; print(numpy.__version__)"
```

Salió `2.5.2`.

> [!warning] Flakes y git
> Cuando la carpeta es un repo git, `nix develop` **solo ve los archivos que git conoce**. Si creas `flake.nix` y no haces `git add flake.nix`, el flake se queja de que no existe. No hace falta commitear, con el `add` alcanza. Lo mismo aplica a cualquier archivo que el flake necesite leer.

## Estructura de carpetas

La razón de no hacer todo en un notebook como el original está en [[index#Organización|Red neuronal from scratch]], pero la versión corta: en un notebook las celdas se pueden ejecutar en desorden y quedan variables vivas de corridas anteriores, lo que hace que cuando algo falla no sepas si es un bug o una celda mal ejecutada. Y lo más importante, en un notebook no puedes escribir un test que importe tus funciones. El gradient check, que es la única forma seria de saber que el backward prop está bien, necesita que las funciones vivan en un módulo.

Se creó todo con:

```bash
mkdir -p data/raw data/processed notebooks src/redneuro scripts tests
touch src/redneuro/__init__.py
```

El `__init__.py` vacío es lo que convierte a `src/redneuro/` en un paquete importable.

## Lo que se rompió aquí

Nada grave. Ver [[Errores y lecciones#Fase 1]] por un typo con la virgulilla que dejó a bash buscando un usuario llamado jupyter.
