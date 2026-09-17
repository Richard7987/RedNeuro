"""Visualización en vivo de la red neuronal.

Dos etapas en la misma ventana:

1. Entrenamiento: los pesos (líneas) cambian iteración a iteración y el panel derecho
   muestra la pérdida bajando.
2. Inferencia: con la red ya entrenada, los candidatos del test pasan UNO POR UNO, despacio:
   entra el dato, se enciende la capa oculta, se enciende la salida, y aparece el veredicto.
   El panel derecho acumula aciertos y errores.

Uso (dentro de `nix develop`, desde la raíz del repo):
    python scripts/train_live.py                 # entrena y luego inferencia
    python scripts/train_live.py --inferencia    # salta el entrenamiento, carga data/processed/params.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from redneuro.data import load_data
from redneuro.train import gradient_descent
from redneuro.viz import draw_confusion_bars, draw_network

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "data" / "processed" / "params.npz"

ALPHA, ITERATIONS, N_H = 0.1, 1000, 16
THRESHOLD = 0.5
PRIOR_LOSS = 0.306        # BCE de predecir siempre la proporción base (9.16%)

TRAIN_DRAW_EVERY = 5      # durante el entrenamiento, redibujar cada N iteraciones
TRAIN_PAUSE = 0.001       # segundos por cuadro durante el entrenamiento

INFER_EXAMPLES = 40       # cuántos candidatos del test pasar uno por uno
INFER_PAUSE = 0.7         # segundos por fase (hay 4 fases por candidato)
INFER_HOLD = 3.0          # segundos que se queda el veredicto en pantalla antes del siguiente candidato
INFER_SEED = 0


def setup_figure():
    plt.ion()
    fig, (ax_net, ax_side) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    fig.canvas.manager.set_window_title("Redneuro: la red trabajando")
    return fig, ax_net, ax_side


def setup_loss_panel(ax):
    ax.clear()
    ax.set_yscale("log")
    ax.set_xlabel("iteración de gradient descent")
    ax.set_ylabel("pérdida BCE en train (escala log)")
    ax.set_title("qué tan equivocadas están las probabilidades\n(más abajo = mejor)")
    ax.axhline(PRIOR_LOSS, color="gray", linestyle="--", linewidth=1,
               label="modelo que solo sabe que 9% son púlsares")
    ax.legend(loc="upper right", fontsize=8)
    (line,) = ax.plot([], [], color="#2a78d6", linewidth=2)
    return line


def train_live(fig, ax_net, ax_side, X_train, Y_train, X_test, Y_test):
    idx_pos = np.flatnonzero(Y_test.ravel() == 1)
    idx_neg = np.flatnonzero(Y_test.ravel() == 0)
    loss_line = setup_loss_panel(ax_side)
    losses = []

    def on_iteration(i, params, loss):
        losses.append(loss)
        if i % TRAIN_DRAW_EVERY != 0:
            return
        frame = i // TRAIN_DRAW_EVERY
        n_ejemplo = frame // 4
        fase = frame % 4 + 1
        pool = idx_pos if n_ejemplo % 2 == 0 else idx_neg
        k = int(pool[n_ejemplo % len(pool)])
        draw_network(ax_net, params, X_test[:, k:k + 1], Y_test[0, k], fase=fase, threshold=THRESHOLD)
        ax_net.set_title(f"ENTRENANDO    iteración {i}    pérdida {loss:.4f}")
        loss_line.set_data(np.arange(len(losses)), losses)
        ax_side.relim()
        ax_side.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(TRAIN_PAUSE)

    params, _ = gradient_descent(
        X_train, Y_train, n_h=N_H, alpha=ALPHA, iterations=ITERATIONS,
        log_every=200, on_iteration=on_iteration,
    )
    ax_net.set_title(f"ENTRENAMIENTO TERMINADO    pérdida final {losses[-1]:.4f}")
    fig.canvas.draw_idle()
    plt.pause(1.5)
    return params


def infer_live(fig, ax_net, ax_side, params, X_test, Y_test):
    rng = np.random.default_rng(INFER_SEED)
    idx_pos = np.flatnonzero(Y_test.ravel() == 1)
    idx_neg = np.flatnonzero(Y_test.ravel() == 0)
    # mitad púlsares y mitad no púlsares, mezclados, para que se vean los dos casos
    n_cada = INFER_EXAMPLES // 2
    orden = np.concatenate([rng.choice(idx_pos, n_cada, replace=False),
                            rng.choice(idx_neg, INFER_EXAMPLES - n_cada, replace=False)])
    rng.shuffle(orden)

    tp = fp = fn = tn = 0
    draw_confusion_bars(ax_side, tp, fp, fn, tn)
    nombres_fase = {1: "entra el dato", 2: "capa oculta", 3: "salida", 4: "veredicto"}

    for n, k in enumerate(orden, start=1):
        x = X_test[:, k:k + 1]
        y = Y_test[0, k]
        for fase in (1, 2, 3, 4):
            a2 = draw_network(ax_net, params, x, y, fase=fase, threshold=THRESHOLD)
            ax_net.set_title(f"INFERENCIA    candidato {n}/{len(orden)}    fase: {nombres_fase[fase]}")
            fig.canvas.draw_idle()
            plt.pause(INFER_PAUSE)
        pred, real = int(a2 > THRESHOLD), int(y)
        if pred == 1 and real == 1:
            tp += 1
        elif pred == 1 and real == 0:
            fp += 1
        elif pred == 0 and real == 1:
            fn += 1
        else:
            tn += 1
        draw_confusion_bars(ax_side, tp, fp, fn, tn)
        fig.canvas.draw_idle()
        plt.pause(INFER_HOLD)   # tiempo para leer el veredicto antes del siguiente candidato

    ax_net.set_title(f"INFERENCIA TERMINADA    {tp + tn} aciertos de {len(orden)}")
    fig.canvas.draw_idle()


def load_params():
    d = np.load(PARAMS_PATH)
    return d["W1"], d["b1"], d["W2"], d["b2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inferencia", action="store_true",
                        help="saltar el entrenamiento y usar los pesos guardados por scripts/train.py")
    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = load_data()
    fig, ax_net, ax_side = setup_figure()

    if args.inferencia:
        params = load_params()
    else:
        params = train_live(fig, ax_net, ax_side, X_train, Y_train, X_test, Y_test)

    infer_live(fig, ax_net, ax_side, params, X_test, Y_test)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
