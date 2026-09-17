from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from redneuro.data import load_data
from redneuro.viz import draw_network

ROOT = Path(__file__).resolve().parent.parent


def load_params():
    d = np.load(ROOT / "data" / "processed" / "params.npz")
    return d["W1"], d["b1"], d["W2"], d["b2"]


if __name__ == "__main__":
    params = load_params()
    _, _, X_test, Y_test = load_data()
    k = int(np.flatnonzero(Y_test.ravel() == 1)[0])   # primer púlsar del test
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_network(ax, params, X_test[:, k:k + 1], Y_test[0, k])
    fig.savefig(ROOT / "data" / "processed" / "red_ejemplo.png", dpi=120, bbox_inches="tight")
    print("guardado data/processed/red_ejemplo.png")
