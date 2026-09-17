from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "HTRU_2.csv"


def load_raw():
    df = pd.read_csv(CSV_PATH, header=None)
    data = df.to_numpy(dtype=np.float64)
    X = data[:, :-1]   # (m, 8)  todas las columnas menos la última
    y = data[:, -1]    # (m,)    la última columna es la clase
    return X, y


def stratified_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)   # posiciones de esta clase
        rng.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def normalize(X_train, X_test):
    mu = X_train.mean(axis=0)     # (8,)  una media por feature
    sigma = X_train.std(axis=0)   # (8,)  una desviación por feature
    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma   # mismas mu y sigma, no se recalculan
    return X_train_norm, X_test_norm, mu, sigma


def load_data(test_frac=0.2, seed=0):
    X, y = load_raw()
    Xtr, ytr, Xte, yte = stratified_split(X, y, test_frac, seed)
    Xtr, Xte, mu, sigma = normalize(Xtr, Xte)
    X_train = Xtr.T               # (8, m_train)
    X_test = Xte.T                # (8, m_test)
    Y_train = ytr.reshape(1, -1)  # (1, m_train)
    Y_test = yte.reshape(1, -1)   # (1, m_test)
    return X_train, Y_train, X_test, Y_test
