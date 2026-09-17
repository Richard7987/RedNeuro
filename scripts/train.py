import numpy as np

from redneuro.data import load_data
from redneuro.metrics import confusion_matrix, precision_recall_f1, predict, threshold_sweep
from redneuro.model import forward_prop
from redneuro.train import gradient_descent

ALPHA = 0.1
ITERATIONS = 1000
N_H = 16


def main():
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"train: {X_train.shape[1]} ejemplos, test: {X_test.shape[1]} ejemplos\n")

    (W1, b1, W2, b2), history = gradient_descent(
        X_train, Y_train, n_h=N_H, alpha=ALPHA, iterations=ITERATIONS, log_every=200
    )
    np.save("data/processed/loss_history.npy", history)
    np.savez("data/processed/params.npz", W1=W1, b1=b1, W2=W2, b2=b2)


    _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train)
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)

    sweep = threshold_sweep(A2_train, Y_train)
    threshold = sweep[np.argmax(sweep[:, 3]), 0]
    print(f"\numbral elegido en train (max F1): {threshold:.2f}")

    for t in (0.5, threshold):
        Y_pred = predict(A2_test, t)
        tp, fp, fn, tn = confusion_matrix(Y_test, Y_pred)
        p, r, f = precision_recall_f1(Y_test, Y_pred)
        acc = (tp + tn) / Y_test.size
        print(f"\nTEST con umbral {t:.2f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")


if __name__ == "__main__":
    main()
