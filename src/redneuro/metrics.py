import numpy as np


def predict(A2, threshold=0.5):
    return (A2 > threshold).astype(int)


def confusion_matrix(Y, Y_pred):
    Y = Y.astype(bool)
    Y_pred = Y_pred.astype(bool)
    tp = int(np.sum(Y & Y_pred))
    fp = int(np.sum(~Y & Y_pred))
    fn = int(np.sum(Y & ~Y_pred))
    tn = int(np.sum(~Y & ~Y_pred))
    return tp, fp, fn, tn


def precision_recall_f1(Y, Y_pred):
    tp, fp, fn, tn = confusion_matrix(Y, Y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def threshold_sweep(A2, Y, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        p, r, f = precision_recall_f1(Y, predict(A2, t))
        rows.append((t, p, r, f))
    return np.array(rows)
