# src/training/utils_metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def binclass_metrics(y_true, proba_pos, threshold=0.5):
    y_true = np.asarray(y_true)
    proba_pos = np.asarray(proba_pos)
    y_pred = (proba_pos >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold_used": float(threshold)
    }

def find_best_threshold_f1(y_true, proba_pos, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)  # simple y suficiente
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (proba_pos >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), float(best_f1)
