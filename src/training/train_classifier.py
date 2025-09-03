# -*- coding: utf-8 -*-
"""
Entrena el clasificador (LogReg balanceada) de Alpha vs Betha y deja artefactos
compatibles con la API y Streamlit.

Cambios clave:
- Lee de:          data/raw/dataset_alpha_betha.csv
- Guarda en:       src/artifacts/classifier/
- Sin CLI:         solo ejecutar este script
- Umbral óptimo:   se busca por F1(Betha) y se guarda en model_threshold.txt
- Reporte JSON:    eval_report.json con métricas y metadatos

Nota: Mantenemos TARGET="class" con etiquetas 'Alpha'/'Betha' (como tu script).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# --------------------- Paths y configuración ---------------------
ROOT = Path(__file__).resolve().parents[2]           # repo root
RAW_PATH = ROOT / "data" / "raw" / "dataset_alpha_betha.csv"

ART_DIR = ROOT / "src" / "artifacts" / "classifier"
ART_DIR.mkdir(parents=True, exist_ok=True)

PIPE_PATH = ART_DIR / "model_best_pipeline.joblib"
THR_PATH  = ART_DIR / "model_threshold.txt"
REPORT    = ART_DIR / "eval_report.json"

TARGET = "class"                 # 'Alpha' / 'Betha'
ID_COLS = ["autoid"]             # columnas a excluir
POS_LABEL = "Betha"              # clase positiva para métricas/umbral
EXCLUDE_FEATURES = ["demand"]    # Esto no hace part del analisis

# --------------------- Utilidades ---------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    df = pd.read_csv(path)
    return clean_cols(df)

def find_best_threshold_f1(y_true, proba_pos, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (proba_pos >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# --------------------- Entrenamiento ---------------------
def train_and_eval(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"No encuentro la columna objetivo '{TARGET}' en el dataset.")
    # quitar IDs si existen
    cols_to_drop = [c for c in [TARGET] + ID_COLS + EXCLUDE_FEATURES if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET].astype("category")

    # tipado simple
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # split estratificado
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # OHE denso (compatibilidad amplia)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipe = Pipeline([("preproc", preproc), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    # --- Probabilidades positivas para la clase Betha ---
    classes = list(pipe.classes_)
    if POS_LABEL not in classes:
        raise ValueError(f"No encuentro la clase positiva '{POS_LABEL}' en {classes}")
    pos_idx = classes.index(POS_LABEL)
    proba_te = pipe.predict_proba(X_te)[:, pos_idx]

    # mapear y a 0/1 para métricas umbralizadas
    y_te_bin = (y_te == POS_LABEL).astype(int).values

    # ROC AUC (probabilidades)
    auc = roc_auc_score(y_te_bin, proba_te)

    # umbral óptimo por F1
    best_thr, best_f1 = find_best_threshold_f1(y_te_bin, proba_te)

    # métricas con umbral óptimo
    y_pred_thr = (proba_te >= best_thr).astype(int)
    acc = accuracy_score(y_te_bin, y_pred_thr)
    bal_acc = balanced_accuracy_score(y_te_bin, y_pred_thr)
    f1b = f1_score(y_te_bin, y_pred_thr, pos_label=1)

    # reporte legible (también guardamos por completitud)
    y_pred_labels = np.where(y_pred_thr == 1, POS_LABEL, "Alpha")
    report_txt = classification_report(y_te, y_pred_labels, digits=3)

    # ---------------- Guardar artefactos ----------------
    joblib.dump(pipe, PIPE_PATH)
    THR_PATH.write_text(str(best_thr), encoding="utf-8")

    meta = {
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "positive_class_label": POS_LABEL,
        "n_samples_train": int(X_tr.shape[0]),
        "n_samples_test": int(X_te.shape[0]),
        "n_features": int(X.shape[1]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "threshold": float(best_thr),
        "metrics_test_at_threshold": {
            "roc_auc": float(auc),
            "balanced_accuracy": float(bal_acc),
            "f1_positive": float(f1b),
            "accuracy": float(acc)
        },
        "random_state": 42,
        "notes": "Métricas umbralizadas sobre test; POS_LABEL='Betha'."
    }
    REPORT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Pipeline  ->", PIPE_PATH)
    print("[OK] Umbral    ->", THR_PATH, f"({best_thr:.6f})")
    print("[OK] Reporte   ->", REPORT)
    print("--- Resumen métricas (test) ---")
    print(json.dumps(meta["metrics_test_at_threshold"], indent=2))
    print("\nClassification report (legible):\n", report_txt)


def main():
    df = load_data(RAW_PATH)
    train_and_eval(df)


if __name__ == "__main__":
    main()
