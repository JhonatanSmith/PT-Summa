# -*- coding: utf-8 -*-
"""
Entrena SARIMA (m=12) y deja artefactos compatibles con la API y Streamlit.

Cambios clave:
- Lee de:          data/raw/dataset_demand_acumulate.csv
- Guarda en:       src/artifacts/sarima/
- Sin CLI:         solo ejecutar este script
- Artefactos:      model_sarima.joblib, config_sarima.json, backtest_metrics.json
- Validación:      últimos VAL_MONTHS meses; Forecast: HORIZON meses

Mantiene la lógica simple de tu script original (rejilla mínima, trend 'n').
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------- Paths y parámetros ---------------------
ROOT = Path(__file__).resolve().parents[2]                 # repo root
DATA_PATH  = ROOT / "data" / "raw" / "dataset_demand_acumulate.csv"

ART_DIR = ROOT / "src" / "artifacts" / "sarima"
ART_DIR.mkdir(parents=True, exist_ok=True)

MODEL   = ART_DIR / "model_sarima.joblib"
CONFIG  = ART_DIR / "config_sarima.json"
METRICS = ART_DIR / "backtest_metrics.json"

VAL_MONTHS = 4   # meses para val
HORIZON    = 3   # meses a pronost
FREQ       = "MS"

# --------------------- Merteicas ---------------------
def rmse(a,b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))
def mae(a,b):  return float(np.mean(np.abs(np.array(a)-np.array(b))))
def mape(a,b):
    a, b = np.array(a), np.array(b); eps = 1e-8
    return float(np.mean(np.abs((a-b)/np.where(a==0, eps, a))) * 100)

# --------------------- Utilidades ---------------------
def load_ts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "year_month" not in df.columns or "demand" not in df.columns:
        raise ValueError("Se esperan columnas 'year_month' (YYYY-MM) y 'demand'.")
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df = df.sort_values("year_month", ignore_index=True)
    ts = df.set_index("year_month").asfreq(FREQ)["demand"]
    return ts

def split_ts(ts: pd.Series, val_months=VAL_MONTHS):
    last_obs = ts.index.max()
    valid_start = last_obs - pd.DateOffset(months=val_months-1)
    train_end = valid_start - pd.DateOffset(months=1)
    y_tr = ts.loc[:train_end]
    y_va = ts.loc[valid_start:]
    return y_tr, y_va

def select_sarima(y_tr, y_va):
    orders  = [(1,0,1),(1,0,2),(2,0,1),(2,0,2)]
    sorders = [(1,1,0,12),(1,1,1,12),(0,1,1,12)]
    best = None
    for o in orders:
        for so in sorders:
            try:
                fit = SARIMAX(y_tr, order=o, seasonal_order=so, trend='n',
                              enforce_stationarity=False, enforce_invertibility=False
                             ).fit(disp=False, maxiter=500)
                pred = fit.predict(start=y_va.index[0], end=y_va.index[-1])
                score = rmse(y_va, pred)
                if (best is None) or (score < best["RMSE"]):
                    best = {"order":o, "seasonal_order":so, "fit":fit, "pred":pred,
                            "RMSE":score, "MAE":mae(y_va,pred), "MAPE":mape(y_va,pred)}
            except Exception:
                continue
    if best is None:
        raise RuntimeError("No se pudo ajustar ningún SARIMA en la rejilla mínima.")
    return best

# --------------------- Entrenar y persistir ---------------------
def train_and_persist(ts: pd.Series):
    # split y selección
    y_tr, y_va = split_ts(ts, VAL_MONTHS)
    best = select_sarima(y_tr, y_va)

    # reentrenar con toda la serie
    fit_full = SARIMAX(ts, order=best["order"], seasonal_order=best["seasonal_order"], trend='n',
                       enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=500)

    # forecast simple (para referencia)
    fc_res  = fit_full.get_forecast(steps=HORIZON)
    fc_mean = fc_res.predicted_mean
    fc_ci   = fc_res.conf_int(0.05)  # (lo, hi)

    # Guardar artefactos (formato simple y estable)
    joblib.dump(
        {"model": fit_full, "freq": FREQ, "last_index": ts.index.max()},
        MODEL
    )

    config = {
        "order":       list(best["order"]),
        "seasonal_order": list(best["seasonal_order"]),
        "freq":        FREQ,
        "date_col":    "year_month",
        "value_col":   "demand",
        "val_months":  int(VAL_MONTHS),
        "horizon":     int(HORIZON)
    }
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = {
        "RMSE":  float(best["RMSE"]),
        "MAE":   float(best["MAE"]),
        "MAPE":  float(best["MAPE"]),
        "n_total": int(len(ts)),
        "n_train": int(len(y_tr)),
        "n_valid": int(len(y_va))
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[OK] Guardado SARIMA     ->", MODEL)
    print("[OK] Guardado config     ->", CONFIG)
    print("[OK] Guardado métricas   ->", METRICS)
    print("Mejor orden:", best["order"], "x", best["seasonal_order"])
    print("Métricas validación:", metrics)

def main():
    ts = load_ts(DATA_PATH)
    train_and_persist(ts)

if __name__ == "__main__":
    main()
