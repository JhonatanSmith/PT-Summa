#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecasting simple para la prueba:
- Lee raw/dataset_demand_acumulate.csv
- Genera 4 gráficos EDA + ACF/PACF
- Split automático: últimos 4 meses = validación
- Busca SARIMA m=12 en una rejilla mínima (sin log, sin intercepto)
- Gráfico de validación con IC95% + forecast 3 meses con IC95%
- Guarda modelo y artefactos en outputs/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----- Paths y parámetros (sin CLI, para mantenerlo simple) -----
DATA_PATH  = Path("raw/dataset_demand_acumulate.csv")
PICS_DIR   = Path("outputs/pics")
MODELS_DIR = Path("outputs/models")
VAL_MONTHS = 4     # meses para validación
HORIZON    = 3     # meses a pronosticar

# ----- Métricas cortas -----
def rmse(a,b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))
def mae(a,b):  return float(np.mean(np.abs(np.array(a)-np.array(b))))
def mape(a,b):
    a, b = np.array(a), np.array(b); eps = 1e-8
    return float(np.mean(np.abs((a-b)/np.where(a==0, eps, a))) * 100)

def ensure_dirs():
    for d in [PICS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ----- Carga serie -----
def load_ts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "year_month" not in df.columns or "demand" not in df.columns:
        raise ValueError("Se esperan columnas 'year_month' (YYYY-MM) y 'demand'.")
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df = df.sort_values("year_month", ignore_index=True)
    ts = df.set_index("year_month").asfreq("MS")["demand"]
    return ts
def plot_eda_and_corr(ts: pd.Series):
    # 1) Serie original
    plt.figure(); plt.plot(ts.index, ts.values)
    plt.title("Serie original"); plt.xlabel("Fecha"); plt.ylabel("Demand"); plt.tight_layout()
    plt.savefig(PICS_DIR / "eda_1_serie.png"); plt.close()

    # 2) Media móvil 12m
    roll12 = ts.rolling(12, min_periods=1).mean()
    plt.figure(); plt.plot(roll12.index, roll12.values)
    plt.title("Media móvil 12m"); plt.xlabel("Fecha"); plt.ylabel("Demand"); plt.tight_layout()
    plt.savefig(PICS_DIR / "eda_2_media_movil_12m.png"); plt.close()

    # 3) Promedio histórico por mes
    monthly_mean = ts.groupby(ts.index.month).mean()
    plt.figure(); plt.plot(monthly_mean.index, monthly_mean.values, marker="o")
    plt.title("Promedio histórico por mes"); plt.xlabel("Mes (1-12)"); plt.ylabel("Demand promedio")
    plt.tight_layout(); plt.savefig(PICS_DIR / "eda_3_promedio_por_mes.png"); plt.close()

    # 4) Boxplot por mes (Matplotlib 3.9+: usar tick_labels, no labels)
    data_by_month = [ts[ts.index.month==m].dropna().values for m in range(1,13)]
    plt.figure()
    plt.boxplot(data_by_month, tick_labels=[str(m) for m in range(1,13)], showmeans=True)
    plt.title("Distribución por mes (Boxplot)"); plt.xlabel("Mes (1-12)"); plt.ylabel("Demand")
    plt.tight_layout(); plt.savefig(PICS_DIR / "eda_4_boxplot_mes.png"); plt.close()

    # ACF
    acf_vals = acf(ts.dropna(), nlags=24, fft=True)
    plt.figure()
    x = range(len(acf_vals))
    # quitar use_line_collection (el parámetro ya no existe en 3.9+)
    plt.stem(x, acf_vals, basefmt=" ")
    plt.title("ACF hasta 24 rezagos"); plt.xlabel("Rezago"); plt.ylabel("ACF")
    plt.tight_layout(); plt.savefig(PICS_DIR / "acf.png"); plt.close()

    # PACF
    pacf_vals = pacf(ts.dropna(), nlags=24, method="ywm")
    plt.figure()
    x = range(len(pacf_vals))
    plt.stem(x, pacf_vals, basefmt=" ")
    plt.title("PACF hasta 24 rezagos"); plt.xlabel("Rezago"); plt.ylabel("PACF")
    plt.tight_layout(); plt.savefig(PICS_DIR / "pacf.png"); plt.close()

# ----- Split -----
def split_ts(ts: pd.Series, val_months=VAL_MONTHS):
    last_obs = ts.index.max()
    valid_start = last_obs - pd.DateOffset(months=val_months-1)
    train_end = valid_start - pd.DateOffset(months=1)
    y_tr = ts.loc[:train_end]
    y_va = ts.loc[valid_start:]
    return y_tr, y_va

# ----- Selección SARIMA (grid mínima, sin intercepto, m=12) -----
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
    return best

# ----- Plots de validación y forecast con IC -----
def plot_validation(y_tr, y_va, fit, label: str):
    pr = fit.get_prediction(start=y_va.index[0], end=y_va.index[-1])
    mean, ci = pr.predicted_mean, pr.conf_int(0.05)
    plt.figure()
    plt.plot(y_tr.index, y_tr.values, label="Train")
    plt.plot(y_va.index, y_va.values, label="Valid (real)")
    plt.plot(mean.index, mean.values, label=label)
    plt.fill_between(mean.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="IC 95%")
    plt.title("Validación SARIMA (m=12) — sin intercepto")
    plt.xlabel("Fecha"); plt.ylabel("Demand"); plt.legend(); plt.tight_layout()
    plt.savefig(PICS_DIR / "validacion_modelo.png"); plt.close()

def retrain_and_forecast(ts: pd.Series, order, seasonal_order, horizon=HORIZON):
    fit_full = SARIMAX(ts, order=order, seasonal_order=seasonal_order, trend='n',
                       enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=500)
    fc_res  = fit_full.get_forecast(steps=horizon)
    fc_mean = fc_res.predicted_mean
    fc_ci   = fc_res.conf_int(0.05)

    plt.figure()
    plt.plot(ts.index, ts.values, label="Serie")
    plt.plot(fc_mean.index, fc_mean.values, label="Forecast 3m")
    plt.fill_between(fc_mean.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label="IC 95%")
    plt.title("Pronóstico 3 meses"); plt.xlabel("Fecha"); plt.ylabel("Demand")
    plt.legend(); plt.tight_layout()
    plt.savefig(PICS_DIR / "forecast_con_ic.png"); plt.close()

    return fit_full, fc_mean, fc_ci

# ----- Guardado de artefactos -----
def persist_artifacts(fit_full, order, seasonal_order, fc_mean, fc_ci, metrics):
    model_name = f"SARIMA_{order}_x_{seasonal_order}_trend-n"
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Modelo
    fit_full.save(model_dir / "model.pkl")
    # Forecast + IC
    out_fc = pd.concat([fc_mean.rename("forecast"),
                        fc_ci.rename(columns={fc_ci.columns[0]:"lo95", fc_ci.columns[1]:"hi95"})], axis=1)
    out_fc.to_csv(model_dir / "forecast.csv", index=True)
    # Metadata
    meta = {
        "order": order, "seasonal_order": seasonal_order,
        "metrics_valid": metrics, "val_months": VAL_MONTHS, "horizon": HORIZON
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_dir

# ----- Orquestación -----
def main():
    ensure_dirs()
    print("[1/6] Cargando serie…")
    ts = load_ts(DATA_PATH)

    print("[2/6] Generando EDA + ACF/PACF…")
    plot_eda_and_corr(ts)

    print("[3/6] Split (últimos 4 meses = validación)…")
    y_tr, y_va = split_ts(ts, VAL_MONTHS)

    print("[4/6] Seleccionando SARIMA (rejilla mínima)…")
    best = select_sarima(y_tr, y_va)
    name = f"SARIMA {best['order']}x{best['seasonal_order']}"
    print(f"    Mejor modelo: {name} | RMSE={best['RMSE']:.2f}  MAE={best['MAE']:.2f}  MAPE={best['MAPE']:.2f}%")

    print("[5/6] Guardando gráfico de validación con IC…")
    plot_validation(y_tr, y_va, best["fit"], label=name)

    print("[6/6] Reentrenando con toda la serie y pronosticando 3 meses…")
    fit_full, fc_mean, fc_ci = retrain_and_forecast(ts, best["order"], best["seasonal_order"])

    model_dir = persist_artifacts(
        fit_full, best["order"], best["seasonal_order"], fc_mean, fc_ci,
        {"RMSE": best["RMSE"], "MAE": best["MAE"], "MAPE": best["MAPE"]}
    )
    print(f"Listo ✅  Artefactos en: {model_dir}")
    print(f"Figuras en: {PICS_DIR}")

if __name__ == "__main__":
    main()
