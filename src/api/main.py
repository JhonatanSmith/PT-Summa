# src/api/main.py
# Ejecuta: python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------
# Rutas de artefactos (fijas)
# -------------------------------
ROOT = Path(__file__).resolve().parents[2]

CLF_DIR = ROOT / "src" / "artifacts" / "classifier"
CLF_PIPE_PATH = CLF_DIR / "model_best_pipeline.joblib"
CLF_THR_PATH  = CLF_DIR / "model_threshold.txt"
CLF_REPORT    = CLF_DIR / "eval_report.json"

SAR_DIR = ROOT / "src" / "artifacts" / "sarima"
SAR_MODEL_PATH  = SAR_DIR / "model_sarima.joblib"
SAR_CONFIG_PATH = SAR_DIR / "config_sarima.json"
SAR_METRICS_PATH= SAR_DIR / "backtest_metrics.json"

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
POS_LABEL = "Betha"   # etiqueta positiva del clasificador

# -------------------------------
# Utilidades locales
# -------------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a snake_case simple."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def to_dataframe(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Lista vacía de registros.")
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("Todos los elementos deben ser objetos JSON (dict).")
        return pd.DataFrame(payload)
    raise ValueError("Formato inválido: use dict o lista de dicts.")

# -------------------------------
# Carga de modelos al iniciar
# -------------------------------
def load_classifier():
    if not CLF_PIPE_PATH.exists():
        raise FileNotFoundError(f"No se encontró el clasificador: {CLF_PIPE_PATH}")
    pipe = joblib.load(CLF_PIPE_PATH)

    thr = DEFAULT_THRESHOLD
    if CLF_THR_PATH.exists():
        try:
            thr = float(CLF_THR_PATH.read_text().strip())
        except Exception:
            pass

    meta = {}
    if CLF_REPORT.exists():
        try:
            meta = json.loads(CLF_REPORT.read_text())
        except Exception:
            meta = {}

    # índice de probabilidad para la clase Betha
    try:
        classes = list(pipe.classes_)
        pos_idx = classes.index(POS_LABEL)
    except Exception as e:
        raise RuntimeError(f"No se encontró la clase '{POS_LABEL}' en el pipeline: {e}")

    return pipe, thr, pos_idx, meta

def load_sarima():
    if not SAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró SARIMA: {SAR_MODEL_PATH}")
    sar_pk = joblib.load(SAR_MODEL_PATH)  # dict: {"model": <SARIMAXResults>, "freq":..., "last_index":...}
    sar_model = sar_pk["model"]

    config, metrics = {}, {}
    if SAR_CONFIG_PATH.exists():
        try:
            config = json.loads(SAR_CONFIG_PATH.read_text())
        except Exception:
            pass
    if SAR_METRICS_PATH.exists():
        try:
            metrics = json.loads(SAR_METRICS_PATH.read_text())
        except Exception:
            pass

    return sar_model, config, metrics

try:
    CLF_PIPE, CLF_THR, CLF_POS_IDX, CLF_META = load_classifier()
    SAR_MODEL, SAR_CONFIG, SAR_METRICS = load_sarima()
except Exception as e:
    raise RuntimeError(f"Error inicializando modelos: {e}")

# Derivar contrato esperado del clasificador
NUM_FEATURES: List[str] = CLF_META.get("num_cols", []) or []
CAT_FEATURES: List[str] = CLF_META.get("cat_cols", []) or []
EXPECTED_FEATURES: List[str] = NUM_FEATURES + CAT_FEATURES

def align_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) snake_case
    2) agrega columnas faltantes:
       - numéricas -> 0
       - categóricas -> "NA"
    3) fuerza numéricas a float (con fillna(0))
    4) reordena exactamente como el entrenamiento
    """
    df = clean_cols(df)

    # agregar faltantes con valores neutros
    for c in EXPECTED_FEATURES:
        if c not in df.columns:
            if c in NUM_FEATURES:
                df[c] = 0
            else:
                df[c] = "NA"

    # convertir numéricas a float (relleno 0 si vienen como string/NaN)
    for c in NUM_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # quedarnos sólo con las esperadas y en el mismo orden
    df = df[EXPECTED_FEATURES]
    return df

# -------------------------------
# FastAPI
# -------------------------------
app = FastAPI(
    title="PT-Summa API",
    description="API mínima para exponer Clasificador Alpha/Betha y SARIMA.",
    version="1.0.0",
)

# -------------------------------
# Esquemas (entrada/salida)
# -------------------------------
class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]  # admite 1 o varios registros

class PredictItem(BaseModel):
    predicted_label: str
    proba_betha: float
    threshold_used: float

class PredictResponse(BaseModel):
    predictions: List[PredictItem]

class ForecastRequest(BaseModel):
    n_steps: int = 3  # horizonte default

class ForecastItem(BaseModel):
    timestamp: str
    forecast: float
    lo95: Union[float, None]
    hi95: Union[float, None]

class ForecastResponse(BaseModel):
    horizon: int
    forecast: List[ForecastItem]

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier": {
            "path": str(CLF_PIPE_PATH),
            "threshold": CLF_THR,
            "positive_label": POS_LABEL,
            "num_features": NUM_FEATURES,
            "cat_features": CAT_FEATURES
        },
        "sarima": {
            "path": str(SAR_MODEL_PATH),
            "config_keys": list(SAR_CONFIG.keys()),
            "metrics_keys": list(SAR_METRICS.keys())
        }
    }

@app.get("/predict/schema")
def predict_schema():
    return {
        "expected_features": EXPECTED_FEATURES,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "notes": "La API normaliza snake_case y completa faltantes con 0 (num) y 'NA' (cat)."
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = to_dataframe(req.data)
        df = align_to_expected(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Entrada inválida: {e}")

    try:
        probas = CLF_PIPE.predict_proba(df)[:, CLF_POS_IDX]
        labels = (probas >= CLF_THR)
        out = [
            PredictItem(
                predicted_label=POS_LABEL if l else "Alpha",
                proba_betha=float(p),
                threshold_used=float(CLF_THR)
            )
            for p, l in zip(probas, labels)
        ]
        return PredictResponse(predictions=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.n_steps <= 0 or req.n_steps > 60:
        raise HTTPException(status_code=400, detail="n_steps debe estar entre 1 y 60.")

    try:
        fc = SAR_MODEL.get_forecast(steps=int(req.n_steps))
        mean = fc.predicted_mean  # pd.Series
        ci = fc.conf_int(alpha=0.05) if hasattr(fc, "conf_int") else None

        items: List[ForecastItem] = []
        for ts in mean.index:
            lo = hi = None
            if ci is not None and len(ci.columns) >= 2:
                lo = float(ci.loc[ts, ci.columns[0]])
                hi = float(ci.loc[ts, ci.columns[1]])
            items.append(ForecastItem(
                timestamp=str(ts),
                forecast=float(mean.loc[ts]),
                lo95=lo,
                hi95=hi
            ))
        return ForecastResponse(horizon=len(items), forecast=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando forecast: {e}")
