# app/pages/3_Predicted_CSV.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.title("📄 Predicted CSV")

# ---------------------------
# Descubrir root (rutas robustas)
# ---------------------------
THIS = Path(__file__).resolve()
CANDIDATES = [THIS.parents[i] for i in range(1, 6)]
def find_repo_root():
    for base in CANDIDATES:
        if (base / "src" / "artifacts").exists():
            return base
    return THIS.parents[3] if len(THIS.parents) >= 4 else THIS.parents[-1]
REPO_ROOT = find_repo_root()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.subheader("Configuración")
api_base = st.sidebar.text_input("API base URL", value="http://localhost:8000")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def get_schema(api_base_url: str) -> Dict[str, Any] | None:
    try:
        r = requests.get(f"{api_base_url.rstrip('/')}/predict/schema", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def call_forecast(api_base_url: str, steps: int) -> pd.DataFrame:
    r = requests.post(f"{api_base_url.rstrip('/')}/forecast", json={"n_steps": int(steps)}, timeout=30)
    r.raise_for_status()
    out = r.json().get("forecast", [])
    df = pd.DataFrame(out)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def call_predict_batch(api_base_url: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r = requests.post(f"{api_base_url.rstrip('/')}/predict", json={"data": rows}, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def keep_model_columns(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    cols = [c for c in expected if c in df.columns]
    return df[cols]

def sanitize_for_json(df: pd.DataFrame, num_features: List[str], cat_features: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in num_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype(object).where(pd.notnull(df[c]), None)
    df = df.replace({np.inf: None, -np.inf: None})
    df = df.fillna({c: 0 for c in num_features})
    df = df.fillna({c: "NA" for c in cat_features})
    return df

# ---------------------------
# UI
# ---------------------------
st.markdown("""
Sube el CSV **original** (p. ej. `to_predict.csv`).  
La página hará:
1) **Forecast SARIMA** → completa la columna **`Demand`** (y agrega `ForecastTimestamp`).  
2) **Clasificación Alpha/Betha** → completa la columna **`Class`**.  
Finalmente podrás **descargar el CSV completado**.
""")

schema = get_schema(api_base)
if not schema:
    st.error("No pude obtener el esquema desde /predict/schema. ¿La API está levantada?")
else:
    num_features: List[str] = schema.get("num_features", []) or []
    cat_features: List[str] = schema.get("cat_features", []) or []
    expected_features: List[str] = schema.get("expected_features", []) or []

    file = st.file_uploader("Selecciona el CSV a completar", type=["csv"])
    if file is not None:
        try:
            # Leer y normalizar columnas
            df_in = pd.read_csv(file)
            df_norm = normalize_cols(df_in)

            st.write("Vista previa del archivo (primeras 5):")
            st.dataframe(df_norm.head(), use_container_width=True)

            if st.button("Generar CSV con predicciones"):
                n = len(df_norm)
                if n == 0:
                    st.warning("El archivo está vacío.")
                else:
                    # Forecast
                    fc_df = call_forecast(api_base, n)
                    if fc_df.empty:
                        st.error("La API devolvió forecast vacío.")
                        st.stop()
                    demand_vals = fc_df["forecast"].tolist()
                    ts_vals = fc_df["timestamp"].astype(str).tolist()

                    df_out = df_in.copy()
                    if "Demand" not in df_out.columns:
                        df_out["Demand"] = np.nan
                    if "Class" not in df_out.columns:
                        df_out["Class"] = None
                    df_out["ForecastTimestamp"] = ts_vals[:n]
                    df_out.loc[:n-1, "Demand"] = demand_vals[:n]

                    # Clasificación (batch)
                    model_df = keep_model_columns(normalize_cols(df_out), expected_features)
                    model_df = sanitize_for_json(model_df, num_features, cat_features)
                    rows = model_df.to_dict(orient="records")
                    r = call_predict_batch(api_base, rows)
                    preds = r.get("predictions", [])

                    pred_df = pd.DataFrame(preds)
                    df_out.loc[:len(pred_df)-1, "Class"] = pred_df["predicted_label"].values
                    df_out["proba_betha"] = pd.NA
                    df_out["threshold_used"] = pd.NA
                    df_out.loc[:len(pred_df)-1, "proba_betha"] = pred_df["proba_betha"].values
                    df_out.loc[:len(pred_df)-1, "threshold_used"] = pred_df["threshold_used"].values

                    # Mostrar y descargar
                    st.success("¡Listo! Archivo completado.")
                    counts = pd.Series(df_out["Class"]).value_counts(dropna=False).rename_axis("Class").reset_index(name="count")
                    st.markdown("**Resumen por clase**")
                    st.dataframe(counts, use_container_width=True)

                    st.markdown("**Vista previa del archivo final**")
                    st.dataframe(df_out.head(10), use_container_width=True)
                    st.download_button(
                        "Descargar CSV completado",
                        data=df_out.to_csv(index=False).encode("utf-8"),
                        file_name="to_predict_completed.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"No pude procesar el CSV: {e}")
