# app/pages/2_Clasificacion.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.title("🔍 Clasificación · Alpha vs Betha")

# ---------------------------------------
# Descubrir root del repo (rutas robustas)
# ---------------------------------------
THIS = Path(__file__).resolve()
CANDIDATES = [THIS.parents[i] for i in range(1, 6)]

def find_repo_root():
    for base in CANDIDATES:
        if (base / "src" / "artifacts" / "classifier").exists():
            return base
    return THIS.parents[3] if len(THIS.parents) >= 4 else THIS.parents[-1]

REPO_ROOT = find_repo_root()
ART_CLS = REPO_ROOT / "src" / "artifacts" / "classifier"
EVAL_JSON = ART_CLS / "eval_report.json"
PICS_DIR = REPO_ROOT / "outputs" / "pics"

# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.subheader("Configuración")
api_base = st.sidebar.text_input("API base URL", value="http://localhost:8000")

# ---------------------------------------
# Helpers
# ---------------------------------------
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_schema(api_base_url: str) -> Dict[str, Any] | None:
    try:
        r = requests.get(f"{api_base_url.rstrip('/')}/predict/schema", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def call_predict_batch(api_base_url: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r = requests.post(f"{api_base_url.rstrip('/')}/predict", json={"data": rows}, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def keep_model_columns(df: pd.DataFrame, expected: List[str], id_col: Optional[str]) -> pd.DataFrame:
    cols = [c for c in expected if c in df.columns]
    if id_col and id_col in df.columns:
        return df[[id_col] + cols]
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

def find_id_col(cols: List[str]) -> Optional[str]:
    candidates = ["autoid", "id", "row_id"]
    for c in candidates:
        if c in cols:
            return c
    return None

# ---------------------------------------
# Métricas del modelo + explicación
# ---------------------------------------
st.subheader("Métricas del modelo (test)")
eval_rep = load_json(EVAL_JSON)
if not eval_rep:
    st.warning(f"No se encontró {EVAL_JSON}")
else:
    mets = eval_rep.get("metrics_test_at_threshold", {})
    thr = eval_rep.get("threshold", 0.5)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f'{mets.get("roc_auc", 0):.3f}')
    c2.metric("Balanced Acc.", f'{mets.get("balanced_accuracy", 0):.3f}')
    c3.metric("F1 (positiva)", f'{mets.get("f1_positive", 0):.3f}')
    c4.metric("Accuracy", f'{mets.get("accuracy", 0):.3f}')
    st.caption(f"Umbral usado en test: **{thr:.2f}** · Clase positiva: **Betha**")

with st.expander("¿Cómo leer estas métricas?"):
    st.markdown("""
- **ROC AUC**: cuán bien separa el modelo clases *Alpha* vs *Betha* para todos los umbrales (1.0 es perfecto).
- **Balanced Accuracy**: promedio del recall por clase (útil con clases desbalanceadas).
- **F1 (positiva)**: equilibrio entre precisión y recall para la clase **Betha**.
- **Accuracy**: porcentaje total de aciertos (puede sesgarse si las clases están desbalanceadas).
- **Umbral**: probabilidad mínima para etiquetar como **Betha**.
    """)

# Confusion matrix guardada (si existe)
cm_path = PICS_DIR / "clf_confusion_matrix.png"
if cm_path.exists():
    st.image(str(cm_path), caption="Matriz de confusión (test)", use_container_width=True)

st.divider()

# ---------------------------------------
# CSV (único flujo de predicción en esta página)
# ---------------------------------------
st.subheader("Subir CSV y predecir")
st.caption("Sube un CSV cualquiera; tomaremos **solo** las columnas del modelo, normalizaremos nombres y enviaremos a la API.")

schema = get_schema(api_base)
if not schema:
    st.error("No pude obtener el esquema desde /predict/schema. ¿La API está levantada?")
else:
    num_features: List[str] = schema.get("num_features", []) or []
    cat_features: List[str] = schema.get("cat_features", []) or []
    expected_features: List[str] = schema.get("expected_features", []) or []

    file = st.file_uploader("Selecciona un CSV", type=["csv"])
    if file is not None:
        try:
            df_up = pd.read_csv(file)
            df_norm = normalize_cols(df_up)
            id_col = find_id_col(df_norm.columns.tolist())

            df_model = keep_model_columns(df_norm, expected_features, id_col)
            df_model_clean = sanitize_for_json(df_model.copy(), num_features, cat_features)

            # Preview (features del modelo + id si viene)
            st.write("Vista previa (primeras 5):")
            st.dataframe(df_model_clean.head(), use_container_width=True)

            if st.button("Predecir lote"):
                # Enviar SIN la columna id
                rows = df_model_clean.drop(columns=[id_col], errors="ignore").to_dict(orient="records")
                resp = call_predict_batch(api_base, rows)
                preds = resp.get("predictions", [])
                if not preds:
                    st.warning("La API no devolvió predicciones.")
                else:
                    pred_df = pd.DataFrame(preds)  # predicted_label, proba_betha, threshold_used
                    # Adjuntar id si lo teníamos
                    if id_col and id_col in df_model_clean.columns:
                        pred_df.insert(0, id_col, df_model_clean[id_col].reset_index(drop=True))
                    else:
                        pred_df.insert(0, "row_id", range(len(pred_df)))

                    # Ordenar por clase para que sea fácil identificar
                    pred_df = pred_df.sort_values(["predicted_label", pred_df.columns[0]]).reset_index(drop=True)

                    # Resumen por clase
                    counts = pred_df["predicted_label"].value_counts().rename_axis("class").reset_index(name="count")
                    st.markdown("**Resumen de predicciones por clase**")
                    st.dataframe(counts, use_container_width=True)

                    st.markdown("**Predicciones**")
                    st.dataframe(pred_df, use_container_width=True)

                    st.download_button(
                        "Descargar predicciones CSV",
                        data=pred_df.to_csv(index=False).encode("utf-8"),
                        file_name="predicciones_classifier.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"No pude procesar el CSV: {e}")

# ---------------------------------------
# Debug rápido de rutas
# ---------------------------------------
with st.expander("🔧 Debug (rutas)"):
    st.write("REPO_ROOT:", REPO_ROOT)
    st.write("ART_CLS:", ART_CLS, "→", ART_CLS.exists())
    st.write("EVAL_JSON:", EVAL_JSON, "→", EVAL_JSON.exists())
    st.write("PICS_DIR:", PICS_DIR, "→", PICS_DIR.exists())
