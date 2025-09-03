# app/pages/1_Forecasting.py
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.title("📈 Forecasting · SARIMA")

# ----------------------------
# Descubrimiento robusto del root
# ----------------------------
THIS = Path(__file__).resolve()
CANDIDATES = [THIS.parents[i] for i in range(1, 6)]  # subir hasta 5 niveles

def find_repo_root():
    for base in CANDIDATES:
        sar_dir = base / "src" / "artifacts" / "sarima"
        pics_dir = base / "outputs" / "pics"
        raw_csv = base / "data" / "raw" / "dataset_demand_acumulate.csv"
        if sar_dir.exists() and pics_dir.exists():
            return base
        # si no existen pics, al menos valida sar_dir (artefactos) para seguir
        if sar_dir.exists():
            return base
    # fallback: el 3er parent (suele ser el root si app está en src/app/pages)
    return THIS.parents[3] if len(THIS.parents) >= 4 else THIS.parents[-1]

REPO_ROOT = find_repo_root()

ART_SAR = REPO_ROOT / "src" / "artifacts" / "sarima"
DATA_RAW = REPO_ROOT / "data" / "raw" / "dataset_demand_acumulate.csv"
PICS_DIR = REPO_ROOT / "outputs" / "pics"

# ----------------------------
# Sidebar (API + controles)
# ----------------------------
st.sidebar.subheader("Configuración")
api_base = st.sidebar.text_input("API base URL", value="http://localhost:8000")
n_steps = st.sidebar.slider("Horizonte (meses)", 1, 24, 3)

# ----------------------------
# Helpers
# ----------------------------
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_series(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    if "year_month" not in cols or "demand" not in cols: return None
    df = df.rename(columns={cols["year_month"]: "ds", cols["demand"]: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)

def call_forecast(api_url: str, steps: int) -> pd.DataFrame:
    r = requests.post(f"{api_url.rstrip('/')}/forecast", json={"n_steps": int(steps)}, timeout=30)
    r.raise_for_status()
    out = r.json().get("forecast", [])
    df = pd.DataFrame(out)
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def metrics_row(mets: dict | None):
    c1, c2, c3 = st.columns(3)
    if not mets:
        c1.metric("RMSE (valid.)", "–")
        c2.metric("MAE (valid.)",  "–")
        c3.metric("MAPE (valid.)", "–")
        return
    c1.metric("RMSE (valid.)", f'{mets.get("RMSE", 0):,.2f}')
    c2.metric("MAE (valid.)",  f'{mets.get("MAE", 0):,.2f}')
    c3.metric("MAPE (valid.)", f'{mets.get("MAPE", 0):,.2f}%')

def plot_series(history_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(history_df["ds"], history_df["y"], linewidth=2, label="Serie")
    ax.set_title("Serie histórica de demanda")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Demand"); ax.grid(alpha=0.25)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    st.pyplot(fig, clear_figure=True)

def plot_forecast(hist_df: pd.DataFrame | None, fc_df: pd.DataFrame, start_label: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    if hist_df is not None and not hist_df.empty:
        ax.plot(hist_df["ds"], hist_df["y"], color="#6b7280", linewidth=1.8, label="Histórico")
    ax.plot(fc_df["timestamp"], fc_df["forecast"], linewidth=2.2, color="#2563eb", label="Forecast")

    if {"lo95","hi95"}.issubset(fc_df.columns):
        fc_df = fc_df.sort_values("timestamp")
        ax.fill_between(fc_df["timestamp"], fc_df["lo95"], fc_df["hi95"], 
                        alpha=0.2, color="#93c5fd", label="IC 95%")

    ax.set_title(f"Pronóstico ({len(fc_df)} meses) · inicio: {start_label}")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Demand"); ax.grid(alpha=0.25)
    ax.legend()

    # ---- mejorar el eje X ----
    total_points = (len(hist_df) if hist_df is not None else 0) + len(fc_df)
    if total_points <= 36:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif total_points <= 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)

def show_saved_pics(dirpath: Path):
    pics = [
        "eda_1_serie.png","eda_2_media_movil_12m.png","eda_3_promedio_por_mes.png",
        "eda_4_boxplot_mes.png","acf.png","pacf.png","validacion_modelo.png","forecast_con_ic.png",
    ]
    existing = [p for p in pics if (dirpath / p).exists()]
    if not existing: return
    st.subheader("📎 Imágenes del análisis")
    cols = st.columns(2)
    for i, name in enumerate(existing):
        with cols[i % 2]:
            st.image(str(dirpath / name), caption=name, use_container_width=True)


# ----------------------------
# Config + métricas
# ----------------------------
cfg = load_json(ART_SAR / "config_sarima.json")
mets = load_json(ART_SAR / "backtest_metrics.json")

st.subheader("Configuración del modelo")
if cfg:
    st.json({
        "order": cfg.get("order"),
        "seasonal_order": cfg.get("seasonal_order"),
        "freq": cfg.get("freq"),
        "val_months": cfg.get("val_months"),
        "horizon_entrenamiento": cfg.get("horizon"),
    })
else:
    st.warning(f"No se encontró config en: {ART_SAR/'config_sarima.json'}")

st.subheader("Métricas de validación")
metrics_row(mets)
if not mets:
    st.info(f"No se encontró métricas en: {ART_SAR/'backtest_metrics.json'}")

st.divider()

# ----------------------------
# Serie histórica (contexto)
# ----------------------------
st.subheader("Serie histórica (contexto)")
hist_df = load_series(DATA_RAW)
if hist_df is None:
    st.warning(f"No se pudo cargar la serie desde {DATA_RAW}. Se omitirá el histórico.")
else:
    last_obs = hist_df["ds"].max()
    forecast_start = (last_obs + pd.offsets.DateOffset(months=1)).strftime("%Y-%m-01")
    st.caption(f"Última observación: **{last_obs.date()}** · El forecast comenzará en: **{forecast_start}**")
    plot_series(hist_df)

st.divider()

# ----------------------------
# Forecast interactivo
# ----------------------------
st.subheader("Pronóstico interactivo")
col_btn, col_hint = st.columns([1, 3])
do_fc = col_btn.button("Calcular forecast")
col_hint.caption(f"Llamando a **{api_base}/forecast** con `n_steps = {n_steps}`")

if do_fc:
    try:
        fc_df = call_forecast(api_base, n_steps)
        if fc_df.empty:
            st.warning("La API respondió sin datos de forecast.")
        else:
            st.dataframe(fc_df, use_container_width=True)
            st.download_button(
                "Descargar CSV del forecast",
                data=fc_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast.csv",
                mime="text/csv"
            )
            start_label = fc_df["timestamp"].min().strftime("%Y-%m-01")
            plot_forecast(hist_df, fc_df, start_label=start_label)
    except requests.RequestException as e:
        st.error(f"Error llamando a la API: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error procesando el forecast: {e}")

st.divider()

# ----------------------------
# Imágenes complementarias
# ----------------------------
show_saved_pics(PICS_DIR)

# Debug rápido
with st.expander("Debug (rutas y archivos quitar LUEGO! NO SE LE OLVIDE CAREMONDA!)"):
    st.write("REPO_ROOT:", REPO_ROOT)
    st.write("ART_SAR:", ART_SAR, "→", ART_SAR.exists())
    st.write("Config:", ART_SAR / "config_sarima.json", "→", (ART_SAR / "config_sarima.json").exists())
    st.write("Metrics:", ART_SAR / "backtest_metrics.json", "→", (ART_SAR / "backtest_metrics.json").exists())
    st.write("DATA_RAW:", DATA_RAW, "→", DATA_RAW.exists())
    st.write("PICS_DIR:", PICS_DIR, "→", PICS_DIR.exists())
