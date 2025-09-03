# Prueba Técnica – Forecasting & Clasificación

Este proyecto implementa un **pipeline completo de análisis y despliegue de modelos** que incluye:

1. **Forecasting (SARIMA)**: entrenamiento y validación de un modelo de series de tiempo.
2. **Clasificación (Random Forest / Logistic Regression)**: entrenamiento y evaluación de un modelo supervisado.
3. **API (FastAPI)**: exposición de endpoints REST para consumir los modelos entrenados.
4. **App (Streamlit)**: interfaz visual que resume los resultados del EDA, muestra las predicciones y permite interactuar con los modelos.

---

## Estructura del proyecto

```bash
proyecto_modelos/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ training/
│  │  ├─ train_sarima.py                 # entrena y guarda SARIMA
│  │  ├─ train_classifier.py             # entrena y guarda LogReg
│  │  └─ utils_metrics.py                # métricas simples (rmse, mae, auc, etc.)
│  ├─ artifacts/
│  │  ├─ sarima/
│  │  │  ├─ model_sarima.joblib
│  │  │  ├─ config_sarima.json           # ordenes (p,d,q)(P,D,Q)m, m, etc.
│  │  │  └─ backtest_metrics.json        # rmse/mae/mape en validación
│  │  └─ classifier/
│  │     ├─ model_best_pipeline.joblib
│  │     ├─ model_threshold.txt          # 0.56184…
│  │     └─ eval_report.json             # auc, balanced_acc, f1, etc.
│  ├─ api/
│  │  ├─ main.py                         # FastAPI: /health, /forecast, /predict
│  │  └─ schemas.py                      # pydantic models de entrada/salida
│  └─ app/
│     └─ streamlit_app.py                # 3 pestañas (SARIMA, Clasificador, API client)
└─ data/                                  # opcional: muestras/fixtures
   └─ sample_payloads/
      ├─ predict_one.json
      ├─ predict_batch.json
      └─ forecast_request.json

```

---

## Flujo de trabajo

1. **Preparación de datos**
   - Colocar los datasets en `raw/` (ej. `serie.csv` para forecasting y `train.csv` para clasificación).

2. **Entrenamiento**
   - `python src/forecasting.py` → entrena un modelo SARIMA y guarda artefactos en `outputs/models/SARIMA_...`.
   - `python src/classifier.py` → entrena un clasificador (Random Forest o Logistic Regression) y guarda artefactos en `outputs/models/classifier_random_forest`.

3. **API**
   - `api/main.py` levanta un servicio con **FastAPI** que expone:
     - `/health` → estado del servicio.
     - `/predict_forecast` → genera predicciones de serie temporal.
     - `/predict_class` → genera predicciones de clasificación.
   - Se ejecuta con:
     ```bash
     uvicorn api.main:app --reload --port 8000
     ```

4. **App (Streamlit)**
   - La aplicación en `app/` permite visualizar resultados y consultar la API.
   - Páginas:
     - **Forecasting**: gráficos de EDA, ACF/PACF y resultados del modelo.
     - **Clasificación**: distribución de variables, métricas de validación y predicciones.
     - **Resumen**: KPIs globales y enlaces a los artefactos más recientes.
   - Se ejecuta con:
     ```bash
     streamlit run app/streamlit_app.py
     ```

---

##  Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

---

## Uso de la aplicación

Una vez entrenados los modelos y generados los artefactos en `src/artifacts/`, puedes interactuar con ellos a través de la **API** y la **App en Streamlit**.

1. **Levantar la API (FastAPI)**  
   Desde la raíz del proyecto, ejecuta:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Esto deja la API corriendo en http://localhost:8000/docs ,donde podrás explorar los endpoints:

GET /health → Verifica el estado y artefactos cargados.

POST /forecast → Genera un pronóstico SARIMA.

POST /predict → Clasifica registros como Alpha o Betha.

 2. **Levantar la App (Streamlit)**
   En otra terminal, ejecuta:

```bash
streamlit run src/app/streamlit_app.py
```

Esto abrirá la interfaz visual con tres pestañas:

- **Forecasting** → Explora el modelo SARIMA, métricas y pronósticos.  
- **Clasificación** → Consulta métricas de validación y realiza predicciones desde un CSV.  
- **Predicted CSV** → Sube el archivo `to_predict.csv` y genera automáticamente un nuevo CSV con las columnas `Demand` y `Class` completadas por los modelos.

### Flujo típico

1. Asegúrate de tener la **API levantada primero**.  
2. Luego abre la app en [http://localhost:8501](http://localhost:8501) y navega por las pestañas.  
3. En la pestaña **Predicted CSV**, sube el archivo original (`to_predict.csv`) y descarga el archivo enriquecido con las predicciones.  
