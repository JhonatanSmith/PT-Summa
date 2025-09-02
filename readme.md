# Prueba Técnica – Forecasting & Clasificación

Este proyecto implementa un **pipeline completo de análisis y despliegue de modelos** que incluye:

1. **Forecasting (SARIMA)**: entrenamiento y validación de un modelo de series de tiempo.
2. **Clasificación (Random Forest / Logistic Regression)**: entrenamiento y evaluación de un modelo supervisado.
3. **API (FastAPI)**: exposición de endpoints REST para consumir los modelos entrenados.
4. **App (Streamlit)**: interfaz visual que resume los resultados del EDA, muestra las predicciones y permite interactuar con los modelos.

---

## Estructura del proyecto

```bash
root/
  raw/                            # datasets de entrada (series y clasificación)
  outputs/
    pics/                         # imágenes (EDA, ACF/PACF, validación, forecast, clf)
    models/
      SARIMA_YYYYMMDD_HHMM/       # un run de forecasting (timestamp)
        model.pkl                 # modelo SARIMA entrenado
        forecast.csv              # predicciones generadas
        metadata.json             # parámetros y métricas del run
      classifier_random_forest/   # último run de clasificación
        model.pkl                 # modelo de clasificación entrenado
        metadata.json             # features, métricas y versión
  src/
    forecasting.py                # entrena SARIMA y guarda artefactos en outputs/models
    classifier.py                 # entrena clasificador (RF o LR) y guarda artefactos
  api/
    main.py                       # FastAPI: /health, /predict_forecast, /predict_class
  app/
    streamlit_app.py              # Home / router de páginas
    pages/
      1_Forecasting.py            # UI: carga imágenes y consume /predict_forecast
      2_Clasificacion.py          # UI: EDA/métricas y consume /predict_class
      3_Resumen.py                # KPIs globales, links a artefactos y últimas métricas
  requirements.txt                # dependencias del proyecto

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