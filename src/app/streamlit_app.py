import streamlit as st

st.set_page_config(page_title="PT-Summa · Modelos", layout="wide")
st.title("PT-Summa · App de Modelos")

st.markdown("""
Bienvenido!

Esta aplicación muestra los modelos desarrollados en la prueba:

- **Forecasting (SARIMA)**: análisis de series temporales de demanda.
- **Clasificación (Logistic Regression)**: predicción Alpha/Betha.
- **Predicted CSV**: interactuar con los endpoints y visualizar resultados. Aca se realiza el proceso completo soo subiendo el csv y procesandolo para completar las predicciones pedidas (Demand y Class) 

Usa el menú lateral para navegar entre las secciones.
""")
