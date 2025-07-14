# app_streamlit.py

import streamlit as st
import joblib
import numpy as np

# Cargar modelo y scaler
modelo = joblib.load("modelo_guardado.pkl")
scaler = joblib.load("scaler_guardado.pkl")

st.title("Predicción de Aprobación Estudiantil")

# Formulario
# Formulario
horas_estudio = st.number_input(
    "Horas de estudio",
    min_value=0.0,
    max_value=20.0,  
    step=0.5
)

clases_asistidas = st.number_input(
    "Clases asistidas",
    min_value=0.0,
    max_value=14.0,  
    step=1.0
)

tareas_entregadas = st.number_input(
    "Tareas entregadas",
    min_value=0.0,
    max_value=9.0,    
    step=1.0
)


if st.button("Predecir"):
    entrada = np.array([[horas_estudio, clases_asistidas, tareas_entregadas]])
    entrada_normalizada = scaler.transform(entrada)
    prediccion = modelo.predict(entrada_normalizada)
    
    if prediccion[0] == 1:
        st.success(" El estudiante probablemente APROBARÁ.")
    else:
        st.error(" El estudiante probablemente NO aprobará.")
