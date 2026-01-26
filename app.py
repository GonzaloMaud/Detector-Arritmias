import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

st.title("🫀 Detector de Arritmias con IA (Modelo Robusto)")
st.markdown("""
Esta versión utiliza un **modelo re-entrenado con Data Augmentation**. 
Es capaz de detectar arritmias incluso si el latido no está perfectamente centrado en la señal.
""")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # ¡OJO! Asegúrate de que este nombre coincide con el archivo que subas
    return tf.keras.models.load_model('modelo_todoterreno.keras')

try:
    with st.spinner('Cargando cerebro digital...'):
        model = load_model()
        st.success("✅ Modelo Robusto Activo")
except Exception as e:
    st.error(f"❌ Error cargando modelo: {e}")
    st.stop()

# --- 3. GESTIÓN DE ARCHIVOS ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Lectura básica
        df = pd.read_csv(uploaded_file, header=None)
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- PRE-PROCESAMIENTO MÍNIMO (Solo longitud) ---
        # Como el modelo es inteligente (Opción B), NO centramos el pico.
        # Solo nos aseguramos de que tenga el tamaño correcto (187) para no romper la matriz.
        TARGET_LEN = 187
        if len(data_raw) < TARGET_LEN:
            padding = np.zeros(TARGET_LEN - len(data_raw))
            data = np.concatenate((data_raw, padding))
        elif len(data_raw) > TARGET_LEN:
            data = data_raw[:TARGET_LEN]
        else:
            data = data_raw
            
        data = data.astype(np.float32)
        
        # Mostramos la señal tal cual viene (para demostrar que el modelo no necesita trampas)
        st.write("### Señal de Entrada")