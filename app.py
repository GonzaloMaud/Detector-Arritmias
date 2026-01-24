import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

st.title("🫀 Detector de Arritmias con IA")
st.markdown("Sube tu ECG. Si el archivo es corto, el sistema lo rellenará automáticamente.")
st.markdown("---")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('modelo_ecg_final.keras')
    return model

try:
    with st.spinner('Cargando cerebro digital...'):
        model = load_model()
        st.success("✅ Modelo cargado y listo")
except Exception as e:
    st.error(f"❌ Error cargando modelo: {e}")
    st.stop()

# --- 3. PROCESAMIENTO INTELIGENTE ---
st.subheader("📂 Sube tu archivo CSV")
uploaded_file = st.file_uploader("Archivo CSV (MIT-BIH)", type="csv")

if uploaded_file is not None:
    try:
        # Leemos el archivo sin cabecera
        df = pd.read_csv(uploaded_file, header=None)
        
        # Cogemos la primera fila y la convertimos a números
        # 'coerce' transforma textos raros en NaN (vacío) para que no falle
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        
        # Eliminamos posibles valores vacíos (NaN)
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- SECCIÓN DE AUTO-REPARACIÓN ---
        target_len = 187
        current_len = len(data_raw)
        
        if current_len < target_len:
            # Si faltan números, rellenamos con ceros al final
            st.warning(f"⚠️ El archivo tiene {current_len} datos. Rellenando con ceros hasta 187...")
            padding = np.zeros(target_len - current_len)
            data = np.concatenate((data_raw, padding))
        elif current_len > target_len:
            # Si sobran, cortamos
            st.info(f"ℹ️ Recortando archivo de {current_len} a 187 puntos.")
            data = data_raw[:target_len]
        else:
            data = data_raw
            
        data = data.astype(np.float32)
        # -----------------------------------
        
        st.line_chart(data)
        
        if st.button("🔍 Analizar Latido"):
            # Reshape para el modelo (1 muestra, 187 tiempos, 1 canal)
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Procesando...'):
                prediction = model.predict(data_reshaped)
                clase_predicha = np.argmax(prediction)
                probabilidad = np.max(prediction) * 100
                
                clases = {
                    0: 'Normal (N)', 
                    1: 'Supraventricular (S)', 
                    2: 'Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Desconocido (Q)'
                }
                resultado = clases.get(clase_predicha, "Error")
            
            # Resultados
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if clase_predicha == 0:
                    st.success(f"### {resultado}")
                else:
                    st.error(f"### {resultado}")
            with col2:
                st.metric("Confianza", f"{probabilidad:.1f}%")
            
            # SHAP
            st.subheader("🧠 Por qué la IA dice esto:")
            try:
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_reshaped)
                shap_val = shap_values[clase_predicha][0]
                
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(data, color='gray', alpha=0.3)
                sc = ax.scatter(range(187), data, c=shap_val.flatten(), cmap='coolwarm_r', s=10)
                plt.colorbar(sc, label='Importancia')
                ax.set_title(f"Mapa de Calor: {resultado}")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"No se pudo cargar gráfico SHAP: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")