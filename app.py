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
    # Asegúrate de que este nombre es el correcto
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
        
        # Convertimos a números y limpiamos errores
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- PRE-PROCESAMIENTO MÍNIMO ---
        TARGET_LEN = 187
        if len(data_raw) < TARGET_LEN:
            padding = np.zeros(TARGET_LEN - len(data_raw))
            data = np.concatenate((data_raw, padding))
        elif len(data_raw) > TARGET_LEN:
            data = data_raw[:TARGET_LEN]
        else:
            data = data_raw
            
        data = data.astype(np.float32)
        
        # Mostramos la señal
        st.write("### Señal de Entrada")
        st.line_chart(data)
        
        # --- 4. DIAGNÓSTICO ---
        if st.button("🔍 Analizar Latido"):
            
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Procesando...'):
                prediction = model.predict(data_reshaped)
                clase_val = np.argmax(prediction)
                confianza = np.max(prediction) * 100
                
                clases = {
                    0: 'Normal (N)', 
                    1: 'Supraventricular (S)', 
                    2: 'Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Desconocido (Q)'
                }
                resultado = clases.get(clase_val, "Error")
            
            # Resultados
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Diagnóstico", resultado)
            c2.metric("Confianza", f"{confianza:.1f}%")
            
            if clase_val != 0:
                st.error(f"⚠️ Detección: {resultado}")
            else:
                st.success(f"✅ {resultado}")

            # --- 5. EXPLICABILIDAD (SHAP) ---
            try:
                st.subheader("Por qué la IA dice esto (SHAP)")
                explainer = shap.DeepExplainer(model, np.zeros((1, 187, 1)))
                shap_values = explainer.shap_values(data_reshaped)
                
                # Gestión de formatos SHAP
                if isinstance(shap_values, list):
                    shap_val = shap_values[clase_val]
                else:
                    shap_val = shap_values
                
                shap_val = np.array(shap_val)
                
                # Corrección dimensiones
                if shap_val.size == (187 * 5): 
                    shap_val = shap_val.reshape(187, 5)[:, clase_val]
                
                shap_val = shap_val.flatten()[:187]

                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(data, color='gray', alpha=0.3)
                sc = ax.scatter(range(187), data, c=shap_val, cmap='coolwarm_r', s=15)
                plt.colorbar(sc, label='Importancia')
                ax.set_title(f"Zonas críticas para: {resultado}")
                st.pyplot(fig)
            
            except Exception as e:
                st.warning("Diagnóstico completado, pero no se pudo generar el gráfico SHAP.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")