import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

st.title("🫀 Detector de Arritmias con IA")
st.markdown("""
Esta aplicación utiliza **Deep Learning** para analizar latidos del corazón.
Sube un archivo CSV con la señal del electrocardiograma (ECG).
""")
st.markdown("---")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # Intenta cargar el modelo
    model = tf.keras.models.load_model('modelo_ecg_final.keras')
    return model

# Mensaje de estado
with st.spinner('Cargando cerebro digital...'):
    try:
        model = load_model()
        st.success("✅ Sistema Inteligente Activo")
    except Exception as e:
        # AQUÍ ES DONDE VEREMOS EL ERROR REAL
        st.error(f"❌ Error crítico cargando el modelo: {e}")
        st.stop()

# --- 3. INTERFAZ DE SUBIDA ---
st.subheader("📂 Paso 1: Sube el Electrocardiograma")
uploaded_file = st.file_uploader("Arrastra tu archivo CSV aquí", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        
        # Tomamos la primera fila como ejemplo
        data = df.iloc[0, :187].values
        data = data.astype(np.float32)
        
        st.write("✅ Señal recibida correctamente.")
        st.line_chart(data)
        
        # --- 4. PREDICCIÓN ---
        if st.button("🔍 Analizar Latido"):
            
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Analizando morfología del latido...'):
                prediction = model.predict(data_reshaped)
                clase_predicha = np.argmax(prediction)
                probabilidad = np.max(prediction) * 100
                
                nombres_clases = {
                    0: 'Normal', 
                    1: 'Arritmia Supraventricular (S)', 
                    2: 'Arritmia Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Latido Desconocido (Q)'
                }
                resultado = nombres_clases.get(clase_predicha, "Desconocido")

            st.markdown("---")
            st.header("🩺 Diagnóstico Clínico")
            
            col1, col2 = st.columns(2)
            with col1:
                if clase_predicha == 0:
                    st.success(f"### {resultado}")
                else:
                    st.error(f"### {resultado}")
            with col2:
                st.metric(label="Confianza", value=f"{probabilidad:.1f}%")

            # --- 5. EXPLICABILIDAD (SHAP) ---
            st.subheader("🧠 Análisis de Caja Blanca")
            try:
                # Fondo para SHAP
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_reshaped)
                
                shap_val = shap_values[clase_predicha][0]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data.flatten(), color='gray', alpha=0.3)
                sc = ax.scatter(range(187), data.flatten(), c=shap_val.flatten(), cmap='coolwarm_r')
                plt.colorbar(sc, label='Importancia')
                ax.set_title(f"Explicación de {resultado}")
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"No se pudo generar el gráfico SHAP: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")