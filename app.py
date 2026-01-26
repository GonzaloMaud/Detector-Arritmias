import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

st.title("🫀 Detector de Arritmias con IA")
st.markdown("""
Esta herramienta utiliza una **Red Neuronal Convolucional (CNN)** entrenada con el dataset **MIT-BIH** para clasificar latidos cardíacos en 5 categorías clínicas.
""")

# Expander con información técnica
with st.expander("ℹ️ ¿Cómo funciona? (Información Técnica)"):
    st.markdown("""
    * **Entrada:** Señal ECG de un solo latido (Ventana de ~1.5s re-muestreada a 125Hz).
    * **Preprocesamiento:** Si el archivo no tiene exactamente 187 puntos, el sistema aplica *Zero-Padding* o recorte automáticamente.
    * **Modelo:** CNN construida con TensorFlow/Keras.
    * **Explicabilidad:** Uso de SHAP para visualizar la importancia de cada punto del latido.
    """)
st.markdown("---")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('modelo_ecg_final.keras')
    return model

try:
    with st.spinner('Inicializando motor de inferencia...'):
        model = load_model()
        st.success("✅ Sistema Inteligente Activo")
except Exception as e:
    st.error(f"❌ Error crítico cargando el modelo: {e}")
    st.stop()

# --- 3. GESTIÓN DE ARCHIVOS ---
st.subheader("📂 Análisis de Señal")
uploaded_file = st.file_uploader("Sube tu archivo CSV (Formato vector fila)", type="csv")

if uploaded_file is not None:
    try:
        # Lectura segura del CSV
        df = pd.read_csv(uploaded_file, header=None)
        
        # Conversión a numérico forzando errores a NaN
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        
        # Limpieza de nulos
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- LÓGICA DE AUTO-REPARACIÓN ---
        TARGET_LENGTH = 187
        current_length = len(data_raw)
        
        if current_length == 0:
            st.error("El archivo parece estar vacío o no contiene números válidos.")
            st.stop()
            
        if current_length < TARGET_LENGTH:
            st.warning(f"⚠️ **Aviso:** El latido tiene {current_length} puntos. Se ha rellenado con ceros hasta llegar a {TARGET_LENGTH}.")
            padding = np.zeros(TARGET_LENGTH - current_length)
            data = np.concatenate((data_raw, padding))
        elif current_length > TARGET_LENGTH:
            st.info(f"ℹ️ **Aviso:** La señal excedía el tamaño estándar. Se ha recortado a {TARGET_LENGTH} puntos.")
            data = data_raw[:TARGET_LENGTH]
        else:
            data = data_raw
            st.success("✅ Longitud de señal correcta (187 puntos).")
            
        # Asegurar tipo float32
        data = data.astype(np.float32)
        
        # Visualización Previa
        st.line_chart(data)
        
        # --- 4. INFERENCIA ---
        if st.button("🔍 Ejecutar Diagnóstico"):
            
            # Reshape para entrar a la CNN: (1, 187, 1)
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Analizando patrones morfológicos...'):
                prediction = model.predict(data_reshaped)
                clase_predicha = np.argmax(prediction)
                probabilidad = np.max(prediction) * 100
                
                # Mapeo de clases
                clases = {
                    0: 'Normal (N) - Ritmo Sinusal', 
                    1: 'Arritmia Supraventricular (S)', 
                    2: 'Arritmia Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Latido Desconocido (Q)'
                }
                resultado = clases.get(clase_predicha, "Clase Desconocida")
            
            # --- RESULTADOS ---
            st.markdown("---")
            st.subheader("🩺 Informe de IA")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if clase_predicha == 0:
                    st.success(f"## {resultado}")
                    st.caption("Morfología compatible con funcionamiento fisiológico estándar.")
                else:
                    st.error(f"## {resultado}")
                    st.caption("⚠️ Se han detectado anomalías morfológicas en el complejo QRS.")
            
            with col2:
                st.metric("Confianza del Modelo", f"{probabilidad:.2f}%")
            
            # --- 5. EXPLICABILIDAD (SHAP - CORREGIDO) ---
            st.subheader("🧠 Interpretación del Modelo (SHAP)")
            st.write("El mapa de calor muestra en **rojo** las áreas que activaron la detección.")
            
            try:
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_reshaped)
                
                # --- CORRECCIÓN INTELIGENTE DEL ERROR DE ÍNDICE ---
                # Detectamos si SHAP devuelve una lista (versiones antiguas) o un array único (versiones nuevas)
                if isinstance(shap_values, list):
                    if len(shap_values) > 1:
                        # Caso A: Una lista con 5 arrays (uno por clase). Usamos la clase predicha.
                        shap_val = shap_values[clase_predicha][0]
                    else:
                        # Caso B: Una lista con 1 solo array que contiene todo. Usamos el índice 0.
                        shap_val = shap_values[0][0]
                        # Si tiene 3 dimensiones (ej: 187, 1, 5), cogemos la capa de la clase predicha
                        if shap_val.ndim == 3 and shap_val.shape[-1] == 5:
                             shap_val = shap_val[:, :, clase_predicha]
                else:
                    # Caso C: No es lista, es un array directo.
                    shap_val = shap_values[0]
                
                # Aplanamos para asegurar que encaje en el gráfico (187 puntos planos)
                shap_val = shap_val.flatten()
                data_plot = data.flatten()

                # Gráfica
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(data_plot, color='gray', alpha=0.3, label='Señal ECG')
                sc = ax.scatter(range(len(data_plot)), data_plot, c=shap_val, cmap='coolwarm_r', s=15, alpha=0.9)
                
                plt.colorbar(sc, label='Importancia')
                ax.set_title(f"Impacto visual en: {resultado}")
                st.pyplot(fig)
                
            except Exception as e:
                # Si falla SHAP, no rompemos la app, solo avisamos
                st.warning(f"No se pudo generar el gráfico de interpretabilidad, pero el diagnóstico es correcto.")
                st.caption(f"Detalle técnico del error: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")