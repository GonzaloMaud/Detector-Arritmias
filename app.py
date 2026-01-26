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

# Expander con información técnica (Ideal para LinkedIn)
with st.expander("ℹ️ ¿Cómo funciona? (Información Técnica)"):
    st.markdown("""
    * **Entrada:** Señal ECG de un solo latido (Ventana de ~1.5s re-muestreada a 125Hz).
    * **Preprocesamiento:** Si el archivo no tiene exactamente 187 puntos, el sistema aplica *Zero-Padding* o recorte automáticamente.
    * **Modelo:** CNN construida con TensorFlow/Keras.
    * **Explicabilidad:** Uso de SHAP (Shapley Additive Explanations) para visualizar qué partes del latido activaron la neurona.
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
        
        # Conversión a numérico forzando errores a NaN (por si hay texto colado)
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        
        # Limpieza de nulos
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- LÓGICA DE AUTO-REPARACIÓN (ROBUSTEZ) ---
        TARGET_LENGTH = 187
        current_length = len(data_raw)
        
        if current_length == 0:
            st.error("El archivo parece estar vacío o no contiene números válidos.")
            st.stop()
            
        if current_length < TARGET_LENGTH:
            st.warning(f"⚠️ **Aviso de Preprocesamiento:** El latido tiene {current_length} puntos (se requieren {TARGET_LENGTH}). Se ha aplicado 'Zero-Padding' para completar la señal.")
            padding = np.zeros(TARGET_LENGTH - current_length)
            data = np.concatenate((data_raw, padding))
        elif current_length > TARGET_LENGTH:
            st.info(f"ℹ️ **Aviso de Preprocesamiento:** La señal excedía el tamaño estándar ({current_length}). Se ha recortado a los primeros {TARGET_LENGTH} puntos.")
            data = data_raw[:TARGET_LENGTH]
        else:
            data = data_raw
            st.success("✅ Longitud de señal correcta (187 puntos).")
            
        # Asegurar tipo de dato float32 para TensorFlow
        data = data.astype(np.float32)
        
        # Visualización Previa
        st.line_chart(data)
        
        # --- 4. INFERENCIA Y RESULTADOS ---
        if st.button("🔍 Ejecutar Diagnóstico"):
            
            # Reshape para entrar a la CNN: (Batch_Size, Time_Steps, Channels)
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Analizando patrones morfológicos...'):
                prediction = model.predict(data_reshaped)
                clase_predicha = np.argmax(prediction)
                probabilidad = np.max(prediction) * 100
                
                # Mapeo de clases según estándar AAMI / MIT-BIH
                clases = {
                    0: 'Normal (N) - Ritmo Sinusal', 
                    1: 'Arritmia Supraventricular (S)', 
                    2: 'Arritmia Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Latido Desconocido (Q)'
                }
                resultado = clases.get(clase_predicha, "Clase Desconocida")
            
            # --- VISUALIZACIÓN DE RESULTADOS ---
            st.markdown("---")
            st.subheader("🩺 Informe de IA")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if clase_predicha == 0:
                    st.success(f"## {resultado}")
                    st.caption("Morfología compatible con funcionamiento fisiológico estándar.")
                else:
                    st.error(f"## {resultado}")
                    st.caption("⚠️ Se han detectado anomalías en la morfología del complejo QRS o la onda P.")
            
            with col2:
                st.metric("Confianza del Modelo", f"{probabilidad:.2f}%")
            
            # --- 5. EXPLICABILIDAD (XAI) ---
            st.subheader("🧠 Interpretación del Modelo (SHAP)")
            st.write("El mapa de calor muestra en **rojo** las áreas que más influyeron en la decisión de la IA.")
            
            try:
                # Fondo base (latido plano) para comparar
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_reshaped)
                
                # Obtener valores para la clase predicha
                shap_val = shap_values[clase_predicha][0]
                
                # Gráfica personalizada con Matplotlib
                fig, ax = plt.subplots(figsize=(10, 3))
                # Dibujamos la línea gris de fondo
                ax.plot(data, color='gray', alpha=0.3, label='Señal ECG')
                # Superponemos los puntos de color según importancia
                sc = ax.scatter(range(187), data, c=shap_val.flatten(), cmap='coolwarm_r', s=15, alpha=0.9)
                
                plt.colorbar(sc, label='Impacto en Predicción')
                ax.set_title(f"Análisis de Morfología: {resultado}")
                ax.set_xlabel("Tiempo (muestras)")
                ax.set_ylabel("Amplitud Normalizada")
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"No se pudo generar la visualización SHAP. Detalle técnico: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo. Asegúrate de que sea un CSV válido. Detalle: {e}")