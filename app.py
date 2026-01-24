import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

# Títulos y Estilo
st.title("🫀 Detector de Arritmias con IA")
st.markdown("""
Esta aplicación utiliza **Deep Learning (CNN)** para analizar latidos del corazón y detectar anomalías.
Sube un archivo CSV con la señal del electrocardiograma (ECG).
""")
st.markdown("---")

# --- 2. CARGA DEL MODELO ---
# Usamos @st.cache_resource para que cargue solo una vez y vaya rápido
@st.cache_resource
def load_model():
    # Asegúrate de que el archivo .keras esté en la misma carpeta
    model = tf.keras.models.load_model('modelo_ecg_final.keras')
    return model

# Mensaje de estado
with st.spinner('Cargando cerebro digital...'):
    try:
        model = load_model()
        st.success("✅ Sistema Inteligente Activo")
    except Exception as e:
        st.error(f"❌ Error crítico: No se encuentra el archivo 'modelo_ecg_final.keras'. Asegúrate de subirlo a GitHub junto con este script.")
        st.stop() # Detiene la app si no hay modelo

# --- 3. INTERFAZ DE SUBIDA ---
st.subheader("📂 Paso 1: Sube el Electrocardiograma")
uploaded_file = st.file_uploader("Arrastra tu archivo CSV aquí (Formato MIT-BIH)", type="csv")

if uploaded_file is not None:
    # Procesar archivo
    try:
        df = pd.read_csv(uploaded_file, header=None)
        
        # Tomamos la primera fila como ejemplo (simulando un latido)
        # Nos aseguramos de coger solo los primeros 187 puntos (sin la etiqueta si la tuviera)
        data = df.iloc[0, :187].values
        data = data.astype(np.float32)
        
        st.write("✅ Señal recibida correctamente.")
        
        # Mostrar gráfica simple del latido
        st.line_chart(data)
        
        # --- 4. PREDICCIÓN ---
        if st.button("🔍 Analizar Latido"):
            
            # Preparar datos para la red neuronal (1, 187, 1)
            data_reshaped = data.reshape(1, 187, 1)
            
            with st.spinner('Analizando morfología del latido...'):
                # Predicción
                prediction = model.predict(data_reshaped)
                clase_predicha = np.argmax(prediction)
                probabilidad = np.max(prediction) * 100
                
                # Diccionario de diagnósticos
                nombres_clases = {
                    0: 'Normal', 
                    1: 'Arritmia Supraventricular (S)', 
                    2: 'Arritmia Ventricular (V)', 
                    3: 'Fusión (F)', 
                    4: 'Latido Desconocido (Q)'
                }
                resultado = nombres_clases.get(clase_predicha, "Desconocido")

            # Mostrar Resultados Bonitos
            st.markdown("---")
            st.header("🩺 Diagnóstico Clínico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if clase_predicha == 0:
                    st.success(f"### {resultado}")
                    st.caption("El latido presenta una morfología estándar.")
                else:
                    st.error(f"### {resultado}")
                    st.caption("⚠️ Se recomienda revisión por un especialista.")
            
            with col2:
                st.metric(label="Confianza del Modelo", value=f"{probabilidad:.1f}%")

            # --- 5. EXPLICABILIDAD (XAI) ---
            st.subheader("🧠 Análisis de Caja Blanca (Explainable AI)")
            st.write("El modelo destaca en **rojo** las partes del latido que le parecieron sospechosas.")
            
            try:
                # Configuración para SHAP
                # Usamos un fondo de ceros para comparar (línea base)
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_reshaped)
                
                # Extraemos los valores para la clase que ha predicho
                shap_val = shap_values[clase_predicha][0]
                
                # Aplanamos para graficar (evita errores de dimensiones)
                signal_flat = data.flatten()
                shap_flat = shap_val.flatten()
                
                # Normalización de colores
                shap_min, shap_max = np.min(shap_flat), np.max(shap_flat)
                
                # Crear figura Matplotlib
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(signal_flat, color='gray', alpha=0.3, label='Señal ECG')
                
                # Puntos coloreados
                sc = ax.scatter(range(len(signal_flat)), signal_flat, 
                                c=shap_flat, cmap='coolwarm_r', # coolwarm_r invierte (rojo=alto) si es necesario
                                vmin=shap_min, vmax=shap_max, s=15)
                
                plt.colorbar(sc, label='Importancia para la IA')
                ax.set_title(f"Mapa de Calor: ¿Por qué es {resultado}?")
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Amplitud")
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"No se pudo generar la explicación visual (SHAP). Detalle: {e}")

    except Exception as e:
        st.error(f"Error procesando el archivo. Asegúrate de que es un CSV válido con números. Detalle: {e}")