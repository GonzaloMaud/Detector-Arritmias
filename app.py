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
Esta herramienta utiliza una **Red Neuronal Convolucional (CNN)** entrenada con el dataset **MIT-BIH** para clasificar latidos cardíacos.
""")

with st.expander("ℹ️ Detalles Técnicos"):
    st.markdown("""
    * **Modelo:** CNN (Keras/TensorFlow).
    * **Entrada:** Señal ECG (187 puntos).
    * **Explicabilidad:** Algoritmo SHAP (DeepExplainer).
    """)
st.markdown("---")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('modelo_ecg_final.keras')
    return model

try:
    with st.spinner('Cargando sistema...'):
        model = load_model()
        st.success("✅ Modelo Activo")
except Exception as e:
    st.error(f"❌ Error crítico cargando el modelo: {e}")
    st.stop()

# --- 3. GESTIÓN DE ARCHIVOS ---
st.subheader("📂 Análisis de Señal")
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        data_raw = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        data_raw = data_raw[~np.isnan(data_raw)]
        
        # --- AUTO-REPARACIÓN ---
        TARGET = 187
        curr = len(data_raw)
        
        if curr == 0:
            st.error("Archivo vacío.")
            st.stop()
            
        if curr < TARGET:
            st.warning(f"⚠️ Rellenando señal (de {curr} a {TARGET} puntos).")
            padding = np.zeros(TARGET - curr)
            data = np.concatenate((data_raw, padding))
        elif curr > TARGET:
            st.info(f"ℹ️ Recortando señal (de {curr} a {TARGET} puntos).")
            data = data_raw[:TARGET]
        else:
            data = data_raw
            st.success("✅ Señal correcta.")
            
        data = data.astype(np.float32)
        st.line_chart(data)
        
        # --- 4. INFERENCIA ---
        if st.button("🔍 Diagnosticar"):
            
            # Reshape (1, 187, 1)
            data_in = data.reshape(1, 187, 1)
            
            with st.spinner('Analizando...'):
                prediction = model.predict(data_in)
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
            
            # --- RESULTADOS ---
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if clase_predicha == 0:
                    st.success(f"## {resultado}")
                else:
                    st.error(f"## {resultado}")
            with c2:
                st.metric("Confianza", f"{probabilidad:.2f}%")
            
            # --- 5. EXPLICABILIDAD (SHAP BLINDADO) ---
            st.subheader("🧠 Mapa de Calor (SHAP)")
            
            try:
                # 1. Calculamos SHAP
                background = np.zeros((1, 187, 1))
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data_in)
                
                # 2. Normalización de formatos (Lista vs Array)
                if isinstance(shap_values, list):
                    # Si devuelve lista de 5 arrays, cogemos el de la clase predicha
                    shap_val = shap_values[clase_predicha]
                else:
                    # Si devuelve un array único
                    shap_val = shap_values

                # 3. CORRECCIÓN MATEMÁTICA (El error 935)
                # Convertimos a array numpy plano primero
                shap_val = np.array(shap_val)
                
                # Si el tamaño es gigante (187 * 5 = 935), significa que están todas las clases pegadas
                if shap_val.size == 935:
                    # Lo reordenamos a (187 filas, 5 columnas)
                    shap_val = shap_val.reshape(187, 5)
                    # Y nos quedamos solo con la columna de la clase ganadora
                    shap_val = shap_val[:, clase_predicha]
                
                # Aplanamos a 187 para asegurar
                shap_val = shap_val.flatten()
                
                # Si por algún motivo sigue sin ser 187, cortamos o rellenamos (seguridad extrema)
                if shap_val.shape[0] > 187:
                    shap_val = shap_val[:187]
                
                # 4. Gráfico
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(data, color='gray', alpha=0.3)
                sc = ax.scatter(range(187), data, c=shap_val, cmap='coolwarm_r', s=15)
                plt.colorbar(sc, label='Importancia')
                ax.set_title(f"Impacto visual para: {resultado}")
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Diagnóstico correcto, pero no se pudo generar el gráfico SHAP.")
                st.caption(f"Error técnico: {e}")

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")