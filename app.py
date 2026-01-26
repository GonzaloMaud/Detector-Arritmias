import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente de Arritmias", page_icon="🫀", layout="centered")

st.title("🫀 Detector de Arritmias con IA")
st.markdown("Sube un latido (CSV). El sistema lo centrará y analizará automáticamente.")

# --- 2. FUNCIÓN DE "AUTO-CENTRADO" (LA MAGIA) ---
def preprocess_ecg(data_raw):
    """
    Toma un latido crudo, encuentra el pico R y lo centra en la posición 72
    (que es el estándar del dataset MIT-BIH con el que se entrenó).
    """
    # 1. Asegurar longitud 187 (Rellenar o Cortar)
    TARGET_LEN = 187
    if len(data_raw) < TARGET_LEN:
        padding = np.zeros(TARGET_LEN - len(data_raw))
        data_raw = np.concatenate((data_raw, padding))
    elif len(data_raw) > TARGET_LEN:
        data_raw = data_raw[:TARGET_LEN]
        
    # 2. Encontrar dónde está el pico mayor ahora
    current_peak = np.argmax(data_raw)
    
    # 3. Calcular cuánto hay que moverlo para que esté en el índice 72
    TARGET_PEAK = 72
    shift = TARGET_PEAK - current_peak
    
    # 4. Desplazar el array (Roll)
    data_centered = np.roll(data_raw, shift)
    
    # 5. Limpiar "basura" que haya podido rotar a los extremos
    # (Si desplazamos mucho, rellenamos el hueco dejado con ceros)
    if shift > 0:
        data_centered[:shift] = 0
    elif shift < 0:
        data_centered[shift:] = 0
        
    return data_centered.astype(np.float32)

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_todoterreno.keras')

try:
    model = load_model()
    st.success("✅ Sistema Listo")
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

# --- 4. INTERFAZ ---
uploaded_file = st.file_uploader("Sube tu CSV", type="csv")

if uploaded_file is not None:
    try:
        # Leer archivo
        df = pd.read_csv(uploaded_file, header=None)
        data_input = pd.to_numeric(df.iloc[0, :], errors='coerce').values
        data_input = data_input[~np.isnan(data_input)]
        
        # --- APLICAMOS LA CORRECCIÓN AUTOMÁTICA ---
        data_final = preprocess_ecg(data_input)
        
        # Mostrar gráfica (El latido ya debería verse centrado)
        st.write("### Señal Preprocesada (Centrada)")
        st.line_chart(data_final)
        
        if st.button("🔍 Diagnosticar"):
            # Reshape para IA
            data_reshaped = data_final.reshape(1, 187, 1)
            
            # Predicción
            prediction = model.predict(data_reshaped)
            clase_val = np.argmax(prediction)
            confianza = np.max(prediction) * 100
            
            clases = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)', 3: 'Fusión (F)', 4: 'Desconocido (Q)'}
            resultado = clases.get(clase_val, "Error")
            
            # Resultados
            c1, c2 = st.columns(2)
            c1.metric("Diagnóstico", resultado)
            c2.metric("Confianza", f"{confianza:.1f}%")
            
            if clase_val != 0:
                st.error(f"⚠️ Detección: {resultado}")
            else:
                st.success(f"✅ {resultado}")

            # SHAP
            try:
                st.subheader("Explicación (SHAP)")
                explainer = shap.DeepExplainer(model, np.zeros((1, 187, 1)))
                shap_values = explainer.shap_values(data_reshaped)
                
                # Corrección formato SHAP
                if isinstance(shap_values, list):
                    shap_val = shap_values[clase_val]
                else:
                    shap_val = shap_values
                
                shap_val = np.array(shap_val)
                # Corrección dimensiones extrañas
                if shap_val.size == 935: 
                    shap_val = shap_val.reshape(187, 5)[:, clase_val]
                
                shap_val = shap_val.flatten()[:187]

                # Plot
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(data_final, 'gray', alpha=0.3)
                ax.scatter(range(187), data_final, c=shap_val, cmap='coolwarm_r')
                st.pyplot(fig)
            except Exception as e:
                st.warning("Diagnóstico correcto, pero gráfico SHAP no disponible.")

    except Exception as e:
        st.error(f"Error: {e}")