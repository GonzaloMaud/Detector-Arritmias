[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-ff0055?style=for-the-badge)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />
<div align="center">
  <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" alt="Logo" width="80" height="80">

  <h3 align="center">Detector de Arritmias con IA & Explicabilidad</h3>

  <p align="center">
    Sistema de Deep Learning para la clasificación e interpretación morfológica de ECGs.
    <br />
    <br />
    <a href="#-análisis-visual-resultados"><strong>Ver Resultados Visuales »</strong></a>
    ·
    <a href="#-instalación-y-uso"><strong>Instalación</strong></a>
    ·
    <a href="#-fundamento-médico"><strong>Base Médica</strong></a>
  </p>
</div>

---

## 🫀 Sobre el Proyecto

El diagnóstico temprano de arritmias es vital. Este proyecto implementa una **Red Neuronal Convolucional (CNN)** robusta, entrenada con el dataset **MIT-BIH**, capaz de clasificar latidos incluso con ruido o desplazamientos (*Data Augmentation*).

Lo más innovador es su módulo de **Explicabilidad (XAI)**. No es una caja negra: el sistema le dice al médico **dónde está mirando** mediante mapas de calor SHAP.

---

## 👁️ Análisis Visual: Resultados

A continuación se muestra cómo el modelo "ve" e interpreta cada tipo de arritmia. A la izquierda la señal procesada, a la derecha el mapa de calor (SHAP) donde los puntos rojos indican las zonas determinantes para la IA.

| Clase Clínica | Señal ECG (Entrada) | Interpretación SHAP (Salida) |
| :--- | :---: | :---: |
| **Normal (N)**<br>Ritmo Sinusal | ![Normal ECG](images/normal_signal.png) | ![Normal SHAP](images/normal_shap.png) |
| **Ventricular (V)**<br>⚠️ *Crítico* | ![Ventricular ECG](images/ventricular_signal.png) | ![Ventricular SHAP](images/ventricular_shap.png) |
| **Supraventricular (S)**<br>Prematuro | ![Supra ECG](images/supra_signal.png) | ![Supra SHAP](images/supra_shap.png) |
| **Fusión (F)**<br>Híbrido | ![Fusion ECG](images/fusion_signal.png) | ![Fusion SHAP](images/fusion_shap.png) |
| **Desconocido/Paced (Q)**<br>Marcapasos | ![Paced ECG](images/paced_signal.png) | ![Paced SHAP](images/paced_shap.png) |

> *Nota: Los puntos rojos en SHAP indican las características morfológicas (como un QRS ancho o una espiga) que activaron la neurona de esa clase específica.*

---

## 📚 Fundamento Médico

El modelo sigue los estándares de la **AAMI** (Association for the Advancement of Medical Instrumentation).

1.  **Clase N (Normal):** Ritmo fisiológico estándar originado en el nodo sinusal.
2.  **Clase S (Supraventricular):** Latido prematuro originado en las aurículas. QRS generalmente estrecho.
3.  **Clase V (Ventricular):** Latido originado en los ventrículos. Se caracteriza por un **QRS ancho y deforme** y ausencia de onda P.
4.  **Clase F (Fusión):** Colisión eléctrica entre un latido normal y uno ventricular.
5.  **Clase Q (Desconocido):** Ritmos de marcapasos artificiales o latidos no clasificables.

---

## 🚀 Instalación y Uso

Si deseas ejecutar este proyecto en local:

1.  **Clonar el repositorio**
    ```bash
    git clone [https://github.com/TU-USUARIO/TU-REPO.git](https://github.com/TU-USUARIO/TU-REPO.git)
    ```
2.  **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecutar la Web App**
    ```bash
    streamlit run app.py
    ```

---

## 📧 Contacto

Desarrollado por **[TU NOMBRE]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/tu-usuario)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tu-usuario)
