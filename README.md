[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />
<div align="center">
  <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" alt="Logo" width="80" height="80">

  <h3 align="center">Detector de Arritmias con IA</h3>

  <p align="center">
    Un sistema avanzado de Deep Learning para la clasificación e interpretación de latidos cardíacos (ECG).
    <br />
    <br />
    <a href="#uso">Ver Demo</a>
    ·
    <a href="#instalación">Instalación</a>
    ·
    <a href="#fundamento-médico">Base Médica</a>
  </p>
</div>

---

## 🫀 Sobre el Proyecto

El diagnóstico temprano de arritmias cardíacas es crucial. Este proyecto implementa una **Red Neuronal Convolucional (CNN)** entrenada con el dataset estándar **MIT-BIH Arrhythmia Database** para clasificar latidos cardíacos en 5 categorías clínicas.

A diferencia de otros modelos, este sistema incluye **Explicabilidad (XAI)** con **SHAP**, permitiendo visualizar qué partes de la onda ECG fueron determinantes para el diagnóstico.

### Funcionalidades Clave
* **Detección Multi-clase:** Normal (N), Supraventricular (S), Ventricular (V), Fusión (F) y Desconocido (Q).
* **Modelo Robusto:** Entrenado con *Data Augmentation* para tolerar latidos desplazados o no centrados.
* **Interpretación Visual:** Mapas de calor para identificar anomalías morfológicas.
* **Interfaz Web:** Despliegue interactivo mediante Streamlit.

---

## 📚 Fundamento Médico

El modelo sigue los estándares de la **AAMI** (Association for the Advancement of Medical Instrumentation).

| Clase | Tipo | Descripción y Referencia |
| :---: | :--- | :--- |
| **N** | **Normal** | Ritmo sinusal fisiológico estándar. _(Ref: Goldberger et al., 2017)_ |
| **S** | **Supraventricular** | Latido prematuro auricular (PAC). QRS estrecho pero adelantado. _(Ref: Conen et al., Circulation 2012)_ |
| **V** | **Ventricular** | Latido prematuro ventricular (PVC). QRS ancho y deforme sin onda P. Es crítico detectarlo. _(Ref: Marcus, Circulation 2020)_ |
| **F** | **Fusión** | Colisión eléctrica entre un latido normal y uno ventricular. Morfología híbrida. _(Ref: Marriott's Practical ECG)_ |
| **Q** | **Desconocido** | Ritmos de marcapasos (con espiga vertical) o latidos no clasificables. |

---

## 🚀 Instalación y Uso

Si quieres ejecutar este proyecto en tu propio ordenador:

1. **Clona el repositorio:**
   ```bash
   git clone [https://github.com/TU-USUARIO/TU-REPOSITORIO.git](https://github.com/TU-USUARIO/TU-REPOSITORIO.git)
