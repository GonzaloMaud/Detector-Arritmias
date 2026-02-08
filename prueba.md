# 🫀 Detector de Arritmias Cardíacas con Deep Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Estudio comparativo de dos estrategias para clasificación de arritmias cardíacas con CNN  
basado en el MIT-BIH Arrhythmia Database.**

[📊 Comparativa de Modelos](#-comparativa-de-modelos-accuracy-vs-seguridad-clínica) • [Fundamentos Médicos](#-fundamentos-médicos-del-ecg) • [Arquitectura](#️-arquitectura-del-modelo) • [Resultados](#-análisis-visual-de-resultados)

</div>

---

## 🚀 Demos Disponibles

Prueba ambas versiones del sistema y compara su comportamiento:

<div align="center">

| Modelo | Enfoque | Demo en Vivo | Optimizado para |
|--------|---------|--------------|-----------------|
| **Modelo v1: Clásico** | Resampling (SMOTE/Oversampling) | [Abrir v1](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias) | **Accuracy global** |
| **Modelo v2: Cost-Sensitive** | `class_weight` + Data Augmentation | [Abrir v2](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmiasv2) | **Recall en clases minoritarias** |

</div>

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Comparativa de Modelos](#-comparativa-de-modelos-accuracy-vs-seguridad-clínica)
- [Preprocesamiento de los Datos](#-preprocesamiento-de-los-datos)
- [Fundamentos Médicos del ECG](#-fundamentos-médicos-del-ecg)
- [Tipos de Latidos Cardíacos](#-tipos-de-latidos-cardíacos)
- [Arquitectura del Modelo](#️-arquitectura-del-modelo)
- [Interpretabilidad con SHAP](#-interpretabilidad-con-shap)
- [Análisis Visual de Resultados](#-análisis-visual-de-resultados)
- [Instalación y Uso](#-instalación-y-uso)
- [Dataset](#-dataset)
- [Referencias Científicas](#-referencias-científicas)
- [Descargo de Responsabilidad Médica](#-descargo-de-responsabilidad-médica)
- [Licencia](#-licencia)
- [Autor](#-autor)

---

## 📄 Descripción General

Este proyecto implementa **dos enfoques diferentes** para la detección automática de arritmias cardíacas mediante **redes neuronales convolucionales 1D (CNN)**, entrenadas y evaluadas sobre el **MIT-BIH Arrhythmia Database**.

### El Dilema Fundamental

En machine learning aplicado a medicina aparece un **trade-off** entre:

1. **Maximizar accuracy** → acertar el máximo número de predicciones totales.
2. **Maximizar recall (sensibilidad)** → minimizar falsos negativos, especialmente en clases clínicas relevantes.

En este contexto:

- Un **falso positivo (FP)** → falsa alarma, más pruebas, coste adicional.
- Un **falso negativo (FN)** → arritmia real no detectada.

El objetivo de este trabajo es comparar:

- Un **modelo clásico** centrado en **accuracy global** mediante técnicas de resampling.
- Un **modelo cost-sensitive** que penaliza más los errores en clases minoritarias (supraventriculares, de fusión, etc.), sacrificando precisión y parte del accuracy global.

---

## 📊 Comparativa de Modelos: Accuracy vs. Seguridad Clínica

### 🧠 Filosofías de Diseño

<div align="center">

| Aspecto | Modelo v1: Clásico (Resampling) | Modelo v2: Cost-Sensitive |
|---------|----------------------------------|----------------------------|
| Objetivo principal | Maximizar **accuracy** del test | Aumentar **recall** en clases minoritarias |
| Técnica de balanceo | Oversampling / SMOTE | `class_weight` proporcional al desbalanceo |
| Datos | Datos balanceados sintéticamente | Datos originales desbalanceados |
| Data augmentation | Limitado | Desplazamiento, ruido, escalado |
| Ventaja principal | Métricas globales muy altas | Sensibilidad alta en S y F |
| Desventaja principal | Riesgo de sobreajuste a datos sintéticos | Más falsos positivos, menor accuracy global |

</div>

---

### 📈 Resultados Cuantitativos (Test oficial: `mitbih_test.csv`)

Los resultados siguientes se corresponden con la evaluación sobre el **test oficial** (`21892` latidos).

#### Modelo v1 – Enfoque Clásico (Resampling, centrado en Accuracy)

![Métricas Modelo v1](images/metricas_modelo_v1.png)

*Resultados del examen final (Test Set) – Modelo v1*

**Métricas globales:**

- **Accuracy**: **97 %**
- **Precision (macro avg)**: 0.82  
- **Recall (macro avg)**: 0.92  
- **F1-score (macro avg)**: 0.87  

**Métricas por clase:**

| Clase | Tipo | Precision | Recall | F1-score | Support |
|-------|------|-----------|--------|----------|---------|
| 0 | Normal (N)          | 0.99 | 0.97 | 0.98 | 18 118 |
| 1 | Supraventricular (S)| 0.66 | 0.82 | 0.73 | 556 |
| 2 | Ventricular (V)     | 0.91 | 0.95 | 0.93 | 1 448 |
| 3 | Fusión (F)          | 0.59 | 0.88 | 0.71 | 162 |
| 4 | Desconocido (Q)     | 0.97 | 0.99 | 0.98 | 1 608 |

Aproximando los falsos negativos:

- FN(N) ≈ 544  
- FN(S) ≈ 100  
- FN(V) ≈ 72  
- FN(F) ≈ 19  
- FN(Q) ≈ 16  

---

#### Modelo v2 – Enfoque Cost-Sensitive (centrado en Recall de minoritarias)

![Métricas Modelo v2](images/metricas_modelov2.png)

*Resultados del examen final (Test Set) – Modelo v2*

**Métricas globales:**

- **Accuracy**: **89 %**
- **Precision (macro avg)**: 0.65  
- **Recall (macro avg)**: 0.91  
- **F1-score (macro avg)**: 0.71  
- **Balanced accuracy** (aprox.): 0.91  

**Métricas por clase:**

| Clase | Tipo | Precision | Recall | F1-score | Support |
|-------|------|-----------|--------|----------|---------|
| 0 | Normal (N)          | 0.99 | 0.88 | 0.93 | 18 118 |
| 1 | Supraventricular (S)| 0.25 | 0.86 | 0.39 | 556 |
| 2 | Ventricular (V)     | 0.82 | 0.94 | 0.87 | 1 448 |
| 3 | Fusión (F)          | 0.24 | 0.90 | 0.38 | 162 |
| 4 | Desconocido (Q)     | 0.96 | 0.97 | 0.97 | 1 608 |

Falsos negativos aproximados:

- FN(N) ≈ 2 174  
- FN(S) ≈ 78  
- FN(V) ≈ 87  
- FN(F) ≈ 16  
- FN(Q) ≈ 48  

---

### 🔍 Análisis de Errores Críticos

Resumiendo para las clases no normales:

| Clase | Modelo v1 – FN | Modelo v2 – FN | Comentario |
|-------|----------------|----------------|------------|
| Supraventricular (S) | ≈ 100 | ≈ 78 | v2 reduce FN a costa de mucha menor precisión (muchos FP) |
| Ventricular (V)      | ≈ 72  | ≈ 87 | v1 detecta algo mejor V; v2 genera más FP y ligeramente más FN |
| Fusión (F)           | ≈ 19  | ≈ 16 | v2 mejora ligeramente el recall |
| Desconocido (Q)      | ≈ 16  | ≈ 48 | v1 es más estable en esta clase |

**Lectura clínica razonable:**

- **Modelo v1**  
  - Muy alto accuracy global (97 %) y buen comportamiento en todas las clases.  
  - Menos falsos positivos y algo mejor en latidos ventriculares.  
  - Puede perder más episodios supraventriculares que el modelo v2.

- **Modelo v2**  
  - Diseñado para **no “relajarse” con las clases minoritarias**: fuerza al modelo a etiquetar más S y F.  
  - Aumenta el **recall en S y F**, pero a cambio introduce muchos más falsos positivos y baja el accuracy global.  
  - Es más “agresivo” detectando actividad potencialmente anómala, a costa de un mayor número de alarmas innecesarias.

En un escenario real, la elección depende del contexto:

- Si el objetivo es **screening masivo** donde se toleran muchas falsas alarmas, el **modelo v2** puede tener sentido al priorizar sensibilidad en S y F.
- Si el objetivo es un sistema de apoyo más equilibrado, con menos ruido y buen rendimiento global, el **modelo v1** es más adecuado.

---

### 🧾 Matrices de Confusión

**Modelo v1 – Matriz de confusión:**

![Matriz de Confusión v1](images/matriz_modelov1.png)

**Modelo v2 – Matriz de confusión:**

![Matriz de Confusión v2](images/matriz_modelov2.png)

Estas matrices permiten ver en detalle cómo se distribuyen los errores entre clases, especialmente las confusiones frecuentes entre:

- **S ↔ N**,  
- **F ↔ N**,  
- y **V ↔ N** en casos de QRS menos extremos.

---

## 🧪 Preprocesamiento de los Datos

Los datasets utilizados **no son señales ECG crudas**, sino segmentos preprocesados siguiendo el estándar del **MIT-BIH Arrhythmia Database**.

### Proceso de Preprocesamiento

1. **Segmentación del ECG** en latidos individuales.  
2. **Alineamiento temporal** de cada latido respecto al pico R del complejo QRS.  
3. **Normalización temporal** a longitud fija de **187 muestras**.  
4. **Normalización de amplitud** al rango [0, 1].  
5. **Asignación de etiquetas** según la clasificación médica validada del MIT-BIH.

### Estructura de los Datos

Cada fila del dataset representa un **único latido**:

| Columnas | Descripción | Valores |
|----------|-------------|---------|
| 0–186 | Muestras del latido (ECG 1D) | 187 valores normalizados en [0, 1] |
| 187   | Etiqueta de clase           | {0, 1, 2, 3, 4} |

### Correspondencia de Etiquetas

| Etiqueta | Tipo de Latido | Descripción | Prevalencia (dataset completo) |
|----------|----------------|------------|--------------------------------|
| 0 | Normal (N)          | Latido sinusal normal                  | ~85.7 % |
| 1 | Supraventricular (S)| Extrasístole supraventricular          | ~2.5 % |
| 2 | Ventricular (V)     | Extrasístole ventricular               | ~7.3 % |
| 3 | Fusión (F)          | Latido de fusión                       | ~0.9 % |
| 4 | Desconocido (Q)     | Latido no clasificable / marcapasos    | ~3.5 % |

Este **desbalanceo extremo** es el motivo de la comparación entre:

- **Resampling (v1)** vs  
- **Cost-Sensitive Learning (v2)**.

---

## 🩺 Fundamentos Médicos del ECG

El electrocardiograma (ECG) registra la actividad eléctrica del corazón. En un ciclo normal aparecen:

- **Onda P** → despolarización auricular.  
- **Complejo QRS** → despolarización ventricular.  
- **Onda T** → repolarización ventricular.  

![Complejo QRS](images/qrs_complex_diagram.png)

El **complejo QRS** es crítico para la detección de muchas arritmias:

| Parámetro | Rango normal | Interpretación |
|-----------|-------------|----------------|
| Duración del QRS | 80–120 ms | QRS ancho suele indicar origen ventricular o bloqueo de conducción |
| Morfología | Estrecho y puntiagudo | Morfologías anchas/bizarras → posible foco ventricular |

---

## ❤️ Tipos de Latidos Cardíacos

![Comparación de Latidos ECG](images/ecg_beats_comparison.png)

| Tipo | Símbolo | Gravedad clínica aproximada | Acción médica típica |
|------|---------|-----------------------------|----------------------|
| Normal | N | Benigno | Sin intervención |
| Supraventricular | S | Monitorizar, valorar contexto | Holter si episodios frecuentes |
| Ventricular | V | Potencialmente grave | ECG urgente, posible hospitalización |
| Fusión | F | Atípico, requiere revisión | Valoración cardiológica |
| Desconocido | Q | Morfología no estándar | Revisar registro y contexto clínico |

---

## 🧱 Arquitectura del Modelo

![Arquitectura del Modelo](images/model_architecture.png)

Se implementa una **CNN 1D** común a ambos modelos:

```text
Input: ECG (187 puntos × 1 canal)
         ↓
Conv1D (64 filtros, kernel=5) + ReLU + MaxPooling
         ↓
Conv1D (128 filtros, kernel=5) + ReLU + MaxPooling
         ↓
Conv1D (256 filtros, kernel=3) + ReLU + GlobalAvgPooling
         ↓
Dense (128) + ReLU + Dropout(0.5)
         ↓
Dense (5) + Softmax
         ↓
Output: [P(N), P(S), P(V), P(F), P(Q)]
