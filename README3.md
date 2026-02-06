# 🫀 Detector de Arritmias Cardíacas con Deep Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Estudio comparativo de dos estrategias para clasificación de arritmias cardíacas mediante CNN:  
Accuracy vs. Seguridad Clínica**

[📊 Comparativa](#comparativa-de-modelos-accuracy-vs-seguridad-clínica) • [Fundamentos Médicos](#fundamentos-médicos-del-ecg) • [Arquitectura](#arquitectura-del-modelo) • [Dataset](#dataset)

</div>

---

## 🚀 Demos Disponibles

Prueba ambas versiones del sistema y compara su comportamiento clínico:

<div align="center">

| Modelo | Enfoque | Demo en Vivo | Optimizado Para |
|--------|---------|--------------|-----------------|
| **Modelo v1: Clásico** | Resampling (SMOTE/Oversampling) | [![Demo v1](https://img.shields.io/badge/🤗-Abrir%20v1-blue?style=flat-square)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias) | **Accuracy** (Exactitud Global) |
| **Modelo v2: Robusto** | Cost-Sensitive + Data Augmentation | [![Demo v2](https://img.shields.io/badge/🤗-Abrir%20v2-green?style=flat-square)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmiasv2) | **Recall** (Seguridad Clínica) |

</div>

---

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Comparativa de Modelos](#comparativa-de-modelos-accuracy-vs-seguridad-clínica)
- [Preprocesamiento de los Datos](#preprocesamiento-de-los-datos)
- [Fundamentos Médicos del ECG](#fundamentos-médicos-del-ecg)
- [Tipos de Latidos Cardíacos](#tipos-de-latidos-cardíacos)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Interpretabilidad con SHAP](#interpretabilidad-con-shap)
- [Análisis Visual de Resultados](#análisis-visual-de-resultados)
- [Dataset](#dataset)
- [Instalación y Uso](#instalación-y-uso)
- [Referencias Científicas](#referencias-científicas)
- [Licencia](#licencia)

---

## 🎯 Descripción General

Este proyecto implementa **dos enfoques diferentes** para la detección automática de arritmias cardíacas mediante redes neuronales convolucionales (CNN), entrenadas con el **MIT-BIH Arrhythmia Database**.

### El Dilema Fundamental

En machine learning médico existe un **trade-off crítico** entre dos objetivos:

1. **Maximizar Accuracy** → Acertar el máximo número de predicciones posibles
2. **Maximizar Recall (Sensibilidad)** → No dejar escapar ningún caso positivo real

**En cardiología, este dilema es literalmente de vida o muerte:**
- Un **Falso Positivo** (FP) → Falsa alarma → Pruebas adicionales innecesarias
- Un **Falso Negativo** (FN) → Arritmia no detectada → **Muerte del paciente**

Este proyecto explora ambos enfoques y demuestra cuál es más apropiado para aplicaciones clínicas reales.

### Características Principales

- **Dos modelos implementados**: Enfoque clásico vs. enfoque clínico
- **Comparativa rigurosa**: Métricas detalladas por clase y análisis de errores críticos
- **Interpretabilidad**: Visualización SHAP de las regiones críticas de la señal
- **Interfaz Web**: Ambos modelos desplegados en Hugging Face Spaces
- **Fundamento médico**: Justificación clínica de la elección del mejor modelo

---

## ⚔️ Comparativa de Modelos: Accuracy vs. Seguridad Clínica

### 📊 Filosofías de Diseño

<div align="center">

| Aspecto | Modelo v1: Clásico | Modelo v2: Robusto |
|---------|----------------------|----------------------|
| **Objetivo** | Maximizar **Accuracy** | Maximizar **Recall** en clases críticas |
| **Técnica de Balanceo** | Resampling (SMOTE/Oversampling) | Cost-Sensitive Learning (`class_weights`) |
| **Data Augmentation** | Mínimo | Vectorizado y agresivo |
| **Filosofía** | "Acertar el máximo posible" | "No dejar morir a nadie" |
| **Prioridad** | Métricas globales altas | Detectar **TODOS** los casos graves |
| **Riesgo Principal** | Overfitting a datos sintéticos | Más falsas alarmas (FP) |

</div>

### 📈 Resultados Cuantitativos

#### Modelo v1: Enfoque Clásico (Resampling)

**Métricas Globales:**
```
Accuracy Global: 98%
Precision Macro Avg: 0.92
Recall Macro Avg: 0.89
F1-Score Macro Avg: 0.91
```

**Métricas por Clase:**

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal (N)** | 0.98 | 0.99 | 0.98 | 15,010 |
| **Supraventricular (S)** | 0.91 | 0.87 | 0.89 | 445 |
| **Ventricular (V)** | 0.97 | 0.95 | 0.96 | 1,286 |
| **Fusión (F)** | 0.88 | 0.82 | 0.85 | 160 |
| **Desconocido (Q)** | 0.86 | 0.84 | 0.85 | 609 |

**Análisis Crítico:**
- **Fortaleza**: Métricas globales excepcionales (98% accuracy)
- **Debilidad**: Recall del 95% en Ventricular significa que **5 de cada 100 arritmias ventriculares NO se detectan**
- **Riesgo Clínico**: En un hospital con 1000 pacientes/día, esto implica **50 arritmias potencialmente mortales pasando desapercibidas**

---

#### Modelo v2: Enfoque Clínico Robusto (Cost-Sensitive)

**Métricas Globales:**
```
Accuracy Global: 94%  (↓ 4% vs. v1)
Precision Macro Avg: 0.87  (↓ 0.05 vs. v1)
Recall Macro Avg: 0.93  (↑ 0.04 vs. v1)
F1-Score Macro Avg: 0.90  (≈ similar a v1)
```

**Métricas por Clase:**

| Clase | Precision | Recall | F1-Score | Support | Cambio vs. v1 |
|-------|-----------|--------|----------|---------|---------------|
| **Normal (N)** | 0.96 | 0.98 | 0.97 | 15,010 | Recall: -1% |
| **Supraventricular (S)** | 0.75 | **0.92** | 0.83 | 445 | Recall: **+5%** |
| **Ventricular (V)** | 0.89 | **0.98** | 0.93 | 1,286 | Recall: **+3%** |
| **Fusión (F)** | 0.82 | **0.91** | 0.86 | 160 | Recall: **+9%** |
| **Desconocido (Q)** | 0.83 | 0.88 | 0.85 | 609 | Recall: +4% |

**Análisis Crítico:**
- **Fortaleza**: Recall del 98% en Ventricular → **Solo 2 de cada 100 arritmias ventriculares se pierden**
- **Seguridad**: En el mismo hospital con 1000 pacientes/día, solo **20 casos críticos** podrían pasar desapercibidos (vs. 50 del v1)
- **Trade-off**: Precision más baja (89% vs 97%) → **Más falsas alarmas**, pero esto es **clínicamente preferible**

---

### 🔬 Análisis de Errores Críticos

<div align="center">

| Métrica de Seguridad | Modelo v1 | Modelo v2 | Ganador |
|---------------------|--------------|--------------|---------|
| **Falsos Negativos (FN) en Ventricular** | 64 casos | **26 casos** | **v2** (60% menos FN) |
| **Falsos Negativos (FN) en Supraventricular** | 58 casos | **36 casos** | **v2** (38% menos FN) |
| **Recall Promedio Clases Minoritarias** | 0.88 | **0.93** | **v2** (+5.7%) |
| **Accuracy Global** | **98%** | 94% | v1 (+4%) |
| **Falsos Positivos (FP)** | 287 casos | **452 casos** | v1 (menos alarmas) |

</div>

**Interpretación Clínica:**

| Escenario | Modelo v1 | Modelo v2 | Consecuencia Real |
|-----------|-----------|-----------|-------------------|
| **Paciente con arritmia ventricular real** | 5% probabilidad de NO detectarlo | 2% probabilidad de NO detectarlo | v2 salva más vidas |
| **Paciente normal** | 1% probabilidad de falsa alarma | 3% probabilidad de falsa alarma | v2 genera más alarmas innecesarias |
| **Costo de error** | Muerte del paciente | Holter 24h adicional (~150€) | **v2 es infinitamente más seguro** |

---

### 🏆 Conclusión Clínica

<div align="center">

## El Modelo v2 (Robusto) es Superior para Aplicaciones Médicas Reales

</div>

#### Por qué el Modelo v2 gana:

1. **Principio Médico Fundamental**: *"Primum non nocere"* (Primero, no hacer daño)
   - Es **éticamente inaceptable** dejar escapar un infarto por optimizar accuracy
   - Una falsa alarma es un inconveniente; un falso negativo es una muerte

2. **Costo-Beneficio Favorable**:
   - **Costo de FP (Falso Positivo)**: Holter 24h (150€), Ecocardiograma (200€), ansiedad del paciente
   - **Costo de FN (Falso Negativo)**: Muerte súbita, demandas millonarias, pérdida de licencia médica

3. **Estándares Regulatorios** (FDA, CE, AEMPS):
   - Los dispositivos médicos deben priorizar **Sensibilidad (Recall) sobre Especificidad**
   - Un modelo con 98% accuracy pero 95% recall NO pasaría certificación

4. **Realidad Hospitalaria**:
   - Los médicos **siempre revisan las alarmas** manualmente
   - Es mejor tener 10 alarmas de más que 1 arritmia mortal sin detectar
   - El modelo actúa como **sistema de screening**, no diagnóstico final

#### Evidencia Numérica:

- **Modelo v1**: De 1,286 arritmias ventriculares reales, **falla en 64** → 64 muertes potenciales
- **Modelo v2**: De 1,286 arritmias ventriculares reales, **falla en 26** → 26 muertes potenciales
- **Resultado**: El modelo v2 salva **38 vidas adicionales** por cada 1,286 pacientes con arritmia ventricular

#### Trade-off Aceptable:

- **Precio**: 165 falsas alarmas adicionales (452 vs 287)
- **Beneficio**: 38 vidas salvadas
- **Ratio**: **1 vida salvada por cada 4.3 falsas alarmas adicionales**
- **Veredicto**: **Totalmente aceptable** desde cualquier perspectiva ética

---

### Recomendación Final

Para **aplicaciones clínicas reales**, utilizar el **Modelo v2 (Robusto)** porque:

- Cumple con el estándar médico de "mejor prevenir que lamentar"
- Reduce muertes evitables en un 60% (FN de V: 64 → 26)
- El trade-off (más falsas alarmas) es manejable clínicamente
- Es el único enfoque éticamente defendible en medicina

> **"En cardiología, una falsa alarma es un inconveniente. Un falso negativo es un certificado de defunción."**  
> — Principio de diseño de sistemas médicos críticos

---

## 📊 Preprocesamiento de los Datos

Los datasets utilizados en este proyecto **no corresponden a señales ECG crudas**, sino que han sido preprocesados previamente siguiendo el formato estándar del **MIT-BIH Arrhythmia Database**.

### Proceso de Preprocesamiento

El preprocesamiento aplicado a los datos originales consiste en:

1. **Segmentación de la señal ECG** en latidos individuales
2. **Alineamiento temporal** de cada latido respecto al pico R del complejo QRS
3. **Normalización temporal** a una longitud fija de 187 muestras
4. **Normalización de amplitud** al rango [0, 1]
5. **Asignación de etiquetas** según la clasificación médica validada

Este formato permite trabajar directamente con algoritmos de Machine Learning sin necesidad de aplicar técnicas complejas de procesamiento de señales sobre registros continuos de ECG.

### Estructura de los Datos

**Cada fila del dataset representa un único latido cardíaco**, con la siguiente estructura:

| Columnas | Descripción | Valores |
|----------|-------------|---------|
| **0 a 186** | Vector de características del latido | 187 valores numéricos normalizados [0, 1] |
| **187** | Etiqueta de clase | Valor entero {0, 1, 2, 3, 4} |

Es decir:
- **Cada fila = 1 latido completo** del ECG representado como un vector de 187 puntos
- **No hay señales continuas**: cada muestra es independiente
- **Formato listo para ML**: sin necesidad de filtrado adicional

### Correspondencia de Etiquetas

| Etiqueta | Tipo de Latido | Descripción Clínica | Prevalencia |
|----------|----------------|---------------------|-------------|
| **0** | Normal (N) | Latido sinusal normal | 85.7% |
| **1** | Supraventricular (S) | Extrasístole supraventricular | 2.5% |
| **2** | Ventricular (V) | Extrasístole ventricular | 7.3% |
| **3** | Fusión (F) | Latido de fusión | 0.9% |
| **4** | Desconocido (Q) | Latido no clasificable | 3.5% |

### Desbalanceo de Clases: El Problema Central

El **desbalanceo extremo** (85.7% vs. 0.9%) es el motivo de esta comparativa:

- **Modelo v1**: Genera datos sintéticos (SMOTE) para equilibrar → Riesgo de overfitting
- **Modelo v2**: No toca los datos, usa pesos de clase → Refleja la realidad clínica

### Implicaciones

Gracias a este preprocesamiento:

- **No es necesario** aplicar filtrado, detección de picos R, ni segmentación adicional
- **Los modelos trabajan** directamente con vectores de latidos individuales
- **El enfoque es adecuado** para clasificación supervisada de patrones cardíacos
- **La interpretación clínica** se centra en la morfología de cada latido aislado

> **Nota importante**: Este proyecto no pretende analizar señales ECG continuas ni realizar diagnósticos globales del ritmo cardíaco, sino **clasificar latidos individuales ya segmentados**, lo cual es coherente con el objetivo del dataset MIT-BIH y con el enfoque de aprendizaje automático utilizado.

---

## 🏥 Fundamentos Médicos del ECG

### Anatomía del Electrocardiograma

El electrocardiograma (ECG) es el registro gráfico de la actividad eléctrica del corazón a lo largo del tiempo. Según estudios clínicos bien establecidos y publicados en literatura médica revisada por pares, cada ciclo cardíaco normal presenta tres componentes principales que reflejan eventos electrofisiológicos específicos:

<div align="center">

![Complejo QRS](images/qrs_complex_diagram.png)

*Anatomía del electrocardiograma mostrando las ondas P, complejo QRS y onda T*

</div>

#### Onda P - Despolarización Auricular

La **onda P** representa la activación eléctrica de las aurículas (despolarización auricular).

**Características normales:**
- **Duración**: 80-120 ms
- **Amplitud**: < 2.5 mm (0.25 mV)
- **Morfología**: Redondeada y positiva en derivaciones inferiores

**Variabilidad morfológica:**
Según investigaciones electrofisiológicas, las alteraciones en la onda P pueden indicar:
- **Ausencia o inversión**: Ritmos de origen no sinusal
- **Onda P' (P prima)**: Activación auricular ectópica (supraventricular)
- **Ondas P múltiples**: Bloqueos auriculoventriculares
- **P picuda o ensanchada**: Crecimiento auricular

> En latidos **supraventriculares**, la onda P frecuentemente está ausente, fusionada con el QRS anterior, o presenta morfología anormal (P'), lo que constituye un marcador diagnóstico clave.

---

#### Complejo QRS - Despolarización Ventricular

El **complejo QRS** es la porción más prominente del ECG y representa la despolarización de los ventrículos, es decir, la propagación del impulso eléctrico a través del músculo ventricular que produce la contracción principal del corazón.

<div align="center">
```
        R
        ↑
        |
    ____│____
   |    |    |
Q  |    |    |  S
   |____|____|
```

</div>

**Componentes del QRS:**

| Componente | Descripción | Significado Fisiológico |
|------------|-------------|------------------------|
| **Onda Q** | Primera deflexión negativa | Despolarización del septum interventricular |
| **Onda R** | Primera deflexión positiva (principal) | Despolarización de la masa ventricular |
| **Onda S** | Deflexión negativa tras la R | Finalización de la despolarización ventricular |

**Parámetros normales del QRS:**

| Parámetro | Valor Normal | Significado Clínico |
|-----------|--------------|---------------------|
| **Duración** | **80-120 ms** | Tiempos > 120 ms sugieren bloqueos de conducción o origen ventricular |
| **Amplitud** | 5-30 mm | Varía según derivación; alteraciones indican hipertrofia o necrosis |
| **Morfología** | Estrecho y puntiagudo | QRS ancho y bizarro indica conducción anormal |

**Importancia del QRS en la detección de arritmias:**

Según la literatura cardiológica establecida:

**QRS estrecho (< 120 ms)**  
→ Indica que el impulso eléctrico ha seguido el **sistema de conducción normal** (haz de His → ramas → red de Purkinje)  
→ Característico de latidos **normales** y **supraventriculares**

**QRS ancho (> 120 ms)**  
→ Indica conducción **ventricular anormal** o impulso originado directamente en el ventrículo  
→ Típico de **extrasístoles ventriculares** y bloqueos de rama

**Morfología del QRS**  
→ La forma exacta (altura, simetría, presencia de muescas) permite diferenciar el origen del impulso  
→ Alteraciones en la morfología son la base de la clasificación automática mediante deep learning

---

#### Onda T - Repolarización Ventricular

La **onda T** representa la recuperación eléctrica de los ventrículos tras su contracción (repolarización ventricular).

**Características normales:**
- **Duración**: 160-200 ms
- **Morfología**: Asimétrica, con pendiente ascendente más lenta
- **Polaridad**: Generalmente positiva en derivaciones con QRS positivo

**Variabilidad clínica:**
- **Inversión de onda T**: Isquemia miocárdica, pericarditis
- **T picuda y alta**: Hiperpotasemia
- **T aplanada**: Hipopotasemia, isquemia
- **T prominente**: Repolarización precoz (normal en atletas)

> Aunque la onda T no es el foco principal en la clasificación de arritmias puntuales, sus alteraciones pueden acompañar a latidos ventriculares ectópicos y ayudar en el diagnóstico diferencial.

---

### El Complejo QRS como Marcador Diagnóstico

El análisis automatizado del **complejo QRS** es fundamental en la detección de arritmias porque:

1. **Su duración** diferencia origen supraventricular (< 120 ms) de ventricular (> 120 ms)
2. **Su morfología** permite identificar patrones específicos de cada tipo de latido
3. **Su amplitud y simetría** revelan alteraciones en la conducción eléctrica
4. **Sus relaciones con P y T** establecen la secuencia de activación cardíaca

Las redes neuronales convolucionales aprenden automáticamente estos patrones morfológicos del QRS que los cardiólogos utilizan en el diagnóstico clínico, pero pueden detectar sutilezas imperceptibles al ojo humano.

---

## 💓 Tipos de Latidos Cardíacos

Este proyecto clasifica latidos en 5 categorías basadas en la clasificación médica estándar del MIT-BIH Arrhythmia Database. A continuación se presenta una descripción detallada de cada tipo desde una perspectiva clínica y electrofisiológica.

<div align="center">

![Comparación de Latidos ECG](images/ecg_beats_comparison.png)

*Comparación de las características electrocardiográficas de los 5 tipos de latidos detectados por el sistema*

</div>

---

### Clasificación por Gravedad Clínica

| Tipo | Símbolo | Gravedad | Frecuencia | Acción Médica |
|------|---------|----------|------------|---------------|
| **Normal** | N | Benigno | 85.7% | Ninguna |
| **Supraventricular** | S | Monitorizar | 2.5% | Holter 24h si frecuente |
| **Ventricular** | V | **Urgente** | 7.3% | ECG urgente, posible hospitalización |
| **Fusión** | F | Atención | 0.9% | Evaluación cardiológica |
| **Desconocido** | Q | Revisar | 3.5% | Repetir ECG |

### Latido Normal (N - Normal Beat)

<div align="center">
```
        R
        ↑ Onda R prominente
    ____│____
P  |    |    |  T
   |____|____|
   ↑         ↑
   Q         S
```

</div>

#### Características Electrocardiográficas

| Parámetro | Valor/Descripción |
|-----------|-------------------|
| **Duración QRS** | 80-120 ms (estrecho) |
| **Morfología** | Onda R prominente, precedida de onda P |
| **Ritmo** | Regular, originado en el nodo sinusal |
| **Frecuencia** | 60-100 latidos por minuto |
| **Onda P** | Presente, positiva, precede al QRS |

#### Fisiología

El impulso eléctrico se origina en el **nodo sinoauricular (SA)** ubicado en la aurícula derecha, viaja a través de:

1. **Aurículas** → genera onda P
2. **Nodo auriculoventricular (AV)** → retraso fisiológico
3. **Haz de His** → entrada al sistema ventricular
4. **Ramas derecha e izquierda** → distribución ventricular
5. **Red de Purkinje** → despolarización coordinada de ambos ventrículos

Esta secuencia produce una **despolarización ventricular rápida y sincronizada**, resultando en un QRS estrecho y una contracción eficiente.

#### Significado Clínico

- Ritmo sinusal normal
- Función cardíaca coordinada
- Sin evidencia de arritmia

---

### Latido Supraventricular (S - Supraventricular Ectopic Beat)

<div align="center">
```
     R
     ↑ Prematuro, QRS estrecho
 ____│____
|    |    |  Sin onda P precedente
|____|____|  o P' anormal
↑         ↑
Q         S
```

</div>

#### Características Electrocardiográficas

| Parámetro | Valor/Descripción |
|-----------|-------------------|
| **Duración QRS** | 80-120 ms (estrecho, similar al normal) |
| **Morfología** | QRS de morfología normal pero **aparición prematura** |
| **Onda P** | Ausente, aberrante (P') o fusionada con el QRS previo |
| **Origen** | Aurículas o unión AV (por encima de los ventrículos) |
| **Timing** | Ocurre antes del siguiente latido sinusal esperado |

#### Fisiopatología

Las **extrasístoles supraventriculares** (también llamadas contracciones auriculares prematuras - PACs) son latidos originados en focos ectópicos ubicados en:

- **Aurículas** (tejido auricular fuera del nodo SA)
- **Unión auriculoventricular** (región del nodo AV)

**Mecanismo:**
1. Un foco irritable en las aurículas genera un impulso prematuro
2. Este impulso se propaga y despolariza las aurículas (P' anormal o ausente)
3. El impulso desciende por el sistema de conducción **normal** (nodo AV → His → Purkinje)
4. Los ventrículos se despolarizan **normalmente** → QRS estrecho

**La clave diagnóstica**: QRS estrecho + aparición prematura + P ausente/anormal

#### Causas Comunes

Según estudios clínicos, los latidos supraventriculares son frecuentes en:

- Consumo excesivo de cafeína o alcohol
- Estrés, ansiedad o fatiga
- Desequilibrios electrolíticos (hipopotasemia, hipomagnesemia)
- Cardiopatías estructurales (dilatación auricular)
- Efectos de ciertos medicamentos

#### Significado Clínico

- **Aislados**: Generalmente benignos en corazones sanos
- **Frecuentes (> 10/hora)**: Pueden indicar predisposición a taquicardia supraventricular
- **Muy frecuentes**: Requieren evaluación cardiológica y posible tratamiento

---

### Latido Ventricular (V - Ventricular Ectopic Beat) - EL MÁS CRÍTICO

<div align="center">
```
        R
       ↗↑↖  Ancho, bizarro
    __/  │ \__
   |     |    |  QRS > 120 ms
   |_____|____|
   ↑          ↑
   Ausencia   Morfología
   de P       anormal
```

</div>

#### Características Electrocardiográficas

| Parámetro | Valor/Descripción |
|-----------|-------------------|
| **Duración QRS** | **> 120 ms** (significativamente ancho) |
| **Morfología** | **Bizarra y deformada**, muy diferente al QRS normal |
| **Onda P** | **Ausente** (no hay relación con actividad auricular) |
| **Amplitud** | Generalmente **mayor** que el latido normal |
| **Onda T** | Frecuentemente discordante (polaridad opuesta al QRS) |

#### Fisiopatología

Las **extrasístoles ventriculares** (PVC - Premature Ventricular Contractions) se originan en focos ectópicos ubicados directamente en el **músculo ventricular**, saltándose completamente el sistema de conducción normal.

**Mecanismo de conducción anormal:**

1. **Impulso ectópico** se origina en el ventrículo (no en aurículas ni nodo AV)
2. **No utiliza el sistema de Purkinje** → la activación se propaga célula a célula por el músculo ventricular
3. **Despolarización lenta y descoordinada** → el impulso tarda mucho más en recorrer ambos ventrículos
4. **Resultado**: QRS muy ancho (> 120 ms) y de morfología bizarra

**Diferencias con la conducción normal:**

| Aspecto | Latido Normal | Latido Ventricular |
|---------|---------------|-------------------|
| Vía de conducción | Purkinje (rápida) | Músculo a músculo (lenta) |
| Duración QRS | 80-120 ms | > 120 ms |
| Morfología | Regular | Bizarra, ancha |
| Sincronización | Coordinada | Descoordinada |

#### Por qué es la clase más importante

Las **extrasístoles ventriculares** pueden preceder:
- Taquicardia ventricular
- Fibrilación ventricular
- **Muerte súbita cardíaca**

**Por esto, el Recall en la clase V es la métrica más crítica del modelo.**

#### Clasificación Clínica

Según la frecuencia y patrón de aparición:

- **Aisladas**: < 30/hora → generalmente benignas
- **Frecuentes**: 30-100/hora → requieren monitorización
- **Muy frecuentes**: > 100/hora → evaluación cardiológica urgente
- **Bigeminismo**: PVC cada 2 latidos
- **Trigeminismo**: PVC cada 3 latidos
- **Salvas**: 3 o más PVCs consecutivas → riesgo de taquicardia ventricular

#### Implicaciones Clínicas

**En corazones sanos**:
- PVCs aisladas son comunes y generalmente benignas
- Pueden ser desencadenadas por estrés, cafeína, fatiga

**En cardiopatías**:
- Pueden indicar isquemia miocárdica
- Riesgo de arritmias ventriculares malignas
- Pueden preceder **taquicardia ventricular** o **fibrilación ventricular**

**Fenómeno R sobre T**: PVC que cae sobre la onda T previa → alto riesgo de fibrilación ventricular

---

### Latido de Fusión (F - Fusion Beat)

<div align="center">
```
      R        R
      ↑       ↑
    __│__   __│__
   |  │  | |  │  |  Morfología híbrida
   |__|__| |__|__|
   ↑              ↑
   Normal      Ventricular
   (supraventricular)
```

</div>

#### Características Electrocardiográficas

| Parámetro | Valor/Descripción |
|-----------|-------------------|
| **Duración QRS** | **Intermedia** (100-140 ms) |
| **Morfología** | **Híbrida** entre latido normal y ventricular |
| **Amplitud** | Variable, depende de la proporción de fusión |
| **Forma** | Mezcla características de ambos tipos de latido |

#### Fisiopatología

Los latidos de fusión son un **fenómeno electrofisiológico único** que ocurre cuando dos impulsos eléctricos de origen diferente colisionan simultáneamente en los ventrículos:

**Mecanismo de formación:**

1. **Impulso supraventricular** (del nodo SA) desciende normalmente por el sistema de conducción
2. **Impulso ventricular** (de foco ectópico) surge desde un ventrículo
3. **Ambos impulsos convergen** y despolarizan diferentes regiones ventriculares al mismo tiempo
4. **Resultado**: Complejo QRS que es una **combinación** de ambos patrones

**Características específicas:**

- El QRS resultante tiene morfología **intermedia** entre normal y ventricular
- La forma exacta depende del **timing relativo** y **localización** de los dos frentes de onda
- **No son una arritmia per se**, sino un fenómeno de superposición

**Visualización del proceso:**
```
Ventrículo izquierdo    Ventrículo derecho
        ↓                       ↓
    [Impulso normal]    [Impulso ectópico]
        ↓                       ↓
        └─────── FUSIÓN ───────┘
                  ↓
            QRS híbrido
```

#### Contexto Clínico

Los latidos de fusión son más comunes cuando hay:

- **Extrasístoles ventriculares frecuentes** compitiendo con el ritmo sinusal
- **Ritmos idioventriculares acelerados**
- **Marcapasos ventriculares** (fusión entre latido estimulado y latido propio)
- **Taquicardia ventricular** intermitente

#### Significado Diagnóstico

- **Confirmación de origen ventricular**: La presencia de latidos de fusión **confirma** que otros latidos anchos en el ECG son de origen ventricular (no bloqueo de rama)
- **Indicador de competencia**: Sugiere que hay **dos marcapasos activos** simultáneamente
- **No patológicos por sí mismos**: El latido de fusión en sí no es peligroso, pero indica actividad ectópica subyacente

---

### Latido Desconocido (Q - Unclassified Beat)

<div align="center">
```
    ????
    Morfología irregular
    o muy atípica
```

</div>

#### Características

| Aspecto | Descripción |
|---------|-------------|
| **Morfología** | No se ajusta claramente a ninguna categoría estándar |
| **Origen** | Incierto, múltiple, o artefacto |
| **Variabilidad** | Alta heterogeneidad morfológica |

#### Causas Posibles

La clase "Desconocido" agrupa latidos que no pueden ser clasificados con certeza debido a:

**1. Artefactos técnicos:**
- Interferencia eléctrica (50/60 Hz de la red eléctrica)
- Artefactos por movimiento muscular
- Contacto pobre de electrodos
- Ruido electromagnético

**2. Arritmias complejas:**
- Latidos con características mixtas no clasificables
- Aberraciones de conducción atípicas
- Morfologías muy distorsionadas por cardiopatías severas

**3. Latidos raros:**
- Extrasístoles de la unión AV con conducción aberrante
- Latidos de escape de diferentes focos
- Variantes morfológicas poco frecuentes

#### Relevancia Clínica

En la práctica médica real, estos latidos requieren:

- **Revisión manual** por cardiólogo experto
- **Repetición del ECG** si hay muchos latidos no clasificables
- **Correlación clínica** con síntomas y contexto del paciente
- **Estudios adicionales**:
  - Holter 24 horas (monitorización continua)
  - Ecocardiograma (evaluación estructural)
  - Prueba de esfuerzo (provocación de arritmias)
  - Estudio electrofisiológico (en casos complejos)

#### Limitaciones del Modelo

Es **normal y esperado** que un porcentaje de latidos caiga en esta categoría porque:

- Algunos patrones son intrínsecamente ambiguos
- Los artefactos son difíciles de distinguir de señales reales
- Existen arritmias raras no representadas suficientemente en el dataset
- La variabilidad biológica excede las 4 categorías principales

> **Nota**: Un buen modelo de clasificación de ECG debe tener una clase "Desconocido" para evitar clasificaciones erróneas con alta confianza en casos ambiguos. Esto es más seguro clínicamente que forzar una etiqueta incorrecta.

---

## 🏗️ Arquitectura del Modelo

### Red Neuronal Convolucional (CNN 1D)

<div align="center">

![Arquitectura del Modelo](images/model_architecture.png)

*Arquitectura de la red neuronal convolucional utilizada para la clasificación de arritmias*

</div>

### Por qué CNN para señales ECG

Las **Redes Neuronales Convolucionales (CNN)** son ideales para analizar señales temporales como el ECG porque:

1. **Detección de patrones locales**: Las capas convolucionales aprenden automáticamente a detectar características morfológicas específicas:
   - Picos (onda R)
   - Valles (ondas Q y S)
   - Pendientes (ascensos y descensos rápidos)
   - Duraciones (anchura del QRS)
   - Formas características (morfología del complejo)

2. **Invariancia temporal limitada**: Las CNN pueden reconocer patrones incluso si están ligeramente desplazados en el tiempo, lo cual es útil dado que los latidos pueden tener pequeñas variaciones en su posición exacta.

3. **Jerarquía de características**: Las capas convolucionales apilen extraen progresivamente características de mayor nivel:
   - **Capa 1**: Detecta bordes, cambios bruscos
   - **Capa 2**: Detecta patrones locales (mini-picos, curvaturas)
   - **Capa 3**: Detecta patrones complejos (complejo QRS completo, morfologías específicas)

4. **Eficiencia computacional**: Comparadas con redes totalmente conectadas, las CNN tienen muchos menos parámetros y se entrenan más rápido.

### Arquitectura Implementada

**Arquitectura común a ambos modelos (v1 y v2):**
```
Input: ECG (187 puntos × 1 canal)
         ↓
┌─────────────────────────┐
│  Conv1D (64 filtros)    │  ← Extrae características básicas
│  Kernel: 5              │     (cambios, pendientes)
│  Activation: ReLU       │
│  MaxPooling1D (2)       │  ← Reduce dimensionalidad
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Conv1D (128 filtros)   │  ← Patrones de nivel medio
│  Kernel: 5              │     (ondas P, picos R, ondas S)
│  Activation: ReLU       │
│  MaxPooling1D (2)       │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Conv1D (256 filtros)   │  ← Características complejas
│  Kernel: 3              │     (morfología QRS completa)
│  Activation: ReLU       │
│  GlobalAvgPooling1D     │  ← Resumen de toda la señal
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Dense (128 neuronas)   │  ← Combinación de características
│  Activation: ReLU       │     para clasificación
│  Dropout (0.5)          │  ← Prevención de overfitting
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Dense (5 neuronas)     │  ← Capa de salida
│  Activation: Softmax    │     (probabilidades de 5 clases)
└─────────────────────────┘
         ↓
   [N, S, V, F, Q]
```

### Detalles Técnicos

| Componente | Configuración | Función |
|------------|--------------|---------|
| **Input** | (187, 1) | Señal ECG de un latido |
| **Conv1D layers** | 3 capas con 64→128→256 filtros | Extracción jerárquica de características |
| **Kernel sizes** | 5, 5, 3 | Ventanas de análisis temporal |
| **Pooling** | MaxPooling1D (pool_size=2) | Reducción de dimensionalidad, invariancia |
| **GlobalAvgPooling** | - | Convierte mapas de características en vector |
| **Dense layer** | 128 neuronas + Dropout(0.5) | Clasificación con regularización |
| **Output** | 5 neuronas + Softmax | Probabilidades para cada clase |

### Diferencias en el Entrenamiento

| Aspecto | Modelo v1 | Modelo v2 |
|---------|-----------|-----------|
| **Datos de Entrada** | Resampling (datos sintéticos) | Datos originales sin alterar |
| **Pesos de Clase** | Uniforme (1.0 para todas) | Inversamente proporcional a frecuencia |
| **Función de Pérdida** | `categorical_crossentropy` | `categorical_crossentropy` con `class_weight` |
| **Data Augmentation** | Mínimo | Desplazamientos + ruido + escalado |
| **Épocas** | 50 | 75 |
| **Early Stopping** | Monitoring: `val_loss` | Monitoring: `val_recall_V` (Recall en V) |

### Cómo Aprende el Modelo

Durante el entrenamiento con **backpropagation**, la red ajusta automáticamente sus filtros convolucionales para maximizar la capacidad de distinguir entre clases. Por ejemplo:

- **Filtros en capas tempranas** aprenden a detectar el inicio y fin del QRS
- **Filtros en capas medias** aprenden a medir la anchura y altura de picos
- **Filtros en capas profundas** aprenden patrones morfológicos completos que distinguen V de S

Este proceso es **completamente automático**: no se programan manualmente las características a detectar, sino que la red las **descubre por sí misma** a partir de los datos etiquetados.

---

## 🔍 Interpretabilidad con SHAP

Uno de los mayores desafíos de los modelos de deep learning es su naturaleza de "caja negra": pueden hacer predicciones precisas, pero es difícil entender **por qué** tomaron una decisión específica. Esto es especialmente problemático en aplicaciones médicas donde la interpretabilidad es crucial para la confianza clínica.

### Qué es SHAP

**SHAP (SHapley Additive exPlanations)** es un método basado en teoría de juegos que explica predicciones de modelos de machine learning asignando a cada característica de entrada un **valor de importancia** (Shapley value).

#### Fundamento Teórico

Los valores de Shapley provienen de la **teoría de juegos cooperativos** (Lloyd Shapley, Premio Nobel de Economía 2012). La idea es:

- Cada característica de entrada (cada punto del ECG) es un "jugador"
- La predicción final es el "premio" que deben repartirse
- El valor de Shapley de cada característica es su **contribución justa** a la predicción

**Matemáticamente**:
- SHAP calcula cuánto cambiaría la predicción si se elimina o incluye cada característica
- Lo hace considerando **todas las combinaciones posibles** de características
- El resultado es un valor numérico para cada punto del ECG

### Cómo se Aplica SHAP al ECG

En este proyecto:
```python
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(latido_a_explicar)
```

1. **DeepExplainer**: Versión de SHAP optimizada para redes neuronales profundas
2. **Background data**: Conjunto de referencia (latidos base) para comparación
3. **SHAP values**: Vector de 187 valores (uno por cada punto del ECG)

### Interpretación Visual

<div align="center">

![Ejemplo SHAP](images/shap_example.png)

*Ejemplo de explicación SHAP mostrando las regiones críticas de la señal ECG para la clasificación*

</div>

En cada gráfico SHAP generado por la aplicación:

| Color | Significado | Interpretación |
|-------|-------------|----------------|
| **Azul intenso** | **Contribución positiva fuerte** | "Esta región de la señal empuja fuertemente la predicción hacia la clase predicha" |
| **Azul suave** | **Contribución positiva débil** | "Esta región apoya ligeramente la predicción" |
| **Gris/Blanco** | **Contribución neutral** | "Esta región no influye en la decisión" |
| **Rojo suave** | **Contribución negativa débil** | "Esta región va ligeramente en contra de esta clase" |
| **Rojo intenso** | **Contribución negativa fuerte** | "Esta región descarta fuertemente esta clase" |

**La intensidad del color** indica la magnitud de la contribución.

### Aplicación Clínica de SHAP

#### Ejemplo 1: Latido Ventricular
```
Predicción: Ventricular (V) - 97.3% confianza
SHAP muestra:
  Azul intenso en QRS ancho → "El QRS ensanchado es la evidencia principal"
  Rojo en segmentos planos → "La ausencia de onda P apoya que NO es normal"
```

El cardiólogo puede **validar** que el modelo está usando los criterios correctos (anchura del QRS).

#### Ejemplo 2: Latido Supraventricular
```
Predicción: Supraventricular (S) - 77.8% confianza
SHAP muestra:
  Azul en inicio del latido → "Irregularidad pre-QRS detectada"
  Azul en QRS estrecho → "Morfología compatible con conducción normal"
  Rojo en regiones regulares → "Descarta latido completamente normal"
```

La menor confianza (77.8% vs 97.3% del ventricular) se refleja en valores SHAP menos extremos, indicando que la señal es más ambigua.

### Validación Médica con SHAP

**Modelo v1 y v2 (ambos correctos):**
- Azul intenso en el **QRS ancho** (> 120 ms)
- Rojo en segmentos planos (ausencia de onda P)

**Validación médica**: Ambos modelos aprenden correctamente que el QRS ensanchado es la característica clave de un latido ventricular.

### Ventajas de SHAP en Aplicaciones Médicas

1. **Transparencia**: Convierte el modelo en explicable, no solo preciso
2. **Validación clínica**: Permite verificar que el modelo usa criterios médicamente relevantes
3. **Confianza**: Los médicos pueden confiar más en predicciones que entienden
4. **Detección de errores**: Si SHAP marca regiones irrelevantes, indica problemas en el modelo
5. **Educación**: Ayuda a entender qué características morfológicas son diagnósticas

### Limitaciones y Consideraciones

**SHAP no es perfecto**:
- Los valores son **aproximaciones** (no siempre únicos matemáticamente)
- La elección del background data afecta los resultados
- La interpretación requiere conocimiento del dominio (ECG)
- SHAP explica **este modelo específico**, no la realidad médica subyacente

> **Nota importante**: SHAP muestra qué usa **el modelo**, no necesariamente qué deberían usar los médicos. Si el modelo está mal entrenado, SHAP mostrará criterios incorrectos con claridad.

---

## 📊 Análisis Visual de Resultados

Las siguientes capturas de pantalla corresponden a **ejecuciones reales de las aplicaciones** desplegadas en Hugging Face Spaces. Cada ejemplo muestra:

1. **La señal ECG del latido** cargado desde un archivo CSV
2. **La predicción del modelo** con su clase y nivel de confianza
3. **El mapa SHAP** con las regiones críticas de la señal resaltadas

Todos los latidos utilizados son **muestras reales** del conjunto de test del MIT-BIH Arrhythmia Database, asegurando que las predicciones reflejan el rendimiento del modelo en datos no vistos durante el entrenamiento.

---

### Resultados del Modelo v1 (Clásico)

<div align="center">

![Matriz de Confusión v1](images/confusion_matrix.png)

*Matriz de confusión del Modelo v1 - Enfoque optimizado para Accuracy*

</div>

**Observaciones clave de la matriz v1:**
- Alta precisión en la diagonal (clases correctamente clasificadas)
- Algunos errores en clases minoritarias (S, F, Q)
- 64 falsos negativos en la clase Ventricular (V) - **casos críticos no detectados**
- 58 falsos negativos en la clase Supraventricular (S)

---

### Resultados del Modelo v2 (Robusto)

<div align="center">

![Matriz de Confusión v2](images/confusion_matrix_v2.png)

*Matriz de confusión del Modelo v2 - Enfoque optimizado para Recall*

</div>

**Observaciones clave de la matriz v2:**
- Diagonal menos "perfecta" que v1, pero **mejores resultados en clases críticas**
- Solo 26 falsos negativos en Ventricular (vs 64 del v1) - **60% de reducción**
- Solo 36 falsos negativos en Supraventricular (vs 58 del v1) - **38% de reducción**
- Mayor cantidad de falsos positivos (más alarmas), pero **clínicamente aceptable**

---

### Ejemplos de Predicciones

Ambos modelos se probaron con los mismos latidos reales del MIT-BIH Test Set:

| Latido Real | Modelo v1 Predice | Modelo v2 Predice | Correcto |
|-------------|-------------------|-------------------|----------|
| Ventricular | Ventricular (98%) | Ventricular (96%) | Ambos |
| Supraventricular | Normal (72%) | Supraventricular (89%) | **Solo v2** |
| Fusión | Fusión (91%) | Fusión (88%) | Ambos |
| Normal | Normal (100%) | Normal (99%) | Ambos |

**Observación clave**: El Modelo v2 detecta más casos de clases minoritarias (S, F) que el v1 pasaba por alto.

---

### Ejemplo de Interfaz: Latido Normal

<div align="center">

![Ejemplo Aplicación - Latido Normal](images/app_example_normal.png)

*Interfaz de la aplicación mostrando la clasificación de un latido normal con su explicación SHAP*

</div>

**Predicción**: Normal (N) - **100% de confianza**

**Interpretación SHAP**:
- Zonas azules concentradas en el complejo QRS: El modelo identifica la morfología típica del QRS (estrecho, simétrico, bien definido) como la característica principal de un latido normal
- Azul en la onda P: La presencia de una onda P regular refuerza la predicción de ritmo sinusal normal
- Zonas rojas en segmentos planos: Las regiones sin variación (línea isoeléctrica) no aportan evidencia de arritmia

**Análisis**:  
La red ha aprendido correctamente que un latido normal se caracteriza por:
- Presencia de onda P
- QRS estrecho y regular
- Morfología estable y predecible

---

### Flujo del Sistema

<div align="center">

![Flujo del Sistema](images/system_flow.png)

*Pipeline completo: Carga → Preprocesamiento → CNN → Predicción → Explicación SHAP*

</div>

---

## 📊 Dataset

### MIT-BIH Arrhythmia Database

Este proyecto utiliza el **MIT-BIH Arrhythmia Database**, uno de los datasets de referencia más utilizados en investigación de arritmias cardíacas.

#### Características del Dataset

| Aspecto | Detalles |
|---------|----------|
| **Fuente** | PhysioNet / MIT-BIH |
| **Año** | 1980 (actualizado regularmente) |
| **Pacientes** | 47 individuos |
| **Duración** | ~30 minutos por registro |
| **Frecuencia de muestreo** | 360 Hz |
| **Anotaciones** | Revisadas por dos cardiólogos expertos independientes |
| **Registros** | 48 grabaciones de ECG de dos canales |

#### Origen y Descripción

El **MIT-BIH Arrhythmia Database** fue desarrollado en 1980 por el **Massachusetts Institute of Technology (MIT)** y el **Beth Israel Hospital** (ahora Beth Israel Deaconess Medical Center) como parte del proyecto **PhysioNet**.

**Objetivo original**: Proporcionar un estándar de referencia para la evaluación de algoritmos de detección de arritmias mediante el análisis automático de señales ECG.

**Proceso de creación**:
1. Selección de 47 pacientes representativos de la población clínica
2. Grabación continua de ECG durante aproximadamente 30 minutos por paciente
3. **Anotación manual** por dos cardiólogos expertos de forma independiente
4. Revisión y resolución de discrepancias para crear anotaciones de consenso
5. Clasificación de cada latido según la nomenclatura estándar de la AAMI (Association for the Advancement of Medical Instrumentation)

**Importancia histórica**: Este dataset se ha convertido en el **estándar de oro** para la validación de algoritmos de clasificación de arritmias, siendo citado en más de 2,000 publicaciones científicas.

#### Distribución de Clases

El dataset preprocesado utilizado en este proyecto tiene la siguiente distribución:

| Clase | Cantidad Original | Porcentaje | Descripción |
|-------|------------------|------------|-------------|
| **Normal (N)** | 75,052 | 85.7% | Latidos sinusales normales |
| **Ventricular (V)** | 6,431 | 7.3% | Extrasístoles ventriculares |
| **Supraventricular (S)** | 2,223 | 2.5% | Extrasístoles supraventriculares |
| **Desconocido (Q)** | 3,046 | 3.5% | Latidos no clasificables o artefactos |
| **Fusión (F)** | 802 | 0.9% | Latidos de fusión |
| **Total Original** | **87,554** | 100% | - |

**Observación**: El fuerte desbalanceo original (85.7% normales vs. 0.9% fusión) es **realista y refleja la distribución natural** de arritmias en la población clínica. Este desbalanceo justifica la necesidad de técnicas especializadas (resampling vs. cost-sensitive learning) evaluadas en este proyecto.

#### Preprocesamiento Kaggle

El dataset fue preprocesado por la comunidad de Kaggle siguiendo estos pasos:

1. **Segmentación**: Extracción de latidos individuales centrados en el pico R
2. **Normalización temporal**: Ajuste a 187 muestras por latido mediante interpolación
3. **Normalización de amplitud**: Escalado al rango [0, 1] para facilitar el entrenamiento
4. **División**: Train (87,554 latidos) y Test (21,892 latidos) con distribución estratificada

#### Acceso al Dataset

- **Kaggle**: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- **PhysioNet (Original)**: https://www.physionet.org/content/mitdb/1.0.0/

#### Relevancia Científica

El MIT-BIH Arrhythmia Database ha sido fundamental para:

- **Validación de algoritmos**: Estándar de referencia desde hace más de 40 años
- **Comparación de métodos**: Permite comparar resultados entre diferentes enfoques
- **Reproducibilidad**: Dataset público que garantiza la reproducibilidad de investigaciones
- **Avances en IA médica**: Ha impulsado el desarrollo de técnicas de machine learning aplicadas a cardiología

> **Nota**: Aunque el dataset tiene más de 40 años, sigue siendo el estándar de referencia debido a la **calidad excepcional de sus anotaciones** (revisadas manualmente por expertos) y su **representatividad clínica**.

---

## 🚀 Instalación y Uso

### Probar Online (Recomendado)

**Modelo v1 (Clásico):**  
https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias

**Modelo v2 (Robusto):**  
https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmiasv2

### Instalación Local
```bash
# Clonar el repositorio
git clone https://github.com/GonzaloMaud/detector-arritmias.git
cd detector-arritmias

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación (elige la versión)
streamlit run app_v1.py  # Modelo Clásico
streamlit run app_v2.py  # Modelo Robusto
```

### Uso de la Aplicación

1. **Preparar un archivo CSV**:
   - Una fila con 187 valores numéricos separados por comas
   - Sin cabecera, sin columna de etiqueta

2. **Cargar el archivo** en la interfaz web

3. **Revisar resultados**:
   - **Gráfico de señal**: Visualización del latido
   - **Diagnóstico**: Tipo de latido detectado
   - **Confianza**: Probabilidad de la predicción
   - **Mapa SHAP**: Regiones críticas de la señal

**Formato del CSV:**
```csv
1.0,0.758,0.111,0.0,0.080,0.158,...(187 valores totales)
```

---

## 📚 Referencias Científicas

Este proyecto está basado en conocimiento médico y técnico establecido en la literatura científica. A continuación se presentan referencias clave:

### Fundamentos Médicos del ECG

1. **Goldberger, A. L., et al.** (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215-e220.
   - Fuente del MIT-BIH Arrhythmia Database

2. **Moody, G. B., & Mark, R. G.** (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
   - Descripción completa del dataset y su impacto en la investigación

3. **Wagner, G. S., et al.** (2009). *AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram.* Journal of the American College of Cardiology, 53(11), 976-981.
   - Criterios clínicos para interpretación de ECG

### Deep Learning para ECG

4. **Rajpurkar, P., et al.** (2017). *Cardiologist-level arrhythmia detection with convolutional neural networks.* arXiv preprint arXiv:1707.01836.
   - CNN para detección de arritmias con rendimiento equiparable a cardiólogos

5. **Hannun, A. Y., et al.** (2019). *Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network.* Nature Medicine, 25(1), 65-69.
   - Aplicación clínica de deep learning en ECG ambulatorio

6. **Acharya, U. R., et al.** (2017). *A deep convolutional neural network model to classify heartbeats.* Computers in Biology and Medicine, 89, 389-396.
   - Arquitecturas CNN específicas para clasificación de latidos

### Interpretabilidad en ML Médico

7. **Lundberg, S. M., & Lee, S. I.** (2017). *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems, 30.
   - Fundamento teórico de SHAP

8. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). *"Why should I trust you?" Explaining the predictions of any classifier.* Proceedings of the 22nd ACM SIGKDD, 1135-1144.
   - Importancia de la interpretabilidad en ML para salud

### Clases Desbalanceadas

9. **Branco, P., Torgo, L., & Ribeiro, R. P.** (2016). *A survey of predictive modeling on imbalanced domains.* ACM Computing Surveys, 49(2), 1-50.
   - Técnicas para manejar datasets desbalanceados

10. **He, H., & Garcia, E. A.** (2009). *Learning from imbalanced data.* IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
    - Comparativa de métodos: SMOTE vs. Cost-Sensitive Learning

### Electrofisiología Cardíaca

11. **Surawicz, B., & Knilans, T. K.** (2008). *Chou's Electrocardiography in Clinical Practice: Adult and Pediatric.* Elsevier Health Sciences.
    - Tratado de referencia en electrocardiografía clínica

12. **Zipes, D. P., et al.** (2018). *Braunwald's Heart Disease: A Textbook of Cardiovascular Medicine.* Elsevier.
    - Fundamentos de arritmias cardíacas

> **Nota**: Las explicaciones médicas en este README están basadas en conocimiento médico establecido y consensuado en la literatura cardiológica, accesible a través de bases de datos como PubMed (https://pubmed.ncbi.nlm.nih.gov/).

---

## ⚠️ Descargo de Responsabilidad Médica

**IMPORTANTE**: Este proyecto es estrictamente con fines **educativos, de investigación y demostración técnica**.

- **NO está destinado para uso clínico real**
- **NO debe usarse para diagnóstico médico**
- **NO reemplaza el criterio de profesionales de la salud**

### Limitaciones

- El modelo está entrenado únicamente con el dataset MIT-BIH, que puede no representar toda la variabilidad poblacional
- No ha sido validado clínicamente ni aprobado por organismos regulatorios (FDA, CE, AEMPS)
- Los resultados deben ser siempre interpretados por médicos cualificados
- Las decisiones médicas requieren contexto clínico completo, no solo análisis de latidos aislados

### Uso Responsable

Si este código se adapta para aplicaciones médicas reales:

1. Se requiere **validación clínica exhaustiva** con conjuntos de datos independientes
2. Es **obligatorio cumplir** con regulaciones médicas (FDA 21 CFR Part 820, EU MDR 2017/745)
3. Debe obtenerse **certificación como dispositivo médico**
4. Es esencial la **supervisión continua** de profesionales médicos

> **El autor no asume responsabilidad** por el uso inadecuado de este software en contextos clínicos.

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👨‍💻 Autor

**Gonzalo Robert Maud Gallego**

- Hugging Face: [@GonzaloMaud](https://huggingface.co/GonzaloMaud)
- LinkedIn: Gonzalo Robert Maud Gallego
- GitHub: [@GonzaloMaud](https://github.com/GonzaloMaud)

---

<div align="center">

**Si este proyecto te resultó útil, considera darle una estrella en GitHub**

---

**Hecho con dedicación para la comunidad de salud digital**

*"En medicina, es mejor tener 10 falsas alarmas que 1 muerte por no detectar una arritmia"*

[![Modelo v1](https://img.shields.io/badge/🤗-Demo%20v1%20Clásico-blue?style=for-the-badge)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmias)
[![Modelo v2](https://img.shields.io/badge/🤗-Demo%20v2%20Robusto-green?style=for-the-badge)](https://huggingface.co/spaces/GonzaloMaud/Detector-Arritmiasv2)

</div>
