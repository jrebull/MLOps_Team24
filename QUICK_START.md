# 🚀 QUICK START: Análisis de Problema "Angry"

## 📦 Paquete Completo Creado

He creado un **sistema completo de análisis profesional** para diagnosticar por qué la clase "Angry" falla en inference. Este paquete incluye:

### 📄 **4 Scripts de Análisis Especializados:**
1. `analyze_1_dataset_distribution.py` - Distribución y balance de datos
2. `analyze_2_confusion_matrix.py` - Patrones de error del modelo
3. `analyze_3_sample_audio_features.py` - Comparación sample_audio vs training
4. `analyze_4_feature_importance.py` - Features discriminativas para "angry"

### 🎛️ **1 Script Maestro:**
- `run_complete_analysis.py` - Ejecuta todos los análisis en secuencia

### 📚 **3 Documentos de Referencia:**
1. `README_ANALYSIS.md` - Guía completa de uso y metodología
2. `DIAGNOSTIC_CHECKLIST.md` - Checklist de debugging y 13 soluciones catalogadas
3. Este documento - Quick start

---

## ⚡ INICIO RÁPIDO (3 pasos)

### **Paso 1: Descargar los archivos**

```bash
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24

# Copiar los archivos que te proporcioné a tu directorio
# Los archivos están en:
# - analyze_1_dataset_distribution.py
# - analyze_2_confusion_matrix.py
# - analyze_3_sample_audio_features.py
# - analyze_4_feature_importance.py
# - run_complete_analysis.py
# - README_ANALYSIS.md
# - DIAGNOSTIC_CHECKLIST.md
```

### **Paso 2: Activar entorno**

```bash
source .venv/bin/activate
```

### **Paso 3: Ejecutar análisis**

```bash
# Opción recomendada: Análisis completo
python3 run_complete_analysis.py

# O ejecutar scripts individuales:
python3 analyze_1_dataset_distribution.py
python3 analyze_2_confusion_matrix.py
python3 analyze_3_sample_audio_features.py
python3 analyze_4_feature_importance.py
```

---

## 🎯 ¿Qué vas a obtener?

### **Outputs Generados:**
- ✅ `confusion_matrix_angry_analysis.png` - Visualización de confusion matrix
- ✅ `sample_audio_features_analysis.csv` - Features de sample_audio
- ✅ `feature_importance_analysis.csv` - Feature importance del modelo

### **Análisis Completo de:**
1. ✅ **Distribución de clases** - ¿Hay desbalance en angry?
2. ✅ **Patrones de confusión** - ¿Con qué se confunde angry?
3. ✅ **Distribution drift** - ¿Sample audio es representativo?
4. ✅ **Feature discriminability** - ¿Qué features identifican angry?

---

## 🔍 Interpretación de Resultados

### **Si Script 1 muestra:**
- 🔴 **Balance ratio < 0.70** → Dataset desbalanceado, aplicar SOLUTION 1
- 🔴 **NaN values > 0** → Problemas de calidad, aplicar SOLUTION 3
- 🟢 **Balance ratio > 0.70** → Dataset OK, continuar

### **Si Script 2 muestra:**
- 🔴 **Angry accuracy < 75%** → Modelo mal entrenado, aplicar SOLUTION 4
- 🔴 **Confusión >30% con una emoción** → Overlap semántico, aplicar SOLUTION 5
- 🟢 **Angry accuracy > 85%** → Modelo OK, problema está en inference

### **Si Script 3 muestra:**
- 🔴 **Z-score > 2.0 en >5 features** → Distribution drift, aplicar SOLUTION 7
- 🔴 **Accuracy sample_audio < 50%** → Sample no representativo, aplicar SOLUTION 8
- 🟢 **Z-scores normales** → Features consistentes

### **Si Script 4 muestra:**
- 🔴 **Cohen's d < 0.3 en top features** → Features no discriminativas, aplicar SOLUTION 10
- 🔴 **Outliers en >5 features** → Datos anómalos, aplicar SOLUTION 11
- 🟢 **Cohen's d > 0.5** → Features discriminativas OK

---

## 📊 Flowchart de Decisión Rápida

```
┌─────────────────────────────────────────┐
│ 1. Ejecutar run_complete_analysis.py   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 2. Revisar outputs de los 4 scripts    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 3. Identificar problema más crítico    │
│    (usar DIAGNOSTIC_CHECKLIST.md)      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 4. Aplicar solución correspondiente     │
│    (13 soluciones catalogadas)         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 5. Re-entrenar y validar mejora         │
└─────────────────────────────────────────┘
```

---

## 🎓 Metodología MLOps

Este análisis sigue **best practices** de MLOps debugging:

### **Data-First Approach:**
- ✅ Verificar calidad y distribución de datos ANTES de ajustar modelo
- ✅ Identificar data drift entre train/test/inference
- ✅ Validar labels y preprocessing

### **Model Interpretability:**
- ✅ Confusion matrix para patrones de error
- ✅ Feature importance para entender decisiones
- ✅ Cohen's d para medir discriminabilidad

### **Reproducibilidad:**
- ✅ Scripts automatizados y reproducibles
- ✅ Documentación completa de cada paso
- ✅ Outputs guardados para referencia

---

## 📚 Documentación Completa

### **Para entender la metodología:**
→ Lee `README_ANALYSIS.md`

### **Para diagnosticar y solucionar:**
→ Usa `DIAGNOSTIC_CHECKLIST.md` con árbol de decisión

### **Para ejecutar:**
→ Sigue las instrucciones en este documento

---

## 🤝 Compartir con el Equipo

**Después de ejecutar los análisis:**

1. **Crear documento de hallazgos:**
```markdown
# Hallazgos: Análisis Angry Classification

## Ejecutado por: [Tu nombre]
## Fecha: [Fecha]

### Problema identificado:
[Resumen del problema principal]

### Evidencia:
- Script 1: [Hallazgo]
- Script 2: [Hallazgo]
- Script 3: [Hallazgo]
- Script 4: [Hallazgo]

### Solución propuesta:
[SOLUTION X del checklist]

### Plan de acción:
1. [Paso 1]
2. [Paso 2]
3. [Paso 3]
```

2. **Compartir archivos generados:**
```bash
# Copiar outputs a carpeta compartida
mkdir -p analysis_results_$(date +%Y%m%d)
cp confusion_matrix_angry_analysis.png analysis_results_*/
cp sample_audio_features_analysis.csv analysis_results_*/
cp feature_importance_analysis.csv analysis_results_*/
```

3. **Agendar reunión de equipo** para discutir hallazgos y decidir estrategia

---

## 🚨 Troubleshooting

### **Error: No se encuentra el dataset**
```bash
# Verificar path
ls data/processed/turkish_music_emotion_v2_cleaned_full.csv
```

### **Error: No se puede importar acoustic_ml**
```bash
# Verificar instalación
pip list | grep acoustic-ml

# Si no está instalado:
pip install -e .
```

### **Error: MLflow no encuentra el modelo**
```bash
# Verificar run_id
mlflow ui
# Navegar a http://localhost:5000 y buscar run_id
```

### **Error: No se pueden extraer features de audio**
```bash
# Verificar librosa instalado
pip install librosa

# Verificar permisos de archivos
ls -la turkish_music_app/assets/sample_audio/angry/
```

---

## 🎯 Objetivos del Análisis

### **Objetivo Principal:**
Entender por qué "Angry" tiene 86% accuracy en test set pero falla en inference con sample_audio

### **Preguntas a Responder:**
1. ❓ ¿Es un problema de **datos** (desbalance, calidad)?
2. ❓ ¿Es un problema de **modelo** (underfitting, overfitting)?
3. ❓ ¿Es un problema de **features** (preprocessing, extractio)?
4. ❓ ¿Es un problema de **labels** (etiquetado incorrecto)?

### **Resultado Esperado:**
✅ Diagnóstico claro con evidencia cuantitativa
✅ Solución específica catalogada
✅ Plan de acción para implementar fix

---

## ⏱️ Tiempo Estimado

- **Análisis completo:** ~10-15 minutos
- **Interpretación de resultados:** ~20-30 minutos
- **Implementación de solución:** Variable (30 min - 2 horas)

---

## 📧 Siguiente Paso

**Después de obtener resultados:**

1. Revisa los 4 outputs generados
2. Usa `DIAGNOSTIC_CHECKLIST.md` para identificar el problema
3. Implementa la solución correspondiente
4. Re-entrena y valida mejora
5. Documenta todo para Phase 3

---

## 🎉 ¡Éxito!

Este análisis te dará claridad sobre el problema y un camino claro hacia la solución. Los scripts están diseñados siguiendo MLOps best practices y son completamente reproducibles.

**¿Listo para empezar?**

```bash
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24
source .venv/bin/activate
python3 run_complete_analysis.py
```

---

**MLOps Team 24** 🚀
**Turkish Music Emotion Recognition Project**
