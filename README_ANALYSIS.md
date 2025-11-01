# 🔍 Análisis Completo: Problema de Clasificación "Angry"

## 📋 Contexto

El modelo de Turkish Music Emotion Recognition tiene un problema específico con la clase **"Angry"**:
- ✅ Test accuracy de "angry": **86%** (bastante bueno)
- ❌ Predicciones en audios reales de sample_audio: **Fallan consistentemente**

**Observaciones:**
```
✅ sad/harman_yeri_surseler_salih_gundogdu.mp3 → Predicción: sad (correcto)
✅ relax/fikret_kizilok_gonul.mp3 → Predicción: relax (correcto)
❌ happy/gir_kanima_harun_kolcak.mp3 → Predicción: angry (incorrecto, esperado: happy)
❌ angry/adanali.mp3 → Predicción: relax (incorrecto, esperado: angry)
```

Este análisis investiga las **causas raíz** de este problema.

---

## 🎯 Hipótesis a Investigar

### 1️⃣ **Dataset Imbalance**
- ¿Hay suficientes samples de "angry" en training?
- ¿Está desbalanceado respecto a otras clases?

### 2️⃣ **Feature Distribution Mismatch**
- ¿Los audios de sample_audio tienen features consistentes con training?
- ¿Hay "distribution drift"?

### 3️⃣ **Confusion Patterns**
- ¿Con qué emociones se confunde "angry" más frecuentemente?
- ¿Hay overlap en feature space?

### 4️⃣ **Feature Discriminability**
- ¿Qué features son críticas para identificar "angry"?
- ¿Son suficientemente discriminativas?

---

## 🛠️ Scripts de Análisis

Se crearon 4 scripts especializados (siguiendo MLOps best practices):

### **Script 1: `analyze_1_dataset_distribution.py`**
**Objetivo:** Analizar distribución de clases y balance del dataset

**Output:**
- Conteo de samples por clase
- Métricas de balance (min/max ratio)
- Estadísticas de features por clase
- Verificación de NaN values y duplicados

### **Script 2: `analyze_2_confusion_matrix.py`**
**Objetivo:** Generar confusion matrix detallada y analizar patrones de error

**Output:**
- Confusion matrix absoluta y normalizada
- Análisis de errores por clase
- **Focus específico en "Angry"**: Con qué se confunde y qué se confunde como angry
- Classification report completo
- Visualización (confusion_matrix_angry_analysis.png)

### **Script 3: `analyze_3_sample_audio_features.py`**
**Objetivo:** Extraer features de sample_audio y comparar con training set

**Output:**
- Features extraídos de todos los audios en sample_audio/
- Comparación estadística (mean, std, Z-scores) con training set
- Identificación de features con distribución diferente
- Predicciones del modelo en sample audios
- Accuracy por emoción en sample_audio
- CSV con features (sample_audio_features_analysis.csv)

### **Script 4: `analyze_4_feature_importance.py`**
**Objetivo:** Analizar qué features son importantes para clasificar "angry"

**Output:**
- Feature importance del Random Forest
- Distribución de top features para "angry" vs otras clases
- Cohen's d (effect size) para medir discriminabilidad
- Comparación "angry" vs cada emoción individual
- Detección de outliers en samples "angry"
- CSV con feature importance (feature_importance_analysis.csv)

---

## 🚀 Instrucciones de Uso

### **Preparación:**

1. **Navegar al directorio del proyecto:**
```bash
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24
```

2. **Activar entorno virtual:**
```bash
source .venv/bin/activate  # macOS/Linux
```

3. **Verificar que tienes los datos:**
```bash
ls data/processed/turkish_music_emotion_v2_cleaned_full.csv
ls turkish_music_app/assets/sample_audio/
```

---

### **Opción A: Ejecutar Análisis Completo (RECOMENDADO)**

```bash
python3 run_complete_analysis.py
```

Esto ejecutará los 4 scripts en secuencia y generará un reporte consolidado.

---

### **Opción B: Ejecutar Scripts Individuales**

```bash
# Script 1: Dataset Distribution
python3 analyze_1_dataset_distribution.py

# Script 2: Confusion Matrix
python3 analyze_2_confusion_matrix.py

# Script 3: Sample Audio Features
python3 analyze_3_sample_audio_features.py

# Script 4: Feature Importance
python3 analyze_4_feature_importance.py
```

---

## 📊 Archivos Generados

Después de ejecutar los análisis, tendrás:

```
confusion_matrix_angry_analysis.png     # Visualización de confusion matrix
sample_audio_features_analysis.csv      # Features de sample_audio
feature_importance_analysis.csv         # Feature importance del modelo
```

---

## 🔬 Metodología de Análisis

Este análisis sigue **MLOps best practices** para debugging de modelos:

### **1. Data-Centric Approach**
- Primero verificamos la calidad y distribución de los datos
- Identificamos desbalances, outliers, NaN values

### **2. Model-Centric Analysis**
- Analizamos cómo el modelo está clasificando (confusion matrix)
- Identificamos patrones de error sistemáticos

### **3. Feature Engineering Validation**
- Verificamos que las features extraídas sean consistentes
- Comparamos training vs inference features

### **4. Interpretability**
- Usamos feature importance para entender decisiones del modelo
- Medimos discriminabilidad con Cohen's d

---

## 🎯 Interpretación de Resultados

### **Dataset Distribution (Script 1)**

**Si encuentras:**
- Balance ratio < 70% → ⚠️ Dataset desbalanceado
- NaN values > 0 → ⚠️ Problemas de calidad de datos
- Una clase tiene < 50 samples → ⚠️ Insuficientes datos

**Acción:** Considerar rebalanceo, limpieza, o recolección de más datos

---

### **Confusion Matrix (Script 2)**

**Si encuentras:**
- "Angry" se confunde >20% con una emoción específica → ⚠️ Overlap semántico
- Accuracy de "angry" < 70% → ⚠️ Problema grave
- Patrones asimétricos (A→B pero no B→A) → 🔍 Investigar features

**Acción:** Identificar qué causa la confusión específica

---

### **Sample Audio Features (Script 3)**

**Si encuentras:**
- Z-score > 2.0 en >5 features → ⚠️ Distribution drift significativo
- Accuracy en sample_audio << test accuracy → ⚠️ Sample audio no representativo
- Todas las emociones fallan excepto angry → 🔍 Problema de preprocessing

**Acción:** 
- Verificar que AudioFeatureExtractor usa mismos parámetros que training
- Revisar si sample_audio es representativo
- Re-extraer features con configuración consistente

---

### **Feature Importance (Script 4)**

**Si encuentras:**
- Cohen's d < 0.3 para top features → ⚠️ Features poco discriminativas
- "Angry" tiene alta varianza en features importantes → ⚠️ Clase heterogénea
- Top features tienen outliers en angry → ⚠️ Problemas de calidad

**Acción:**
- Considerar feature engineering adicional
- Ajustar hyperparámetros del modelo
- Limpiar outliers si son errores de extracción

---

## 📈 Plan de Acción Post-Análisis

Basado en los hallazgos, el siguiente flujo de decisión:

```
┌─────────────────────────────────┐
│ ¿Dataset desbalanceado?         │
│ (Script 1)                      │
└───────┬─────────────────────────┘
        │
        ├─ SÍ → Rebalancear con SMOTE / class_weight
        │
        └─ NO → Continuar
                    │
        ┌───────────▼─────────────────────┐
        │ ¿Sample audio no representativo?│
        │ (Script 3)                      │
        └───────┬─────────────────────────┘
                │
                ├─ SÍ → Probar con más audios / revisar labels
                │
                └─ NO → Continuar
                            │
                ┌───────────▼─────────────────────┐
                │ ¿Features poco discriminativas? │
                │ (Script 4)                      │
                └───────┬─────────────────────────┘
                        │
                        ├─ SÍ → Feature engineering / cambiar modelo
                        │
                        └─ NO → Ajustar hyperparámetros
```

---

## ⚙️ Configuración Técnica

**Requisitos:**
- Python 3.12+
- scikit-learn 1.7.2
- pandas, numpy, matplotlib, seaborn
- MLflow
- acoustic_ml package instalado

**MLflow Run ID:**
- Modelo actual: `eb05c7698f12499b86ed35ca6efc15a7`
- Test accuracy: 84.30%

**Dataset:**
- Path: `data/processed/turkish_music_emotion_v2_cleaned_full.csv`
- Total samples: 403 (después de limpieza)
- Features: 50 acoustic features

---

## 📝 Notas Adicionales

### **Sobre Cohen's d (Effect Size):**
- **d < 0.2**: Diferencia pequeña
- **d = 0.5**: Diferencia mediana
- **d > 0.8**: Diferencia grande

Para clasificación, queremos **d > 0.5** en features importantes.

### **Sobre Z-scores:**
- **Z > 2.0**: Valor a más de 2 desviaciones estándar del mean
- Indica que sample_audio tiene features "fuera de lo normal"

### **Sobre Confusion Matrix:**
- Diagonal = Predicciones correctas
- Off-diagonal = Errores
- Normalización por fila muestra "recall" por clase

---

## 🤝 Siguientes Pasos para el Equipo

1. **Ejecutar análisis** y compartir resultados en el grupo
2. **Documentar hallazgos** en un documento colaborativo
3. **Decidir estrategia** de corrección basada en evidencia
4. **Implementar cambios** siguiendo MLOps workflow (DVC + MLflow)
5. **Re-entrenar y validar** mejoras

---

## 📧 Contacto

**MLOps Team 24:**
- David Cruz Beltrán (Software Engineer)
- Javier Augusto Rebull Saucedo (SRE/Data Engineer)
- Sandra Luz Cervantes Espinoza (ML Engineer/Data Scientist)

**Proyecto:** Turkish Music Emotion Recognition
**Fase:** 2 (MLOps Implementation)
**Próximo deadline:** Fase 3 (Production Deployment)

---

**¡Éxito con el análisis!** 🚀
