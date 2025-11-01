# 📦 Angry Analysis Package - Índice Completo

```
angry_analysis_package/
│
├── 🚀 QUICK_START.md                    ← EMPIEZA AQUÍ
│   └── Instrucciones rápidas de uso (3 pasos)
│
├── 📚 README_ANALYSIS.md                 ← Guía completa
│   ├── Contexto del problema
│   ├── Hipótesis a investigar
│   ├── Descripción de scripts
│   ├── Instrucciones de uso
│   └── Metodología MLOps
│
├── 🩺 DIAGNOSTIC_CHECKLIST.md            ← Soluciones catalogadas
│   ├── Árbol de decisión para debugging
│   ├── 13 soluciones catalogadas
│   ├── Priority matrix
│   └── Templates de documentación
│
├── ⚙️ setup_analysis.sh                  ← Script de instalación
│   └── Verifica dependencias y configura entorno
│
├── 🎯 run_complete_analysis.py           ← Script maestro
│   └── Ejecuta los 4 análisis en secuencia
│
└── 🔬 Scripts de Análisis Individual:
    │
    ├── 📊 analyze_1_dataset_distribution.py
    │   ├── Distribución de clases
    │   ├── Métricas de balance
    │   ├── Estadísticas por clase
    │   └── Verificación de integridad
    │
    ├── 🎯 analyze_2_confusion_matrix.py
    │   ├── Confusion matrix (absoluta y normalizada)
    │   ├── Análisis de errores por clase
    │   ├── Focus específico en "Angry"
    │   ├── Classification report
    │   └── Visualización (confusion_matrix_angry_analysis.png)
    │
    ├── 🎵 analyze_3_sample_audio_features.py
    │   ├── Extracción de features de sample_audio/
    │   ├── Comparación con training set (Z-scores)
    │   ├── Identificación de distribution drift
    │   ├── Predicciones en sample audios
    │   └── Output: sample_audio_features_analysis.csv
    │
    └── 🌳 analyze_4_feature_importance.py
        ├── Feature importance del Random Forest
        ├── Análisis de features para "Angry"
        ├── Cohen's d para discriminabilidad
        ├── Comparación Angry vs cada emoción
        ├── Detección de outliers
        └── Output: feature_importance_analysis.csv
```

---

## 📊 Workflow de Análisis

```
┌─────────────────────────────────────────────────────────────┐
│                    INICIO DEL ANÁLISIS                      │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │   setup_analysis.sh           │
         │   (Verificar entorno)         │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   run_complete_analysis.py    │
         │   (Script maestro)            │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────┐                     ┌───▼────┐
    │ Script 1│                     │Script 2│
    │ Dataset │                     │Confusion│
    │  Dist.  │                     │ Matrix │
    └────┬────┘                     └───┬────┘
         │                              │
         └──────────────┬───────────────┘
                        │
         ┌──────────────┴───────────────┐
         │                              │
    ┌────▼────┐                    ┌───▼────┐
    │ Script 3│                    │Script 4│
    │ Sample  │                    │Feature │
    │ Audio   │                    │Import. │
    └────┬────┘                    └───┬────┘
         │                              │
         └──────────────┬───────────────┘
                        │
         ┌──────────────▼───────────────┐
         │       OUTPUTS GENERADOS       │
         ├───────────────────────────────┤
         │ • confusion_matrix_*.png      │
         │ • sample_audio_features.csv   │
         │ • feature_importance.csv      │
         └──────────────┬────────────────┘
                        │
         ┌──────────────▼───────────────┐
         │   DIAGNOSTIC_CHECKLIST.md    │
         │   (Identificar problema)     │
         └──────────────┬────────────────┘
                        │
         ┌──────────────▼───────────────┐
         │   Aplicar Solución (1-13)    │
         └──────────────┬────────────────┘
                        │
         ┌──────────────▼───────────────┐
         │    Re-entrenar y Validar     │
         └───────────────────────────────┘
```

---

## 🎯 Guía de Lectura por Objetivo

### **Si quieres EMPEZAR RÁPIDO:**
1. `QUICK_START.md` (3 pasos para ejecutar)
2. Ejecutar: `python3 run_complete_analysis.py`
3. Revisar outputs generados

### **Si quieres ENTENDER la METODOLOGÍA:**
1. `README_ANALYSIS.md` (guía completa)
2. Sección "Metodología de Análisis"
3. Sección "Interpretación de Resultados"

### **Si quieres SOLUCIONAR el PROBLEMA:**
1. Ejecutar análisis primero
2. `DIAGNOSTIC_CHECKLIST.md` (árbol de decisión)
3. Identificar problema en Decision Tree
4. Aplicar solución correspondiente (1-13)

### **Si quieres COMPARTIR con el EQUIPO:**
1. Ejecutar análisis completo
2. Crear documento de hallazgos (template en DIAGNOSTIC_CHECKLIST.md)
3. Copiar outputs generados (.png, .csv)
4. Usar README_ANALYSIS.md como referencia

---

## 📦 Archivos de Entrada (requeridos)

```
Tu proyecto/
├── data/processed/
│   └── turkish_music_emotion_v2_cleaned_full.csv  ← REQUERIDO
│
├── turkish_music_app/assets/sample_audio/         ← REQUERIDO para Script 3
│   ├── angry/*.mp3
│   ├── happy/*.mp3
│   ├── sad/*.mp3
│   └── relax/*.mp3
│
├── mlruns/                                         ← REQUERIDO para Scripts 2, 3, 4
│   └── [experiment_id]/
│       └── eb05c7698f12499b86ed35ca6efc15a7/     ← Run ID actual
│
└── acoustic_ml/                                    ← REQUERIDO (package)
    └── features/
        └── audio_features.py
```

---

## 📤 Archivos de Salida (generados)

```
Tu proyecto/
├── confusion_matrix_angry_analysis.png             ← Visualización
├── sample_audio_features_analysis.csv              ← Features extraídos
├── feature_importance_analysis.csv                 ← Feature importance
│
└── (logs en terminal con análisis detallado)
```

---

## 🔧 Dependencias

### **Python Packages:**
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Modelo y métricas
- `mlflow` - Experiment tracking
- `librosa` - Extracción de features de audio
- `matplotlib` - Visualización
- `seaborn` - Visualización avanzada

### **Custom Package:**
- `acoustic_ml` - Pipeline de features (tu paquete)

---

## 🚀 Comando Único de Instalación

```bash
# Desde el directorio raíz del proyecto
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24

# Copiar archivos del paquete aquí, luego:
chmod +x setup_analysis.sh
./setup_analysis.sh

# Ejecutar análisis
python3 run_complete_analysis.py
```

---

## 📝 Checklist de Uso

### **Pre-Análisis:**
- [ ] Estás en el directorio correcto del proyecto
- [ ] Virtual environment activado
- [ ] Dependencias instaladas
- [ ] Dataset y sample_audio disponibles

### **Durante Análisis:**
- [ ] Script 1 ejecutado sin errores
- [ ] Script 2 ejecutado sin errores
- [ ] Script 3 ejecutado sin errores
- [ ] Script 4 ejecutado sin errores
- [ ] Outputs generados (.png, .csv)

### **Post-Análisis:**
- [ ] Revisaste outputs de cada script
- [ ] Identificaste problema principal
- [ ] Consultaste DIAGNOSTIC_CHECKLIST.md
- [ ] Seleccionaste solución apropiada
- [ ] Documentaste hallazgos

---

## 🎓 Conceptos Clave

### **Balance Ratio:**
`min_class_count / max_class_count`
- > 0.70 = Balanceado ✅
- < 0.70 = Desbalanceado ⚠️

### **Z-score:**
`(sample_value - train_mean) / train_std`
- > 2.0 = Fuera de distribución ⚠️
- < 2.0 = Dentro de distribución ✅

### **Cohen's d:**
`(mean_angry - mean_others) / pooled_std`
- > 0.8 = Diferencia grande ✅
- 0.5-0.8 = Diferencia mediana ⚠️
- < 0.5 = Diferencia pequeña ❌

---

## 🆘 Soporte

**Si algo no funciona:**
1. Revisa `setup_analysis.sh` output para errores
2. Verifica que todos los archivos de entrada existen
3. Consulta sección "Troubleshooting" en QUICK_START.md
4. Ejecuta scripts individuales para identificar cuál falla

**Para preguntas:**
- Revisa README_ANALYSIS.md
- Consulta DIAGNOSTIC_CHECKLIST.md
- Contacta al equipo MLOps Team 24

---

## 📊 Tiempo Estimado Total

| Actividad | Tiempo |
|-----------|--------|
| Setup (primera vez) | 5-10 min |
| Ejecutar análisis completo | 10-15 min |
| Interpretar resultados | 20-30 min |
| Seleccionar solución | 10 min |
| Implementar fix | 30 min - 2 hrs |
| **TOTAL** | **1.5 - 3 horas** |

---

## 🎉 Resultado Final Esperado

Después de usar este paquete, tendrás:

✅ **Diagnóstico claro** del problema de "Angry" con evidencia cuantitativa
✅ **Solución específica** identificada del catálogo de 13 soluciones
✅ **Outputs visuales y datos** para compartir con el equipo
✅ **Plan de acción** documentado para implementar fix
✅ **Metodología MLOps** profesional aplicada

---

**¡Éxito con tu análisis!** 🚀

**MLOps Team 24**
*Turkish Music Emotion Recognition Project*
