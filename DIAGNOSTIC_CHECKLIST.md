# 🩺 Diagnostic Checklist & Solutions: "Angry" Classification Problem

## 📋 Pre-Analysis Checklist

Antes de ejecutar los scripts, verificar:

- [ ] Estás en el directorio correcto del proyecto
- [ ] Virtual environment activado (`.venv`)
- [ ] Dataset existe: `data/processed/turkish_music_emotion_v2_cleaned_full.csv`
- [ ] Sample audios existen: `turkish_music_app/assets/sample_audio/`
- [ ] MLflow tracking URI configurado correctamente
- [ ] Modelo está cargado en MLflow (run_id: `eb05c7698f12499b86ed35ca6efc15a7`)

---

## 🔬 Diagnostic Decision Tree

Sigue este árbol de decisión después de ejecutar los análisis:

```
START: ¿Por qué "Angry" falla en inference?
    │
    ├─► SCRIPT 1: Dataset Distribution Analysis
    │   │
    │   ├─ ¿Balance ratio < 0.70?
    │   │  └─► YES → [SOLUTION 1: Rebalanceo de Dataset]
    │   │  └─► NO  → Continuar
    │   │
    │   ├─ ¿"Angry" tiene < 60 samples?
    │   │  └─► YES → [SOLUTION 2: Recolección de más datos]
    │   │  └─► NO  → Continuar
    │   │
    │   └─ ¿Hay NaN values en "angry" samples?
    │      └─► YES → [SOLUTION 3: Limpieza de datos]
    │      └─► NO  → Continuar
    │
    ├─► SCRIPT 2: Confusion Matrix Analysis
    │   │
    │   ├─ ¿"Angry" accuracy < 75%?
    │   │  └─► YES → [SOLUTION 4: Re-entrenamiento con hyperparameters]
    │   │  └─► NO  → Continuar
    │   │
    │   ├─ ¿"Angry" se confunde >30% con una emoción?
    │   │  └─► YES → [SOLUTION 5: Feature engineering específico]
    │   │  └─► NO  → Continuar
    │   │
    │   └─ ¿Patrón asimétrico (angry→X pero no X→angry)?
    │      └─► YES → [SOLUTION 6: Revisar labels]
    │      └─► NO  → Continuar
    │
    ├─► SCRIPT 3: Sample Audio Features Analysis
    │   │
    │   ├─ ¿>5 features con Z-score > 2.0?
    │   │  └─► YES → [SOLUTION 7: Verificar preprocessing consistency]
    │   │  └─► NO  → Continuar
    │   │
    │   ├─ ¿Accuracy en sample_audio < 50%?
    │   │  └─► YES → [SOLUTION 8: Sample audio no representativo]
    │   │  └─► NO  → Continuar
    │   │
    │   └─ ¿Solo "angry" falla (otras emociones OK)?
    │      └─► YES → [SOLUTION 9: Problema específico de angry labels]
    │      └─► NO  → Continuar
    │
    └─► SCRIPT 4: Feature Importance Analysis
        │
        ├─ ¿Top features tienen Cohen's d < 0.3?
        │  └─► YES → [SOLUTION 10: Features no discriminativas]
        │  └─► NO  → Continuar
        │
        ├─ ¿"Angry" tiene outliers en >5 top features?
        │  └─► YES → [SOLUTION 11: Remover outliers]
        │  └─► NO  → Continuar
        │
        └─ ¿Varianza muy baja en top features?
           └─► YES → [SOLUTION 12: Agregar features con más varianza]
           └─► NO  → [SOLUTION 13: Problema en inference pipeline]
```

---

## 💊 Solutions Catalog

### **SOLUTION 1: Rebalanceo de Dataset**

**Problema:** Dataset desbalanceado (angry tiene muchos menos samples que otras clases)

**Diagnóstico confirmado si:**
- Balance ratio < 0.70
- "Angry" tiene significativamente menos samples

**Implementación:**

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Opción A: SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Opción B: Class weights en Random Forest
rf = RandomForestClassifier(
    class_weight='balanced',  # Ajusta automáticamente
    random_state=42
)

# Opción C: Stratified sampling + class_weight manual
class_weights = {
    'angry': 2.0,   # Mayor peso
    'happy': 1.0,
    'sad': 1.0,
    'relax': 1.0
}
```

**Script de implementación:**
```bash
python3 scripts/retrain_with_balanced_data.py
```

---

### **SOLUTION 2: Recolección de más datos**

**Problema:** Insuficientes samples de "angry" para entrenar adecuadamente

**Diagnóstico confirmado si:**
- "Angry" tiene < 60 samples
- Alta varianza en métricas

**Implementación:**
1. Buscar más audios de música turca con emoción "angry"
2. Usar data augmentation:
   ```python
   from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
   
   augment = Compose([
       AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
       TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
       PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
   ])
   
   # Aplicar a samples de angry
   for audio_file in angry_samples:
       augmented = augment(audio, sample_rate=22050)
   ```

---

### **SOLUTION 3: Limpieza de datos**

**Problema:** Hay NaN values o datos corruptos en samples "angry"

**Diagnóstico confirmado si:**
- NaN values detectados
- Features con valores anómalos (inf, -inf)

**Implementación:**
```python
# Remover filas con NaN
df_clean = df[df['Class'] == 'angry'].dropna()

# Remover outliers extremos (>4 std)
for col in feature_columns:
    mean = df_clean[col].mean()
    std = df_clean[col].std()
    df_clean = df_clean[
        (df_clean[col] > mean - 4*std) & 
        (df_clean[col] < mean + 4*std)
    ]

# Re-guardar dataset limpio
df_clean.to_csv('data/processed/turkish_music_emotion_v2_cleaned_full.csv')
```

---

### **SOLUTION 4: Re-entrenamiento con hyperparameters**

**Problema:** Modelo no está optimizado para clasificar "angry"

**Diagnóstico confirmado si:**
- Accuracy de "angry" < 75% en test set
- Otras clases tienen mejor performance

**Implementación:**
```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning específico
param_grid = {
    'random_forest__n_estimators': [200, 300, 500],
    'random_forest__max_depth': [10, 20, 30, None],
    'random_forest__min_samples_split': [2, 5, 10],
    'random_forest__min_samples_leaf': [1, 2, 4],
    'random_forest__class_weight': ['balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',  # Para datasets desbalanceados
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

### **SOLUTION 5: Feature engineering específico**

**Problema:** "Angry" se confunde consistentemente con otra emoción específica (e.g., "happy")

**Diagnóstico confirmado si:**
- Confusion matrix muestra >30% de confusión con una emoción
- Features actuales no discriminan bien

**Implementación:**
```python
# Agregar features específicas para angry vs happy
def extract_angry_specific_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    # Energy ratio (angry suele tener más energía)
    energy = np.sum(librosa.feature.rms(y=y)**2)
    
    # Spectral rolloff (angry tiene más high-frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    
    # Zero crossing rate (angry tiene más cambios bruscos)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    
    # Harmonic-to-noise ratio (angry puede tener más ruido)
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)
    
    return {
        'energy_ratio': energy,
        'spectral_rolloff_mean': rolloff,
        'zcr_mean': zcr,
        'harmonic_noise_ratio': hnr
    }
```

---

### **SOLUTION 6: Revisar labels**

**Problema:** Labels inconsistentes o incorrectos

**Diagnóstico confirmado si:**
- Patrón asimétrico (angry→relax frecuente, pero relax→angry raro)
- Listening test confirma que labels son cuestionables

**Implementación:**
1. Escuchar manualmente los audios más confundidos:
   ```bash
   # Identificar audios de angry clasificados como relax
   python3 scripts/identify_misclassified_samples.py
   ```

2. Re-etiquetar si es necesario:
   ```python
   # Manual relabeling
   relabel_map = {
       'sample_123.mp3': 'relax',  # Era angry, pero suena relax
       'sample_456.mp3': 'angry',  # Era relax, pero suena angry
   }
   ```

3. Usar voting ensemble para validar labels:
   ```python
   # 3 anotadores independientes
   # Solo usar samples donde 2+ acuerdan
   ```

---

### **SOLUTION 7: Verificar preprocessing consistency**

**Problema:** Features de sample_audio se extraen diferente que training

**Diagnóstico confirmado si:**
- Z-scores > 2.0 en múltiples features
- Distribution drift evidente

**Implementación:**
```python
# VERIFICAR: ¿Training usó estos parámetros?
extractor = AudioFeatureExtractor(
    sr=22050,          # Sample rate
    duration=30.0,     # Duration in seconds
    n_mfcc=13,        # Número de MFCCs
    hop_length=512,   # Hop length
    n_fft=2048        # FFT window
)

# IMPORTANTE: Debe ser EXACTAMENTE igual que en training
# Revisar logs de MLflow del run_id original:
mlflow.get_run("eb05c7698f12499b86ed35ca6efc15a7")
```

**Checklist de consistencia:**
- [ ] Sample rate igual (22050 Hz)
- [ ] Duration igual (30s)
- [ ] Mismo número de MFCCs (13)
- [ ] Mismo hop_length (512)
- [ ] Mismo n_fft (2048)
- [ ] Mismas transformaciones (PowerTransformer + RobustScaler)

---

### **SOLUTION 8: Sample audio no representativo**

**Problema:** Los audios en sample_audio/ no son del mismo "estilo" que training

**Diagnóstico confirmado si:**
- Accuracy en sample_audio << test accuracy
- Todas las emociones fallan (no solo angry)

**Implementación:**
1. **Probar con más audios:**
   ```bash
   # Agregar 10-20 audios por emoción a sample_audio/
   # Fuentes: dataset original, nuevos downloads
   ```

2. **Validar con cross-validation:**
   ```python
   from sklearn.model_selection import cross_val_score
   
   scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
   print(f"CV scores: {scores}")
   print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
   ```

3. **Test con audios del training set:**
   ```python
   # Copiar algunos audios originales del training a sample_audio/
   # Si el modelo clasifica bien estos, entonces sample_audio es el problema
   ```

---

### **SOLUTION 9: Problema específico de angry labels**

**Problema:** Solo "angry" falla, otras emociones se clasifican bien

**Diagnóstico confirmado si:**
- happy, sad, relax tienen >80% accuracy
- angry tiene <60% accuracy
- Solo en sample_audio (no en test set)

**Implementación:**
1. **Escuchar todos los audios de angry en sample_audio:**
   ```bash
   ls turkish_music_app/assets/sample_audio/angry/*.mp3
   # Escuchar cada uno y verificar manualmente
   ```

2. **Comparar con training samples de angry:**
   ```python
   # Extraer features de training angry samples
   df_train_angry = df[df['Class'] == 'angry']
   
   # Comparar con sample_audio angry
   # ¿Son realmente del mismo tipo de música?
   ```

3. **Posible solución: Re-definir qué es "angry":**
   - Música agresiva vs música enérgica
   - Angry en contexto turco puede ser diferente

---

### **SOLUTION 10: Features no discriminativas**

**Problema:** Features actuales no capturan bien la diferencia de "angry"

**Diagnóstico confirmado si:**
- Cohen's d < 0.3 para top features
- No hay features con alta discriminación para angry

**Implementación:**
```python
# Agregar nuevas features más discriminativas
from librosa.feature import spectral_contrast, tonnetz

def extract_additional_features(y, sr):
    # Spectral contrast (diferencia entre peaks y valleys)
    contrast = spectral_contrast(y=y, sr=sr)
    
    # Tonnetz (tonal centroid features)
    tonnetz_features = tonnetz(y=y, sr=sr)
    
    # Attack time (tiempo de inicio de notas)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Tempogram (tempo variations)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    
    return {
        'spectral_contrast_mean': contrast.mean(axis=1),
        'tonnetz_mean': tonnetz_features.mean(axis=1),
        'onset_strength': onset_env.mean(),
        'tempogram_mean': tempogram.mean()
    }
```

---

### **SOLUTION 11: Remover outliers**

**Problema:** Samples de "angry" con valores extremos en features

**Diagnóstico confirmado si:**
- >5 top features tienen outliers en angry samples
- Outliers distorsionan el modelo

**Implementación:**
```python
from sklearn.covariance import EllipticEnvelope

# Detectar outliers multivariados
outlier_detector = EllipticEnvelope(contamination=0.1)
predictions = outlier_detector.fit_predict(X_angry)

# Remover outliers
X_angry_clean = X_angry[predictions == 1]
y_angry_clean = y_angry[predictions == 1]

print(f"Removidos {sum(predictions == -1)} outliers de angry")
```

---

### **SOLUTION 12: Agregar features con más varianza**

**Problema:** Top features tienen muy poca varianza para "angry"

**Diagnóstico confirmado si:**
- Features importantes para angry tienen varianza < 0.001
- No hay suficiente información en las features

**Implementación:**
```python
# Identificar features con baja varianza
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_high_variance = selector.fit_transform(X_angry)

# Agregar features derivadas con más varianza
X_angry['mfcc_1_squared'] = X_angry['mfcc_1'] ** 2
X_angry['energy_log'] = np.log(X_angry['energy'] + 1)
X_angry['chroma_ratio'] = X_angry['chroma_1'] / (X_angry['chroma_2'] + 1e-10)
```

---

### **SOLUTION 13: Problema en inference pipeline**

**Problema:** El pipeline de inference no está funcionando igual que en training

**Diagnóstico confirmado si:**
- Test accuracy es bueno (84%)
- Sample audio accuracy es malo (<50%)
- No hay diferencias obvias en features o datos

**Implementación:**
```python
# Verificar que el pipeline completo se usa en inference
def predict_with_full_pipeline(audio_path, model, extractor):
    # 1. Extraer features (MISMO extractor que training)
    features = extractor.extract_features(audio_path)
    
    # 2. Convertir a DataFrame con MISMAS columnas
    X = pd.DataFrame([features])
    
    # 3. Asegurar orden de columnas
    X = X[model.feature_names_in_]
    
    # 4. Predicción
    prediction = model.predict(X)
    
    return prediction[0]

# Debugging: Imprimir features antes y después de transformación
X_raw = extractor.extract_features(audio_path)
X_transformed = model.named_steps['power_transformer'].transform([X_raw])
print(f"Raw features: {X_raw[:5]}")
print(f"Transformed features: {X_transformed[:5]}")
```

---

## 🎯 Priority Matrix

Basado en la frecuencia de problemas en proyectos similares:

| Solución | Probabilidad | Impacto | Esfuerzo | Prioridad |
|----------|--------------|---------|----------|-----------|
| SOLUTION 7: Preprocessing | Alta | Alto | Bajo | 🔴 ALTA |
| SOLUTION 8: Sample audio | Alta | Medio | Bajo | 🔴 ALTA |
| SOLUTION 4: Hyperparameters | Media | Alto | Medio | 🟡 MEDIA |
| SOLUTION 5: Feature engineering | Media | Alto | Alto | 🟡 MEDIA |
| SOLUTION 1: Rebalanceo | Baja | Medio | Medio | 🟢 BAJA |
| SOLUTION 2: Más datos | Baja | Alto | Alto | 🟢 BAJA |

**Recomendación:** Empezar por SOLUTION 7 y 8 (verificación básica), luego pasar a 4 y 5 si es necesario.

---

## 📊 Success Metrics

Después de aplicar una solución, medir:

- [ ] Test accuracy de "angry" > 85%
- [ ] Sample audio accuracy de "angry" > 75%
- [ ] Confusión con otras clases < 15%
- [ ] Cohen's d de top features > 0.5
- [ ] Consistency entre train/test/inference

---

## 📝 Documentation Template

Documentar cada solución intentada:

```markdown
## Solución Intentada: [NOMBRE]

**Fecha:** YYYY-MM-DD
**Ejecutado por:** [Nombre]

### Diagnóstico:
- [Hallazgos que llevaron a esta solución]

### Implementación:
- [Comandos ejecutados]
- [Código modificado]

### Resultados:
- Test accuracy antes: X%
- Test accuracy después: Y%
- Sample audio accuracy antes: A%
- Sample audio accuracy después: B%

### Conclusión:
✅ Solución efectiva / ❌ No resolvió el problema

### Próximos pasos:
- [Si no funcionó, qué intentar después]
```

---

## 🚀 Quick Reference: Command Cheatsheet

```bash
# Ejecutar análisis completo
python3 run_complete_analysis.py

# Scripts individuales
python3 analyze_1_dataset_distribution.py
python3 analyze_2_confusion_matrix.py
python3 analyze_3_sample_audio_features.py
python3 analyze_4_feature_importance.py

# Re-entrenar con solución específica
python3 scripts/retrain_with_balanced_data.py  # SOLUTION 1
python3 scripts/retrain_with_new_features.py   # SOLUTION 5
python3 scripts/retrain_with_tuned_params.py   # SOLUTION 4

# Validar modelo
python3 scripts/validate_model_consistency.py  # SOLUTION 7

# Exportar resultados
python3 scripts/export_analysis_report.py
```

---

**¡Debugging sistemático y basado en evidencia!** 🔬✨
