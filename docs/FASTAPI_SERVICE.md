# 🚀 FastAPI Service - Turkish Music Emotion Recognition

**Phase 3: Production Model Serving**

---

## 📊 Resumen

Servicio REST API que expone un modelo Random Forest (84.29% accuracy) para clasificación de emociones en música turca.

## 🏃 Inicio Rápido
```bash
# 1. Crear carpeta de logs
mkdir -p logs

# 2. Iniciar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## 📚 Endpoints

### 1. Health Check
**GET** `/api/v1/health`
```bash
curl -X GET "http://localhost:8001/api/v1/health"
```

### 2. Predicción
**POST** `/api/v1/predict`

Requiere exactamente **50 características acústicas**.
```bash
curl -X POST "http://localhost:8001/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, ..., 0.7]}'
```

**Response:**
```json
{
  "prediction": 0,
  "emotion": "Happy",
  "confidence": 0.87,
  "probabilities": [0.87, 0.05, 0.04, 0.04],
  "probabilities_mapped": {
    "Happy": 0.87,
    "Sad": 0.05,
    "Angry": 0.04,
    "Relax": 0.04
  },
  "timestamp": "2025-11-14T10:30:45.123456"
}
```

### 3. Model Info
**GET** `/api/v1/model-info`

Retorna metadata del modelo.

## 🔗 Documentación Interactiva

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
- OpenAPI: http://localhost:8001/openapi.json

## 🐍 Ejemplos

### Python
```python
import requests

url = "http://localhost:8001/api/v1/predict"
features = [0.5] * 50
response = requests.post(url, json={"features": features})
result = response.json()
print(f"Emoción: {result['emotion']}")
print(f"Confianza: {result['confidence']:.2%}")
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8001/api/v1/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({features: Array(50).fill(0.5)})
});
const result = await response.json();
console.log(`Emoción: ${result.emotion}`);
```

## 📋 Características Esperadas (50)

| Posición | Nombre | Descripción |
|----------|--------|-------------|
| 0 | RMSenergy_Mean | Energía RMS |
| 1 | Lowenergy_Mean | Energía baja |
| 2 | Fluctuation_Mean | Fluctuación |
| 3 | Tempo_Mean | Tempo |
| 4-16 | MFCC_Mean_1-13 | Coeficientes MFCC |
| 17-31 | Spectral Features | Características espectrales |
| 32-43 | Chromagram_Mean_1-12 | Cromatograma |
| 44-49 | HCDF Features | Harmonic Change Detection |

## ⚙️ Configuración

Environment: `app/core/config.py`
- MODEL_PATH: Ruta del modelo
- LOG_LEVEL: Nivel de logging
- ENV: Ambiente (dev/prod)

## 🔍 MLflow Integration

Modelo registrado:
- Name: `turkish-music-emotion-rf`
- Run ID: `3e7ecefffa2343d59a23e6d31e0ab705`
- Artifact: `model/rf_raw_84pct.pkl`
- Accuracy: 84.29%

## 📊 Monitoreo

Logs: `logs/app.log` (rotación automática)

MLflow UI:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

## 🚨 Troubleshooting

Puerto ocupado:
```bash
lsof -i :8001
uvicorn app.main:app --port 8002
```

Features incorrectos:
```
Error: "Deben ser 50 características, recibidas: X"
```
→ Enviar exactamente 50 características en formato correcto.

---

**Team:** MLOps Team 24 - Tecnológico de Monterrey  
**Last Update:** 2025-11-14
