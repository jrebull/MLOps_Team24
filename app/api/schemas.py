"""
Schemas para API - Pydantic models con validación robusta
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class EmotionEnum(str, Enum):
    """Emociones soportadas por el modelo"""
    HAPPY = "Happy"
    SAD = "Sad"
    ANGRY = "Angry"
    RELAX = "Relax"


class PredictRequest(BaseModel):
    """
    Schema para solicitud de predicción.
    
    **50 características acústicas requeridas (en este orden):**
    
    1. RMSenergy_Mean
    2. Lowenergy_Mean
    3. Fluctuation_Mean
    4. Tempo_Mean
    5-17. MFCC_Mean_1 a MFCC_Mean_13 (13 características)
    18. Roughness_Mean
    19. Roughness_Slope
    20. Zero-crossingrate_Mean
    21. AttackTime_Mean
    22. AttackTime_Slope
    23. Rolloff_Mean
    24. Eventdensity_Mean
    25. Pulseclarity_Mean
    26. Brightness_Mean
    27. Spectralcentroid_Mean
    28. Spectralspread_Mean
    29. Spectralskewness_Mean
    30. Spectralkurtosis_Mean
    31. Spectralflatness_Mean
    32. EntropyofSpectrum_Mean
    33-44. Chromagram_Mean_1 a Chromagram_Mean_12 (12 características)
    45. HarmonicChangeDetectionFunction_Mean
    46. HarmonicChangeDetectionFunction_Std
    47. HarmonicChangeDetectionFunction_Slope
    48. HarmonicChangeDetectionFunction_PeriodFreq
    49. HarmonicChangeDetectionFunction_PeriodAmp
    50. HarmonicChangeDetectionFunction_PeriodEntropy
    
    Ejemplo:
```json
    {
      "features": [0.5, 0.3, ..., 0.7]  // 50 características
    }
```
    """
    features: List[float] = Field(
        ...,
        description="50 características acústicas (en orden específico)",
        min_items=50,
        max_items=50,
        examples=[[0.5] * 50]
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        """Valida que features tenga exactamente 50 elementos y sean numéricos válidos"""
        if len(v) != 50:
            raise ValueError(f"Deben ser 50 características, recibidas: {len(v)}")
        
        # Validar que no haya NaN o Inf
        for i, f in enumerate(v):
            if not isinstance(f, (int, float)):
                raise ValueError(f"Feature {i} no es numérico: {type(f)}")
            if f != f:  # NaN check
                raise ValueError(f"Feature {i} es NaN")
            if f == float('inf') or f == float('-inf'):
                raise ValueError(f"Feature {i} es Infinity")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.123] * 50
            }
        }


class ProbabilityDistribution(BaseModel):
    """Distribución de probabilidades por emoción"""
    happy: float = Field(..., ge=0, le=1, description="Probabilidad: Happy")
    sad: float = Field(..., ge=0, le=1, description="Probabilidad: Sad")
    angry: float = Field(..., ge=0, le=1, description="Probabilidad: Angry")
    relax: float = Field(..., ge=0, le=1, description="Probabilidad: Relax")


class PredictResponse(BaseModel):
    """
    Schema para respuesta de predicción.
    
    Ejemplo:
```json
    {
      "prediction": 0,
      "emotion": "Happy",
      "confidence": 0.85,
      "probabilities": [0.85, 0.05, 0.05, 0.05],
      "probabilities_mapped": {
        "Happy": 0.85,
        "Sad": 0.05,
        "Angry": 0.05,
        "Relax": 0.05
      },
      "timestamp": "2025-11-14T10:30:45.123456"
    }
```
    """
    prediction: int = Field(
        ...,
        ge=0,
        le=3,
        description="Índice de clase predicha (0=Happy, 1=Sad, 2=Angry, 3=Relax)"
    )
    emotion: EmotionEnum = Field(
        ...,
        description="Nombre de la emoción predicha"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confianza de la predicción (probabilidad máxima)"
    )
    probabilities: Optional[List[float]] = Field(
        None,
        description="Array de 4 probabilidades [Happy, Sad, Angry, Relax]"
    )
    probabilities_mapped: Optional[Dict[str, float]] = Field(
        None,
        description="Probabilidades mapeadas por nombre de emoción"
    )
    timestamp: str = Field(
        ...,
        description="Timestamp ISO de la predicción (UTC)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado del servicio")
    model_name: Optional[str] = Field(None, description="Nombre del modelo")
    model_version: Optional[str] = Field(None, description="Versión del modelo")


class ErrorResponse(BaseModel):
    """Respuesta de error"""
    detail: str = Field(..., description="Descripción del error")
    error_code: Optional[str] = Field(None, description="Código del error")
    timestamp: str = Field(..., description="Timestamp ISO del error (UTC)")
