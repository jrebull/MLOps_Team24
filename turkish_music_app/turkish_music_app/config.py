"""
Configuration Module for Turkish Music Emotion Recognition App

Centralizes all configuration settings following best practices.

Author: MLOps Team 24
Date: November 2025
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = ASSETS_DIR / "sample_audio"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# MLflow Run ID para cargar modelo desde MLflow
MLFLOW_RUN_ID = "0137897a63fc4468b34c8d89071b08a4"

MODEL_CONFIG = {
    "production_model": {
        "path": MODELS_DIR / "production_model.pkl",
        "name": "Voting Ensemble",
        "accuracy": 0.8017,
        "n_features": 50,
        "description": "Ensemble: RF + GB + ExtraTrees"
    }
}

# ============================================================================
# EMOTION CLASSES
# ============================================================================
EMOTION_CLASSES = ["angry", "happy", "relax", "sad"]

EMOTION_COLORS = {
    "angry": "#DC143C",
    "happy": "#FFD700",
    "relax": "#32CD32",
    "sad": "#4682B4"
}

EMOTION_EMOJIS = {
    "angry": "😠",
    "happy": "😊",
    "relax": "😌",
    "sad": "😢"
}

# ============================================================================
# SAMPLE SONGS CATALOG
# ============================================================================
SAMPLE_SONGS = {
    "Angry": [
        {"name": "Adanalı", "file": "angry/adanali.mp3"},
        {"name": "Cemberin İçinde", "file": "angry/cemberin_icinde_dizi_muzigi.mp3"},
        {"name": "Filinta Dizi", "file": "angry/filinta_dizi_muzigi1.mp3"},
        {"name": "Yeni Çeri Marşı", "file": "angry/yeni_ceri_marsi.mp3"}
    ],
    "Happy": [
        {"name": "Adana Köprü Başı", "file": "happy/adana_kopru_basi_murat_kursun.mp3"},
        {"name": "Çit Çit Çetene", "file": "happy/cit_cit_cetene_ahmet_kurt.mp3"},
        {"name": "Gir Kanıma", "file": "happy/gir_kanima_harun_kolcak.mp3"},
        {"name": "Vay Sürmeli", "file": "happy/vay_surmeli_surmeli.mp3"}
    ],
    "Relax": [
        {"name": "Elvan Günaydın", "file": "relax/elvan_gunaydin.mp3"},
        {"name": "Fikret Kızılok - Gönül", "file": "relax/fikret_kizilok_gonul.mp3"},
        {"name": "Uyanma Uyu", "file": "relax/uyanma_uyu_selim_gungoren.mp3"}
    ],
    "Sad": [
        {"name": "Al Yazmalım", "file": "sad/al_yazmali_m.mp3"},
        {"name": "Derdimi Kimlere", "file": "sad/derdimi_kimlere_desem_rusen_yilmaz.mp3"},
        {"name": "Harman Yeri", "file": "sad/harman_yeri_surseler_salih_gundogdu.mp3"},
        {"name": "Suzan Suzi", "file": "sad/suzan_suzi_incesaz.mp3"}
    ]
}

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================
UI_CONFIG = {
    "page_title": "Turkish Music Emotion Recognition",
    "page_icon": "🎵",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============================================================================
# MLOPS METADATA
# ============================================================================
MLOPS_INFO = {
    "team": "MLOps Team 24",
    "project": "Turkish Music Emotion Recognition",
    "phase": "Phase 2 - Production Demo",
    "model_version": "v2.1 - Ensemble",
    "last_updated": "2025-11-01"
}

AUDIO_DIR = ASSETS_DIR / "sample_audio"
