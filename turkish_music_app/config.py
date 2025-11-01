"""Configuration Module for Turkish Music Emotion Recognition App"""
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = ASSETS_DIR / "sample_audio"

# Solo modelos compatibles con Streamlit Cloud (sin dependencias de acoustic_ml)
AVAILABLE_MODELS = {
    "Baseline_Model": {
        "path": MODELS_DIR / "baseline_model.pkl",
        "name": "🎯 Random Forest Model - 76.9%",
        "accuracy": 0.769,
        "n_features": 50,
        "description": "Production-ready sklearn model"
    }
}

DEFAULT_MODEL = "Baseline_Model"
EMOTION_CLASSES = ["angry", "happy", "relax", "sad"]

EMOTION_COLORS = {"angry": "#DC143C", "happy": "#FFD700", "relax": "#32CD32", "sad": "#4682B4"}
EMOTION_EMOJIS = {"angry": "😠", "happy": "😊", "relax": "😌", "sad": "😢"}

FEATURE_NAMES = ["_RMSenergy_Mean", "_Lowenergy_Mean", "_Fluctuation_Mean", "_Tempo_Mean", "_MFCC_Mean_1", "_MFCC_Mean_2", "_MFCC_Mean_3", "_MFCC_Mean_4", "_MFCC_Mean_5", "_MFCC_Mean_6", "_MFCC_Mean_7", "_MFCC_Mean_8", "_MFCC_Mean_9", "_MFCC_Mean_10", "_MFCC_Mean_11", "_MFCC_Mean_12", "_MFCC_Mean_13", "_Roughness_Mean", "_Roughness_Slope", "_Zero-crossingrate_Mean", "_AttackTime_Mean", "_AttackTime_Slope", "_Rolloff_Mean", "_Eventdensity_Mean", "_Pulseclarity_Mean", "_Brightness_Mean", "_Spectralcentroid_Mean", "_Spectralspread_Mean", "_Spectralskewness_Mean", "_Spectralkurtosis_Mean", "_Spectralflatness_Mean", "_EntropyofSpectrum_Mean", "_Chromagram_Mean_1", "_Chromagram_Mean_2", "_Chromagram_Mean_3", "_Chromagram_Mean_4", "_Chromagram_Mean_5", "_Chromagram_Mean_6", "_Chromagram_Mean_7", "_Chromagram_Mean_8", "_Chromagram_Mean_9", "_Chromagram_Mean_10", "_Chromagram_Mean_11", "_Chromagram_Mean_12", "_HarmonicChangeDetectionFunction_Mean", "_HarmonicChangeDetectionFunction_Std", "_HarmonicChangeDetectionFunction_Slope", "_HarmonicChangeDetectionFunction_PeriodFreq", "_HarmonicChangeDetectionFunction_PeriodAmp", "_HarmonicChangeDetectionFunction_PeriodEntropy"]

SAMPLE_SONGS = {"Angry": [{"name": "Adanali", "file": "adanali.mp3"}, {"name": "Cemberin Icinde", "file": "cemberin_icinde_dizi_muzigi.mp3"}], "Happy": [{"name": "Adana Kopru Basi", "file": "adana_kopru_basi_murat_kursun.mp3"}, {"name": "Cit Cit Cetene", "file": "cit_cit_cetene_ahmet_kurt.mp3"}], "Relax": [{"name": "Elvan Gunaydin", "file": "elvan_gunaydin.mp3"}, {"name": "Fikret Kizilok", "file": "fikret_kizilok_gonul.mp3"}], "Sad": [{"name": "Al Yazmalim", "file": "al_yazmalim.mp3"}, {"name": "Derdimi Kimlere", "file": "derdimi_kimlere_desem_rusen_yilmaz.mp3"}]}

UI_CONFIG = {"page_title": "Turkish Music Emotion Recognition", "page_icon": "🎸", "layout": "wide", "initial_sidebar_state": "expanded"}

MLOPS_INFO = {"team": "MLOps Team 24", "project": "Turkish Music Emotion Recognition", "phase": "Phase 2 - Production Demo", "institution": "Tecnologico de Monterrey", "model_version": "v2.2 - Streamlit Cloud", "last_updated": "2025-11-01"}
