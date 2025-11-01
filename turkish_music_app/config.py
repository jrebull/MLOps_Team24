"""
Configuration file for Turkish Music Emotion Recognition App
Contains all constants, paths, and configuration settings
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "assets" / "sample_audio"
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# EMOTION CONFIGURATION - Blue Color Palette
# ============================================================================

EMOTION_CLASSES = ["Angry", "Happy", "Relax", "Sad"]

# Blue-themed color palette for emotions
EMOTION_COLORS = {
    "Happy": "#3b82f6",    # Bright blue
    "Sad": "#1e40af",      # Deep blue
    "Angry": "#dc2626",    # Red (for contrast)
    "Relax": "#06b6d4"     # Cyan blue
}

EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Relax": "😌"
}

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Feature names used by the model (50 features)
FEATURE_NAMES = [
    # MFCCs (20 features)
    'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
    'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10',
    'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5',
    'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10',
    
    # Spectral features (10 features)
    'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_bandwidth_std',
    'spectral_rolloff_mean', 'spectral_rolloff_std',
    'spectral_contrast_mean', 'spectral_contrast_std',
    'spectral_flatness_mean', 'spectral_flatness_std',
    
    # Rhythm features (8 features)
    'tempo', 'tempo_std',
    'zero_crossing_rate_mean', 'zero_crossing_rate_std',
    'onset_strength_mean', 'onset_strength_std',
    'beat_strength_mean', 'beat_strength_std',
    
    # Chroma features (12 features)
    'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6',
    'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'
]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "random_forest"

AVAILABLE_MODELS = {
    "random_forest": {
        "name": "Random Forest Model",
        "path": MODELS_DIR / "baseline_model.pkl",
        "accuracy": 0.769,
        "n_features": len(FEATURE_NAMES),  # 50 features
        "description": "Baseline Random Forest model trained on Turkish music dataset"
    }
}

# ============================================================================
# SAMPLE SONGS CONFIGURATION
# ============================================================================

SAMPLE_SONGS = {
    "Angry": ["Song 1", "Song 2"],
    "Happy": ["Song 1", "Song 2"],
    "Relax": ["Song 1", "Song 2"],
    "Sad": ["Song 1", "Song 2"]
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    "page_title": "Turkish Music Emotion Recognition",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============================================================================
# MLOPS INFO
# ============================================================================

MLOPS_INFO = {
    "project": "Turkish Music Emotion Recognition",
    "institution": "Tecnológico de Monterrey",
    "team": "Team 24",
    "phase": "Phase 2 - Production Demo",
    "last_updated": "November 2024"
}