# 🎵 Turkish Music Emotion Recognition - Streamlit App

Professional web application for predicting emotions in Turkish music using Machine Learning and MLOps best practices.

**Built by**: MLOps Team 24  
**Institution**: Tecnológico de Monterrey (ITESM)  
**Phase**: Production Demo - Phase 2

---

## 🎯 Features

- **Real-time Emotion Prediction**: Analyze Turkish music and predict emotions (Happy, Sad, Angry, Relax)
- **Audio Feature Extraction**: Extracts 50 acoustic features from audio files
- **Professional Visualizations**: Waveforms, spectrograms, feature importance plots
- **Three Modes**:
  - 🎵 **Sample Mode**: Select from pre-loaded songs
  - 🎤 **Upload Mode**: Upload your own audio files
  - 📊 **Batch Mode**: Analyze multiple songs at once
- **Model Performance**: 76.86% accuracy on test set
- **SOLID Architecture**: Clean code following software engineering best practices

---

## 📁 Project Structure

```
turkish_music_app/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── audio_feature_extractor.py  # Extracts 50 acoustic features
│   ├── model_loader.py             # Loads model + preprocessing
│   ├── predictions.py              # Prediction orchestration
│   └── visualizations.py           # Plot generation
├── models/
│   └── production_model.pkl        # Trained Random Forest model
├── data/
│   └── precomputed_features.json   # Sample pre-computed features
└── assets/
    └── sample_audio/               # Sample audio files
        ├── angry/    (4 songs)
        ├── happy/    (4 songs)
        ├── relax/    (4 songs)
        └── sad/      (4 songs)
```

---

## 🚀 Quick Start

### 1. **Setup Directory and Copy Model**

From your MLOps_Team24 project root:

```bash
# Create app directory (if not already done)
mkdir -p turkish_music_app/utils
mkdir -p turkish_music_app/models
mkdir -p turkish_music_app/data
mkdir -p turkish_music_app/assets/sample_audio/{angry,happy,relax,sad}

# Copy production model
cp models/optimized/production_model.pkl turkish_music_app/models/

# Navigate to app directory
cd turkish_music_app
```

### 2. **Add Audio Files**

Download your 16 songs from Google Drive and organize them:

```
assets/sample_audio/
├── angry/
│   ├── adanali.mp3
│   ├── cemberin_icinde_dizi_muzigi.mp3
│   ├── filinta_dizi_muzigi1.mp3
│   └── yeni_ceri_marsi.mp3
├── happy/
│   ├── adana_kopru_basi_murat_kursun.mp3
│   ├── cit_cit_cetene_ahmet_kurt.mp3
│   ├── gir_kanima_harun_kolcak.mp3
│   └── vay_surmeli_surmeli.mp3
├── relax/
│   ├── ajda-pekkan-ya-sonra-Part.mp3
│   ├── elvan_gunaydin.mp3
│   ├── fikret_kizilok_gonul.mp3
│   └── uyanma_uyu_selim_gungoren.mp3
└── sad/
    ├── al_yazmalim.mp3
    ├── derdimi_kimlere_desem_rusen_yilmaz.mp3
    ├── harman_yeri_surseler_salih_gundogdu.mp3
    └── suzan_suzi_incesaz.mp3
```

### 3. **Install Dependencies**

```bash
# Using your existing conda environment
conda activate .venv

# Install additional requirements (if needed)
pip install streamlit plotly --break-system-packages

# Or create a new environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. **Run the App**

```bash
# From turkish_music_app directory
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎮 How to Use

### Mode 1: Sample Songs (Recommended for Demo)

1. Select an emotion category from dropdown
2. Choose a specific song
3. Click "🎧 Preview" to listen
4. Click "🎯 Analyze Emotion" to predict
5. View results: prediction, confidence, probabilities, and visualizations

### Mode 2: Upload Your Own

1. Click "Choose an audio file" to upload
2. Preview the audio
3. Click "🎯 Analyze Emotion"
4. View results and visualizations

### Mode 3: Batch Analysis

1. Select multiple songs from different categories
2. Click "🚀 Run Batch Analysis"
3. View accuracy metrics and detailed results table
4. Download results as CSV

---

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier (Optimized)
- **Accuracy**: 76.86% on test set
- **Features**: 50 acoustic features extracted using librosa
- **Classes**: Happy, Sad, Angry, Relax
- **Preprocessing**: Yeo-Johnson Power Transform + Robust Scaling

### Feature Categories

1. **Energy Features**: RMS Energy, Low Energy
2. **Temporal Features**: Tempo, Fluctuation, Attack Time
3. **Spectral Features**: Centroid, Spread, Rolloff, Flatness, etc.
4. **Timbral Features**: 13 MFCCs (Mel-Frequency Cepstral Coefficients)
5. **Harmonic Features**: 12 Chromagram values
6. **Perceptual Features**: Roughness, Brightness, Pulse Clarity
7. **HCDF Features**: Harmonic Change Detection Function

---

## 🏗️ Architecture & Design Principles

### SOLID Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible without modifying existing code
- **Liskov Substitution**: Components are interchangeable
- **Interface Segregation**: Clean, minimal interfaces
- **Dependency Inversion**: Depends on abstractions, not concretions

### Clean Code Practices

- **Descriptive names**: Clear, intention-revealing names
- **Small functions**: Each function does one thing well
- **DRY**: Don't Repeat Yourself
- **Error handling**: Proper exception handling and logging
- **Documentation**: Comprehensive docstrings

### MLOps Integration

- **Version Control**: All code in Git
- **Model Versioning**: Uses production model from MLflow
- **Feature Engineering**: Consistent with training pipeline
- **Preprocessing**: Replicates acoustic_ml pipeline
- **Monitoring**: Logging for debugging and analysis

---

## 🌐 Deployment to Streamlit Cloud

### Prerequisites

1. Push your code to GitHub
2. Ensure all files are committed
3. Add `.gitattributes` for audio files (Git LFS) if files > 100MB total

### Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `MLOps_Team24`
5. Branch: `main`
6. Main file path: `turkish_music_app/app.py`
7. Click "Deploy"

### Configuration

If you need environment variables or secrets:

1. In Streamlit Cloud app settings
2. Add secrets in TOML format
3. Access via `st.secrets` in code

---

## 🧪 Testing

### Manual Testing Checklist

- [ ] App loads without errors
- [ ] Model loads successfully
- [ ] Audio files are accessible
- [ ] Prediction works for all emotions
- [ ] Visualizations render correctly
- [ ] Batch analysis completes
- [ ] Download CSV works

### Automated Testing (Future)

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 76.86% |
| Precision (Macro) | 77.72% |
| Recall (Macro) | 77.11% |
| F1-Score (Macro) | 76.87% |

### Per-Class Performance

| Emotion | Accuracy | F1-Score | Support |
|---------|----------|----------|---------|
| Happy   | 96.67%   | 84.06%   | 30      |
| Sad     | 61.29%   | 63.33%   | 31      |
| Angry   | 82.76%   | 88.89%   | 29      |
| Relax   | 67.74%   | 71.19%   | 31      |

---

## 🛠️ Troubleshooting

### Model Not Loading

```bash
# Verify model exists
ls -lh models/production_model.pkl

# If missing, copy from main project
cp ../models/optimized/production_model.pkl models/
```

### Audio Files Not Found

```bash
# Check audio directory structure
tree assets/sample_audio/

# Ensure correct file names (no spaces, correct extensions)
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version (requires 3.9+)
python --version
```

### Streamlit Errors

```bash
# Clear cache
streamlit cache clear

# Run in debug mode
streamlit run app.py --logger.level=debug
```

---

## 📝 Future Enhancements

- [ ] Add more Turkish music samples
- [ ] Implement feature importance visualization
- [ ] Add confusion matrix on batch analysis
- [ ] Support for YouTube URL input
- [ ] Real-time microphone input
- [ ] Multi-language support (Turkish/English)
- [ ] Export analysis reports as PDF
- [ ] A/B testing with multiple models
- [ ] User feedback collection
- [ ] Integration with MLflow for live model updates

---

## 👥 Team

**MLOps Team 24**
- David Cruz Beltrán - Software Engineer
- Javier Augusto Rebull Saucedo - SRE/Data Engineer
- Sandra Luz Cervantes Espinoza - ML Engineer/Data Scientist

---

## 📄 License

This project is part of academic coursework at Tecnológico de Monterrey.

---

## 🙏 Acknowledgments

- Tecnológico de Monterrey - Master's in Applied AI
- Turkish Music Emotion Dataset contributors
- Librosa library for audio processing
- Streamlit for the amazing framework

---

## 📧 Contact

For questions or feedback, contact the MLOps Team 24.

---

**Made with ❤️ and 🤖 by MLOps Team 24**
# Last update: Sat Nov  1 16:57:13 EDT 2025
