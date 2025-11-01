"""
Turkish Music Emotion Recognition - Streamlit App (Enhanced UX/UI)

Professional web application for predicting emotions in Turkish music.
Built following MLOps best practices by Team 24.

Author: MLOps Team 24
Date: November 2024
Institution: Tecnológico de Monterrey

UX/UI Enhancements:
- Improved horizontal space usage with multi-column layouts
- Enhanced visual cards with gradients and shadows
- Better progress indicators and loading states
- Musical-themed color palette
- Larger, more prominent emojis and results
- Organized visualizations with tabs
- Better visual hierarchy and spacing
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import json
import plotly.graph_objects as go  # Added Plotly for colored charts

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    UI_CONFIG, EMOTION_CLASSES, EMOTION_COLORS, EMOTION_EMOJIS,
    SAMPLE_SONGS, AVAILABLE_MODELS, DEFAULT_MODEL, MLOPS_INFO, AUDIO_DIR
)
from utils import (
    MusicEmotionPredictor,
    AudioVisualizer,
    AudioFeatureExtractor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon="🎸",
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

# ============================================================================
# ENHANCED CUSTOM CSS - Blue Theme with Dark Mode Support
# ============================================================================

st.markdown("""
    <style>
    /* Dark mode detection and variables */
    :root {
        --primary-blue: #2563eb;
        --secondary-blue: #3b82f6;
        --light-blue: #60a5fa;
        --dark-blue: #1e40af;
        --accent-blue: #0ea5e9;
    }
    
    /* Main header styling - Blue gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Dark mode support for headers */
    @media (prefers-color-scheme: dark) {
        .sub-header {
            color: #94a3b8;
        }
    }
    
    /* Enhanced emotion result card with blue gradient */
    .emotion-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(239,246,255,0.95) 100%);
        border: 2px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* Dark mode emotion card */
    @media (prefers-color-scheme: dark) {
        .emotion-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.95) 0%, rgba(15,23,42,0.95) 100%);
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
        }
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(37, 99, 235, 0.3);
    }
    
    /* Large emoji display */
    .emotion-emoji {
        font-size: 6rem;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .emotion-name {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
        color: #475569;
        font-weight: 500;
    }
    
    @media (prefers-color-scheme: dark) {
        .confidence-text {
            color: #cbd5e1;
        }
    }
    
    /* Enhanced metric cards - Blue theme */
    .metric-card {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.15);
        margin: 0.5rem 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }
    }
    
    /* Probability bars with animation - Blue theme */
    .prob-bar-container {
        margin: 1rem 0;
        padding: 0.5rem;
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    @media (prefers-color-scheme: dark) {
        .prob-bar-container {
            background: #1e293b;
        }
    }
    
    .prob-bar {
        height: 35px;
        border-radius: 8px;
        transition: width 1s ease-out;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Enhanced buttons - Blue theme */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%);
    }
    
    /* Section headers - Blue theme */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e40af;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    @media (prefers-color-scheme: dark) {
        .section-header {
            color: #60a5fa;
            border-bottom-color: #3b82f6;
        }
    }
    
    /* Info boxes - Blue theme */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #2563eb;
        color: #1e40af;
    }
    
    @media (prefers-color-scheme: dark) {
        .info-box {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            border-left-color: #60a5fa;
            color: #e0f2fe;
        }
    }
    
    /* Sidebar styling - Blue theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    @media (prefers-color-scheme: dark) {
        .css-1d391kg {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
    }
    
    /* Tab styling - Blue theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f5f9;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #475569;
    }
    
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            background-color: #1e293b;
            color: #cbd5e1;
        }
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: white !important;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 10px;
    }
    
    /* Expander styling - Blue theme */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        background-color: #f0f9ff;
        border-radius: 10px;
    }
    
    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader {
            background-color: #1e293b;
        }
    }
    
    /* Improve contrast for dark mode text */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, p, span, div {
            color: #e2e8f0;
        }
        
        strong {
            color: #f1f5f9;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'selected_audio_path' not in st.session_state:
    st.session_state.selected_audio_path = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_predictor(model_path=None):
    """Load the model predictor (cached)."""
    try:
        predictor = MusicEmotionPredictor(model_path=model_path)
        model_name = predictor.load_model()
        st.sidebar.success(f"🎯 Using: {model_name}")
        logger.info("✅ Predictor loaded successfully")
        return predictor
    except Exception as e:
        logger.error(f"❌ Failed to load predictor: {e}")
        st.error(f"Error loading model: {e}")
        return None


def get_audio_files():
    """Get available audio files organized by emotion (loads ALL songs dynamically)."""
    from pathlib import Path
    
    audio_catalog = {}
    emotions = ["Angry", "Happy", "Relax", "Sad"]
    
    for emotion in emotions:
        audio_catalog[emotion] = []
        emotion_dir = AUDIO_DIR / emotion.lower()
        
        if emotion_dir.exists():
            mp3_files = sorted(emotion_dir.glob("*.mp3"))
            
            for audio_path in mp3_files:
                display_name = audio_path.stem.replace("_", " ").title()
                
                audio_catalog[emotion].append({
                    "name": display_name,
                    "path": audio_path
                })
    
    return audio_catalog


def display_emotion_result(emotion: str, confidence: float, probabilities: dict):
    """
    Display prediction result in an enhanced, visually appealing format.
    UX Enhancement: Larger emojis, gradient cards, animated bars, dynamic emotion colors
    """
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        emoji = EMOTION_EMOJIS.get(emotion, "🎵")
        color = EMOTION_COLORS.get(emotion, '#808080')
        
        st.markdown(
            f"""
            <div class="emotion-card" style="
                border-color: {color};
                background: linear-gradient(135deg, {color}90 0%, {color}70 100%);
                box-shadow: 0 10px 30px {color}80;">
                <div class="emotion-emoji">{emoji}</div>
                <div class="emotion-name" style="color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{emotion}</div>
                <div class="confidence-text" style="color: white;">
                    Confidence: <strong style="color: white;">{confidence:.1%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('<div class="section-header">📊 Emotion Probabilities</div>', unsafe_allow_html=True)
    
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for Plotly
    emotions_list = [emo for emo, _ in sorted_probs]
    probs_list = [prob for _, prob in sorted_probs]
    colors_list = [EMOTION_COLORS.get(emo, '#808080') for emo in emotions_list]
    emojis_list = [EMOTION_EMOJIS.get(emo, '🎵') for emo in emotions_list]
    
    # Create horizontal bar chart with Plotly
    fig = go.Figure()
    
    for i, (emo, prob, color, emoji) in enumerate(zip(emotions_list, probs_list, colors_list, emojis_list)):
        is_predicted = emo == emotion
        
        fig.add_trace(go.Bar(
            y=[f"{emoji} {emo}"],
            x=[prob],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color=color if not is_predicted else 'white', width=3 if is_predicted else 0)
            ),
            text=f"{prob:.1%}",
            textposition='outside',
            textfont=dict(size=14, color=color, family='Arial Black'),
            hovertemplate=f"<b>{emo}</b><br>Probability: {prob:.1%}<extra></extra>",
            showlegend=False
        ))
    
    # Update layout for better appearance
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=100, t=20, b=20),
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showticklabels=True,
            tickformat='.0%'
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14, family='Arial', color='#64748b')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show predicted badge
    for emo, prob in sorted_probs:
        if emo == emotion:
            st.success(f"✅ Predicted: {emo} with {prob:.1%} confidence")
            break


def display_model_info(selected_model):
    """
    Display model information in sidebar with enhanced visuals.
    UX Enhancement: Better metric display with icons
    """
    st.sidebar.markdown("---")
    model_info = AVAILABLE_MODELS[selected_model]
    
    st.sidebar.markdown("### 📈 Model Performance")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🎯 Accuracy", f"{model_info['accuracy']:.1%}")
    with col2:
        st.metric("🔢 Features", model_info["n_features"])
    
    with st.sidebar.expander("ℹ️ Model Details", expanded=False):
        st.markdown(f"**Description:**  \n{model_info['description']}")
        st.markdown(f"**Classes:**  \n{', '.join(EMOTION_CLASSES)}")


def display_team_info():
    """Display team information in sidebar with enhanced styling."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👥 About This Project")
    
    st.sidebar.info(f"""
    **{MLOPS_INFO['project']}**
    
    🏫 {MLOPS_INFO['institution']}  
    👥 {MLOPS_INFO['team']}  
    📅 Phase: {MLOPS_INFO['phase']}  
    🔄 Updated: {MLOPS_INFO['last_updated']}
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application logic with enhanced UX/UI.
    UX Enhancements: Better layout, progress indicators, visual feedback
    """
    
    st.markdown('<p class="main-header">🎵 Turkish Music Emotion Recognition</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">🎸 AI-powered emotion detection in Turkish music using MLOps best practices 🎼</p>',
        unsafe_allow_html=True
    )
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("### 🎯 Select Model")
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} ({AVAILABLE_MODELS[x]['accuracy']:.1%})",
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
        help="Select the AI model for emotion prediction"
    )
    
    with st.spinner("🔄 Loading AI model... Please wait"):
        predictor = load_predictor(AVAILABLE_MODELS[selected_model]["path"])
        if predictor is None:
            st.error("❌ Failed to load model. Please check model file exists.")
            return
        st.session_state.predictor = predictor
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🎮 Select Mode")
    mode = st.sidebar.radio(
        "Choose analysis mode:",
        ["🎵 Predict from Sample", "🎤 Upload Your Audio", "📊 Batch Analysis"],
        index=0,
        help="Select how you want to analyze music"
    )
    
    # Display model and team info
    display_model_info(selected_model)
    display_team_info()
    
    # ========================================================================
    # MODE 1: PREDICT FROM SAMPLE
    # ========================================================================
    if mode == "🎵 Predict from Sample":
        st.markdown('<div class="section-header">🎵 Select a Sample Song</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <strong>💡 How it works:</strong> Choose an emotion category and a song to analyze. 
                Our AI will predict the emotional content of the music!
            </div>
        """, unsafe_allow_html=True)
        
        audio_catalog = get_audio_files()
        
        if not audio_catalog or all(len(songs) == 0 for songs in audio_catalog.values()):
            st.warning("""
            ⚠️ **No audio files found!**
            
            Please add audio files to the `assets/sample_audio/` directory:
            - `assets/sample_audio/angry/` - Songs with angry emotion
            - `assets/sample_audio/happy/` - Songs with happy emotion
            - `assets/sample_audio/relax/` - Songs with relaxing emotion
            - `assets/sample_audio/sad/` - Songs with sad emotion
            """)
            return
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_emotion = st.selectbox(
                "🎭 Select Emotion Category",
                options=list(audio_catalog.keys()),
                format_func=lambda x: f"{EMOTION_EMOJIS.get(x, '🎵')} {x}",
                help="Choose the emotion category to explore"
            )
        
        with col2:
            songs_in_category = audio_catalog.get(selected_emotion, [])
            if songs_in_category:
                selected_song = st.selectbox(
                    "🎼 Select Song",
                    options=range(len(songs_in_category)),
                    format_func=lambda i: songs_in_category[i]["name"],
                    help="Choose a specific song to analyze"
                )
                audio_path = songs_in_category[selected_song]["path"]
                st.session_state.selected_audio_path = audio_path
            else:
                st.warning(f"No songs found for {selected_emotion}")
                return
        
        with col3:
            st.metric("📚 Available", f"{len(songs_in_category)} songs")
        
        if st.session_state.selected_audio_path:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 🎧 Preview Audio")
                st.audio(str(st.session_state.selected_audio_path))
        
        st.markdown("---")
        
        if st.button("🎯 Analyze Emotion", type="primary", help="Click to start AI analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🎵 Loading audio file...")
            progress_bar.progress(20)
            
            status_text.text("🔊 Extracting audio features...")
            progress_bar.progress(50)
            
            status_text.text("🧠 Running AI prediction...")
            progress_bar.progress(80)
            
            try:
                result = predictor.predict_from_audio(
                    audio_path=st.session_state.selected_audio_path,
                    return_probabilities=True
                )
                st.session_state.prediction_result = result
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Analysis complete! Here are the results:")
                
                # Display main result
                display_emotion_result(
                    emotion=result['predicted_emotion'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities']
                )
                
                st.markdown("---")
                
                st.markdown('<div class="section-header">🎯 Accuracy Check</div>', unsafe_allow_html=True)
                
                is_correct = result['predicted_emotion'] == selected_emotion
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎭 True Emotion", 
                        selected_emotion,
                        help="The actual emotion category of this song"
                    )
                
                with col2:
                    st.metric(
                        "🤖 AI Prediction", 
                        result['predicted_emotion'],
                        help="What our AI model predicted"
                    )
                
                with col3:
                    st.metric(
                        "📊 Confidence", 
                        f"{result['confidence']:.1%}",
                        help="How confident the AI is about this prediction"
                    )
                
                with col4:
                    if is_correct:
                        st.success("✅ Correct!")
                    else:
                        st.error("❌ Incorrect")
                
                st.caption(f"🤖 Model used: {result.get('model_used', 'Unknown')}")
                
                st.markdown("---")
                st.markdown('<div class="section-header">📊 Audio Analysis & Visualizations</div>', unsafe_allow_html=True)
                
                viz = AudioVisualizer()
                
                tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🎨 Spectrogram", "🎼 Feature Importance"])
                
                with tab1:
                    st.markdown("**Audio waveform showing amplitude over time**")
                    fig_wave = viz.plot_waveform(st.session_state.selected_audio_path)
                    st.pyplot(fig_wave)
                    st.caption("The waveform shows the audio signal's amplitude variations")
                
                with tab2:
                    st.markdown("**Frequency spectrum visualization**")
                    fig_spec = viz.plot_spectrogram(st.session_state.selected_audio_path)
                    st.pyplot(fig_spec)
                    st.caption("The spectrogram displays frequency content over time")
                
                with tab3:
                    st.markdown("**Top 10 most important features for prediction**")
                    fig_features = viz.plot_top_features(result['features'], top_n=10)
                    st.pyplot(fig_features)
                    st.caption("These audio features had the most influence on the prediction")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Prediction failed: {e}")
                logger.error(f"Prediction error: {e}")
    
    # ========================================================================
    # MODE 2: UPLOAD AUDIO
    # ========================================================================
    elif mode == "🎤 Upload Your Audio":
        st.markdown('<div class="section-header">🎤 Upload Your Own Audio</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <strong>💡 Upload your music:</strong> Upload an audio file (.mp3, .wav, .flac) 
                and our AI will analyze its emotional content!
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'flac'],
                help="Upload a Turkish music file to analyze (max 200MB)"
            )
        
        if uploaded_file is not None:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 File Name", uploaded_file.name)
            with col2:
                st.metric("📏 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("🎵 Format", uploaded_file.type.split("/")[-1].upper())
            
            # Save temporarily
            temp_path = Path("temp_upload.mp3")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 🎧 Preview Your Audio")
                st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
            
            st.markdown("---")
            
            # Analyze button
            if st.button("🎯 Analyze Emotion", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🎵 Processing uploaded file...")
                progress_bar.progress(25)
                
                status_text.text("🔊 Extracting audio features...")
                progress_bar.progress(60)
                
                status_text.text("🧠 Running AI prediction...")
                progress_bar.progress(90)
                
                try:
                    result = predictor.predict_from_audio(
                        audio_path=temp_path,
                        return_probabilities=True
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Analysis complete! Here are the results:")
                    
                    display_emotion_result(
                        emotion=result['predicted_emotion'],
                        confidence=result['confidence'],
                        probabilities=result['probabilities']
                    )
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">📊 Audio Analysis</div>', unsafe_allow_html=True)
                    
                    viz = AudioVisualizer()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🌊 Waveform**")
                        fig_wave = viz.plot_waveform(temp_path)
                        st.pyplot(fig_wave)
                    
                    with col2:
                        st.markdown("**🎨 Spectrogram**")
                        fig_spec = viz.plot_spectrogram(temp_path)
                        st.pyplot(fig_spec)
                    
                    # Clean up
                    temp_path.unlink()
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
    
    # ========================================================================
    # MODE 3: BATCH ANALYSIS
    # ========================================================================
    elif mode == "📊 Batch Analysis":
        st.markdown('<div class="section-header">📊 Batch Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <strong>💡 Analyze multiple songs:</strong> Select multiple songs from different categories 
                and analyze them all at once. Perfect for comparing model performance!
            </div>
        """, unsafe_allow_html=True)
        
        audio_catalog = get_audio_files()
        
        if not audio_catalog or all(len(songs) == 0 for songs in audio_catalog.values()):
            st.warning("⚠️ No audio files found for batch analysis.")
            return
        
        st.markdown("### 🎼 Select Songs for Analysis")
        
        selected_songs = []
        
        # Display emotions in a 2x2 grid
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        emotion_columns = [row1_col1, row1_col2, row2_col1, row2_col2]
        
        for idx, (emotion, songs) in enumerate(audio_catalog.items()):
            if songs and idx < len(emotion_columns):
                with emotion_columns[idx]:
                    with st.expander(f"{EMOTION_EMOJIS.get(emotion, '🎵')} {emotion} ({len(songs)} songs)", expanded=True):
                        for i, song in enumerate(songs):
                            if st.checkbox(f"{song['name']}", key=f"{emotion}_{i}"):
                                selected_songs.append({
                                    "name": song["name"],
                                    "path": song["path"],
                                    "true_emotion": emotion
                                })
        
        if selected_songs:
            col1, col2, col3 = st.columns(3)
            with col2:
                st.info(f"✅ Selected **{len(selected_songs)}** songs for batch analysis")
            
            st.markdown("---")
            
            if st.button("🚀 Run Batch Analysis", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, song_info in enumerate(selected_songs):
                    status_text.text(f"🎵 Analyzing {i+1}/{len(selected_songs)}: {song_info['name']}")
                    
                    try:
                        result = predictor.predict_from_audio(
                            audio_path=song_info["path"],
                            return_probabilities=True
                        )
                        
                        results.append({
                            "Song": song_info["name"],
                            "True Emotion": song_info["true_emotion"],
                            "Predicted": result['predicted_emotion'],
                            "Confidence": f"{result['confidence']:.1%}",
                            "Correct": "✅" if result['predicted_emotion'] == song_info["true_emotion"] else "❌"
                        })
                        
                    except Exception as e:
                        st.warning(f"Failed to analyze {song_info['name']}: {e}")
                    
                    progress_bar.progress((i + 1) / len(selected_songs))
                
                status_text.text("✅ Batch analysis complete!")
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success("✅ Batch analysis complete!")
                
                df_results = pd.DataFrame(results)
                
                st.markdown('<div class="section-header">📈 Overall Performance</div>', unsafe_allow_html=True)
                
                correct_count = df_results['Correct'].str.contains('✅').sum()
                accuracy = correct_count / len(df_results) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📚 Total Songs", len(df_results))
                with col2:
                    st.metric("✅ Correct", correct_count)
                with col3:
                    st.metric("❌ Incorrect", len(df_results) - correct_count)
                with col4:
                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%", 
                             delta=f"{accuracy - 75:.1f}%" if accuracy > 75 else None)
                
                st.markdown("---")
                st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
                
                st.dataframe(
                    df_results, 
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df_results.to_csv(index=False)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv",
                        help="Download the analysis results as a CSV file"
                    )
        else:
            st.info("👆 Select songs from the categories above to start batch analysis")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Custom logo in sidebar
    with st.sidebar:
        try:
            st.image(str(Path(__file__).parent / "assets" / "logo.png"), width=200)
        except:
            pass  # Logo is optional
        import matplotlib.pyplot as plt
        plt.close("all")  # Clear any stray figures
        st.markdown("---")
    
    main()
