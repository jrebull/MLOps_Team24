import streamlit as st
import logging
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    UI_CONFIG, EMOTION_CLASSES, EMOTION_COLORS, EMOTION_EMOJIS,
    SAMPLE_SONGS, MODEL_CONFIG, MLOPS_INFO, AUDIO_DIR
)
from utils import (
    MusicEmotionPredictor,
    AudioVisualizer,
    AudioFeatureExtractor
)

st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

@st.cache_resource
def load_predictor():
    try:
        predictor = MusicEmotionPredictor()
        logger.info("✅ Predictor loaded successfully")
        return predictor
    except Exception as e:
        logger.error(f"❌ Failed to load predictor: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("🎵 Turkish Music Emotion Recognition")
    st.markdown("*Predict emotions in Turkish music using Machine Learning*")
    
    predictor = load_predictor()
    if predictor is None:
        st.error("Failed to load model. Please check logs.")
        return
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses an ensemble ML model to classify emotions in Turkish music:
        - 🔴 **Angry**
        - 🟢 **Happy**
        - 🔵 **Relax**
        - 🟡 **Sad**
        """)
        
        st.header("📊 Model Info")
        st.metric("Accuracy", "80.17%")
        st.metric("Model Type", "Voting Ensemble")
        st.caption("RF + GB + ExtraTrees")
    
    tab1, tab2 = st.tabs(["🎵 Analyze Music", "📖 About Model"])
    
    with tab1:
        st.subheader("Select Audio Source")
        
        audio_source = st.radio(
            "Choose input method:",
            ["Sample Songs", "Upload Your Own"],
            horizontal=True
        )
        
        audio_file = None
        
        if audio_source == "Sample Songs":
            # Flatten sample songs
            all_songs = []
            for emotion, songs in SAMPLE_SONGS.items():
                for song in songs:
                    all_songs.append({
                        "display": f"{song['name']} ({emotion})",
                        "file": song["file"]
                    })
            
            song_options = {s["display"]: s["file"] for s in all_songs}
            selected_song = st.selectbox("Choose a sample:", list(song_options.keys()))
            audio_file = AUDIO_DIR / song_options[selected_song]
            
            st.audio(str(audio_file))
            
        else:
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=['mp3', 'wav', 'ogg'],
                help="Supported formats: MP3, WAV, OGG"
            )
            if uploaded_file:
                audio_file = uploaded_file
        
        if audio_file and st.button("🎯 Analyze Emotion", type="primary"):
            with st.spinner("Analyzing audio..."):
                try:
                    extractor = AudioFeatureExtractor()
                    features_df = extractor.extract_features_to_dataframe(audio_file)
                    
                    result = predictor.predict(features_df)
                    
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("🎯 Predicted Emotion")
                        emotion = result["emotion"]
                        confidence = result["confidence"]
                        
                        emoji = EMOTION_EMOJIS.get(emotion, "🎵")
                        st.markdown(f"## {emoji} **{emotion.upper()}**")
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.1%}")
                    
                    with col2:
                        st.subheader("📊 All Probabilities")
                        for emo, prob in sorted(result["probabilities"].items(), 
                                               key=lambda x: x[1], reverse=True):
                            emoji = EMOTION_EMOJIS.get(emo, "🎵")
                            st.metric(f"{emoji} {emo.capitalize()}", f"{prob:.1%}")
                    
                    st.subheader("📈 Audio Analysis")
                    visualizer = AudioVisualizer()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(visualizer.plot_probabilities(result["probabilities"], EMOTION_COLORS, result["emotion"]))
                    with col2:
                        # st.pyplot(visualizer.plot_confidence_gauge(confidence))
                        pass  # Removed gauge plot
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}", exc_info=True)
    
    with tab2:
        st.subheader("🤖 Model Architecture")
        st.markdown("""
        ### Voting Ensemble Classifier
        Our model combines three powerful algorithms:
        
        1. **Random Forest** (300 estimators, depth=25)
        2. **Gradient Boosting** (200 estimators, lr=0.05)
        3. **Extra Trees** (300 estimators, depth=25)
        
        **Voting Strategy:** Soft voting with weights [2, 1, 2]
        
        ### Performance Metrics
        - Overall Accuracy: **80.17%**
        - Angry Recall: **93%** ⭐
        - Happy Recall: **68%**
        - Relax Recall: **83%**
        - Sad Recall: **77%**
        
        ### Features Used
        50 acoustic features extracted from audio
        """)

if __name__ == "__main__":
    main()
