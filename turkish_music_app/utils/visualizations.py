"""
Visualization Module

Professional visualizations for audio analysis and prediction results.
Creates publication-quality plots following data visualization best practices.

Author: MLOps Team 24
Date: November 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class AudioVisualizer:
    """
    Creates professional visualizations for audio and predictions.
    
    Single Responsibility: Only handles visualization creation
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_waveform(
        self,
        audio_path: Path,
        sr: int = 22050,
        duration: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot audio waveform.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            duration: Duration to load (None = full)
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4', alpha=0.7)
        
        ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info("✅ Waveform plot created")
        return fig
    
    def plot_spectrogram(
        self,
        audio_path: Path,
        sr: int = 22050,
        duration: Optional[float] = None,
        n_mels: int = 128
    ) -> plt.Figure:
        """
        Plot mel spectrogram.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            duration: Duration to load
            n_mels: Number of mel bands
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot spectrogram
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        plt.tight_layout()
        
        logger.info("✅ Spectrogram plot created")
        return fig
    
    def plot_probabilities(
        self,
        probabilities: Dict[str, float],
        emotion_colors: Dict[str, str],
        predicted_emotion: str
    ) -> plt.Figure:
        """
        Plot emotion prediction probabilities as horizontal bars.
        
        Args:
            probabilities: Dict of emotion: probability
            emotion_colors: Dict of emotion: color
            predicted_emotion: The predicted emotion (to highlight)
            
        Returns:
            Matplotlib figure
        """
        # Sort by probability
        sorted_probs = dict(sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        emotions = list(sorted_probs.keys())
        probs = list(sorted_probs.values())
        
        # Assign colors
        colors = [emotion_colors.get(e, '#808080') for e in emotions]
        
        # Highlight predicted emotion
        alphas = [1.0 if e == predicted_emotion else 0.6 for e in emotions]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        bars = ax.barh(emotions, probs, color=colors, alpha=0.8)
        
        # Highlight predicted
        for i, (bar, alpha, emotion) in enumerate(zip(bars, alphas, emotions)):
            bar.set_alpha(alpha)
            if emotion == predicted_emotion:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        # Add value labels
        for i, (emotion, prob) in enumerate(zip(emotions, probs)):
            ax.text(
                prob + 0.01,
                i,
                f'{prob:.1%}',
                va='center',
                fontsize=11,
                fontweight='bold' if emotion == predicted_emotion else 'normal'
            )
        
        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('Emotion Prediction Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        logger.info("✅ Probability plot created")
        return fig
    
    def plot_top_features(
        self,
        features: Dict[str, float],
        top_n: int = 10,
        title: str = "Top Audio Features"
    ) -> plt.Figure:
        """
        Plot top N features by absolute value.
        
        Args:
            features: Dict of feature_name: value
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Get top features by absolute value
        sorted_features = dict(sorted(
            features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        top_features = dict(list(sorted_features.items())[:top_n])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': list(top_features.keys()),
            'Value': list(top_features.values())
        })
        
        # Shorten feature names for display
        df['Feature_Short'] = df['Feature'].str.replace('_Mean', '').str.replace('_', ' ')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in df['Value']]
        bars = ax.barh(df['Feature_Short'], df['Value'], color=colors, alpha=0.7)
        
        ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        logger.info(f"✅ Top {top_n} features plot created")
        return fig
    
    def plot_chromagram(
        self,
        audio_path: Path,
        sr: int = 22050,
        duration: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot chromagram (pitch class distribution over time).
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            duration: Duration to load
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot chromagram
        img = librosa.display.specshow(
            chroma,
            sr=sr,
            x_axis='time',
            y_axis='chroma',
            ax=ax,
            cmap='coolwarm'
        )
        
        fig.colorbar(img, ax=ax)
        ax.set_title('Chromagram (Pitch Classes)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Pitch Class', fontsize=12)
        
        plt.tight_layout()
        
        logger.info("✅ Chromagram plot created")
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: list,
        y_pred: list,
        labels: list,
        normalize: bool = False
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        logger.info("✅ Confusion matrix plot created")
        return fig
    
    def create_prediction_dashboard(
        self,
        audio_path: Path,
        prediction_result: Dict,
        emotion_colors: Dict[str, str],
        sr: int = 22050
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            audio_path: Path to audio file
            prediction_result: Prediction result dictionary
            emotion_colors: Emotion color mapping
            sr: Sample rate
            
        Returns:
            Matplotlib figure with subplots
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Load audio once
        y, sr = librosa.load(audio_path, sr=sr, duration=30)
        
        # 1. Waveform
        ax1 = fig.add_subplot(gs[0, :])
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#1f77b4', alpha=0.7)
        ax1.set_title('Waveform', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram
        ax2 = fig.add_subplot(gs[1, :])
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
        
        # 3. Probabilities
        ax3 = fig.add_subplot(gs[2, 0])
        probs = prediction_result['probabilities']
        emotions = list(probs.keys())
        prob_values = list(probs.values())
        colors = [emotion_colors.get(e, '#808080') for e in emotions]
        
        bars = ax3.barh(emotions, prob_values, color=colors, alpha=0.8)
        predicted = prediction_result['predicted_emotion']
        for bar, emotion in zip(bars, emotions):
            if emotion == predicted:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        ax3.set_xlabel('Probability')
        ax3.set_title('Emotion Probabilities', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1.0)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Top Features
        ax4 = fig.add_subplot(gs[2, 1])
        features = prediction_result['features']
        top_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:8])
        
        feature_names = [k.replace('_Mean', '').replace('_', ' ')[:20] for k in top_features.keys()]
        feature_values = list(top_features.values())
        colors_feat = ['#2ecc71' if v > 0 else '#e74c3c' for v in feature_values]
        
        ax4.barh(feature_names, feature_values, color=colors_feat, alpha=0.7)
        ax4.set_xlabel('Value')
        ax4.set_title('Top Audio Features', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        ax4.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        # Overall title
        fig.suptitle(
            f'Music Emotion Analysis: {predicted} ({prediction_result["confidence"]:.1%} confidence)',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        logger.info("✅ Dashboard created")
        return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_waveform(audio_path: Path, sr: int = 22050) -> plt.Figure:
    """Quick waveform plot."""
    viz = AudioVisualizer()
    return viz.plot_waveform(audio_path, sr=sr)


def quick_spectrogram(audio_path: Path, sr: int = 22050) -> plt.Figure:
    """Quick spectrogram plot."""
    viz = AudioVisualizer()
    return viz.plot_spectrogram(audio_path, sr=sr)
