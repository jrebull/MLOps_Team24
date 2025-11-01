"""
Audio Feature Extraction Module

Extracts 50 acoustic features from audio files to match the model's expected input.
Uses librosa for feature extraction with approximations for MIRtoolbox features.

Author: MLOps Team 24
Date: November 2024
"""

import numpy as np
import librosa
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    Extracts 50 acoustic features from audio files following SOLID principles.
    
    Single Responsibility: Only handles feature extraction from audio
    Open/Closed: Extensible for new feature types
    """
    
    def __init__(self, sr: int = 22050, duration: Optional[float] = 30.0):
        """
        Initialize feature extractor.
        
        Args:
            sr: Sample rate for audio loading
            duration: Duration in seconds to load (None = full audio)
        """
        self.sr = sr
        self.duration = duration
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> list:
        """Get list of 50 feature names in correct order."""
        try:
            from ..config import FEATURE_NAMES
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import FEATURE_NAMES
        return FEATURE_NAMES
    
    def load_audio(self, audio_path) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling.
        
        Args:
            audio_path: Path to audio file or UploadedFile
            
        Returns:
            Tuple of (audio_signal, sample_rate)
        """
        try:
            # Handle both Path and UploadedFile
            if hasattr(audio_path, 'read'):
                # It's an UploadedFile
                import io
                audio_bytes = io.BytesIO(audio_path.read())
                y, sr = librosa.load(audio_bytes, sr=self.sr, duration=self.duration, mono=True)
                name = audio_path.name
            else:
                # It's a Path
                audio_path = Path(audio_path)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
                name = audio_path.name
            
            logger.info(f"✅ Audio loaded: {name} ({len(y)/sr:.2f}s)")
            return y, sr
        except Exception as e:
            logger.error(f"❌ Failed to load audio: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
    
    def extract_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract all 50 features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with feature_name: value pairs
        """
        y, sr = self.load_audio(audio_path)
        
        features = {}
        
        # 1. RMS Energy
        features["_RMSenergy_Mean"] = self._extract_rms_energy(y)
        
        # 2. Low Energy
        features["_Lowenergy_Mean"] = self._extract_low_energy(y)
        
        # 3. Fluctuation (approximated as tempo variability)
        features["_Fluctuation_Mean"] = self._extract_fluctuation(y, sr)
        
        # 4. Tempo
        features["_Tempo_Mean"] = self._extract_tempo(y, sr)
        
        # 5-17. MFCCs (13 coefficients)
        mfccs = self._extract_mfccs(y, sr, n_mfcc=13)
        for i, mfcc_val in enumerate(mfccs, start=1):
            features[f"_MFCC_Mean_{i}"] = mfcc_val
        
        # 18-19. Roughness
        roughness_mean, roughness_slope = self._extract_roughness(y, sr)
        features["_Roughness_Mean"] = roughness_mean
        features["_Roughness_Slope"] = roughness_slope
        
        # 20. Zero Crossing Rate
        features["_Zero-crossingrate_Mean"] = self._extract_zcr(y)
        
        # 21-22. Attack Time
        attack_mean, attack_slope = self._extract_attack_time(y, sr)
        features["_AttackTime_Mean"] = attack_mean
        features["_AttackTime_Slope"] = attack_slope
        
        # 23. Spectral Rolloff
        features["_Rolloff_Mean"] = self._extract_rolloff(y, sr)
        
        # 24. Event Density (onset density)
        features["_Eventdensity_Mean"] = self._extract_event_density(y, sr)
        
        # 25. Pulse Clarity (rhythm strength)
        features["_Pulseclarity_Mean"] = self._extract_pulse_clarity(y, sr)
        
        # 26. Brightness
        features["_Brightness_Mean"] = self._extract_brightness(y, sr)
        
        # 27. Spectral Centroid
        features["_Spectralcentroid_Mean"] = self._extract_spectral_centroid(y, sr)
        
        # 28. Spectral Spread
        features["_Spectralspread_Mean"] = self._extract_spectral_spread(y, sr)
        
        # 29. Spectral Skewness
        features["_Spectralskewness_Mean"] = self._extract_spectral_skewness(y, sr)
        
        # 30. Spectral Kurtosis
        features["_Spectralkurtosis_Mean"] = self._extract_spectral_kurtosis(y, sr)
        
        # 31. Spectral Flatness
        features["_Spectralflatness_Mean"] = self._extract_spectral_flatness(y, sr)
        
        # 32. Entropy of Spectrum
        features["_EntropyofSpectrum_Mean"] = self._extract_spectral_entropy(y, sr)
        
        # 33-44. Chromagram (12 chroma features)
        chroma = self._extract_chromagram(y, sr)
        for i, chroma_val in enumerate(chroma, start=1):
            features[f"_Chromagram_Mean_{i}"] = chroma_val
        
        # 45-50. Harmonic Change Detection Function
        hcdf_features = self._extract_hcdf(y, sr)
        features["_HarmonicChangeDetectionFunction_Mean"] = hcdf_features["mean"]
        features["_HarmonicChangeDetectionFunction_Std"] = hcdf_features["std"]
        features["_HarmonicChangeDetectionFunction_Slope"] = hcdf_features["slope"]
        features["_HarmonicChangeDetectionFunction_PeriodFreq"] = hcdf_features["period_freq"]
        features["_HarmonicChangeDetectionFunction_PeriodAmp"] = hcdf_features["period_amp"]
        features["_HarmonicChangeDetectionFunction_PeriodEntropy"] = hcdf_features["period_entropy"]
        
        logger.info(f"✅ Extracted {len(features)} features")
        return features
    
    # ========================================================================
    # FEATURE EXTRACTION METHODS
    # ========================================================================
    
    def _extract_rms_energy(self, y: np.ndarray) -> float:
        """Extract RMS energy."""
        rms = librosa.feature.rms(y=y)
        return float(np.mean(rms))
    
    def _extract_low_energy(self, y: np.ndarray) -> float:
        """Extract low energy ratio (frames below mean)."""
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms)
        low_energy_ratio = np.sum(rms < threshold) / len(rms)
        return float(low_energy_ratio)
    
    def _extract_fluctuation(self, y: np.ndarray, sr: int) -> float:
        """Extract fluctuation (tempo variability approximation)."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        fluctuation = np.std(onset_env)
        return float(fluctuation)
    
    def _extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo in BPM."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    
    def _extract_mfccs(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC mean values."""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)
    
    def _extract_roughness(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Extract roughness (dissonance approximation)."""
        # Roughness approximated using spectral dissonance
        spec = np.abs(librosa.stft(y))
        roughness = np.mean(np.diff(spec, axis=0) ** 2)
        
        # Roughness slope (trend over time)
        roughness_time = np.mean(np.diff(spec, axis=0) ** 2, axis=0)
        slope = np.polyfit(np.arange(len(roughness_time)), roughness_time, 1)[0]
        
        return float(roughness), float(slope)
    
    def _extract_zcr(self, y: np.ndarray) -> float:
        """Extract zero crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(y)
        return float(np.mean(zcr))
    
    def _extract_attack_time(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Extract attack time (onset characteristics)."""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        
        if len(onset_frames) > 1:
            # Attack time as mean time between onsets
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            attack_mean = float(np.mean(np.diff(onset_times)))
            attack_slope = float(np.polyfit(np.arange(len(onset_times)), onset_times, 1)[0])
        else:
            attack_mean = 0.0
            attack_slope = 0.0
        
        return attack_mean, attack_slope
    
    def _extract_rolloff(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral rolloff."""
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        return float(np.mean(rolloff))
    
    def _extract_event_density(self, y: np.ndarray, sr: int) -> float:
        """Extract event density (onset density)."""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        density = len(onset_frames) / duration if duration > 0 else 0
        return float(density)
    
    def _extract_pulse_clarity(self, y: np.ndarray, sr: int) -> float:
        """Extract pulse clarity (rhythm strength)."""
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        pulse_clarity = float(np.max(tempogram) / (np.mean(tempogram) + 1e-6))
        return pulse_clarity
    
    def _extract_brightness(self, y: np.ndarray, sr: int) -> float:
        """Extract brightness (spectral centroid normalized)."""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = np.mean(centroid) / (sr / 2)  # Normalize
        return float(brightness)
    
    def _extract_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral centroid."""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return float(np.mean(centroid))
    
    def _extract_spectral_spread(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral spread (bandwidth)."""
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return float(np.mean(bandwidth))
    
    def _extract_spectral_skewness(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral skewness."""
        spec = np.abs(librosa.stft(y))
        skewness = stats.skew(spec, axis=0)
        return float(np.mean(skewness))
    
    def _extract_spectral_kurtosis(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral kurtosis."""
        spec = np.abs(librosa.stft(y))
        kurtosis = stats.kurtosis(spec, axis=0)
        return float(np.mean(kurtosis))
    
    def _extract_spectral_flatness(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral flatness."""
        flatness = librosa.feature.spectral_flatness(y=y)
        return float(np.mean(flatness))
    
    def _extract_spectral_entropy(self, y: np.ndarray, sr: int) -> float:
        """Extract spectral entropy."""
        spec = np.abs(librosa.stft(y))
        # Normalize spectrum
        spec_norm = spec / (np.sum(spec, axis=0, keepdims=True) + 1e-10)
        # Calculate entropy
        entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10), axis=0)
        return float(np.mean(entropy))
    
    def _extract_chromagram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract 12 chromagram features."""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return np.mean(chroma, axis=1)
    
    def _extract_hcdf(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract Harmonic Change Detection Function features.
        Approximated using harmonic-percussive separation.
        """
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # HCDF as difference in harmonic content over time
        harmonic_strength = np.abs(librosa.stft(y_harmonic))
        hcdf = np.diff(np.sum(harmonic_strength, axis=0))
        
        # Basic statistics
        mean_hcdf = float(np.mean(hcdf))
        std_hcdf = float(np.std(hcdf))
        
        # Slope (trend)
        slope = float(np.polyfit(np.arange(len(hcdf)), hcdf, 1)[0])
        
        # Periodicity analysis using autocorrelation
        autocorr = np.correlate(hcdf, hcdf, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks for periodicity
        peaks, properties = find_peaks(autocorr, height=np.max(autocorr)*0.5)
        
        if len(peaks) > 0:
            period_freq = float(1.0 / (peaks[0] + 1))  # Dominant frequency
            period_amp = float(properties['peak_heights'][0])
            # Entropy of peak distribution
            peak_heights = properties['peak_heights']
            peak_norm = peak_heights / np.sum(peak_heights)
            period_entropy = float(-np.sum(peak_norm * np.log2(peak_norm + 1e-10)))
        else:
            period_freq = 0.0
            period_amp = 0.0
            period_entropy = 0.0
        
        return {
            "mean": mean_hcdf,
            "std": std_hcdf,
            "slope": slope,
            "period_freq": period_freq,
            "period_amp": period_amp,
            "period_entropy": period_entropy
        }
    
    def extract_features_to_dataframe(self, audio_path: Path):
        """
        Extract features and return as DataFrame (for model input).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            pandas DataFrame with single row of features
        """
        import pandas as pd
        
        features = self.extract_features(audio_path)
        
        # Ensure correct order
        ordered_features = {name: features[name] for name in self.feature_names}
        
        return pd.DataFrame([ordered_features])


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def extract_features_from_audio(audio_path: Path, sr: int = 22050) -> Dict[str, float]:
    """
    Convenience function to extract features from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        Dictionary of features
    """
    extractor = AudioFeatureExtractor(sr=sr)
    return extractor.extract_features(audio_path)


def extract_features_to_df(audio_path: Path, sr: int = 22050):
    """
    Convenience function to extract features as DataFrame.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        pandas DataFrame with features
    """
    extractor = AudioFeatureExtractor(sr=sr)
    return extractor.extract_features_to_dataframe(audio_path)
