#!/usr/bin/env python3
"""
Advanced EQ Feature Extraction for Neural Network Training
Specialized features for EQ correction tasks
"""

import os
import glob
import numpy as np
import librosa
import random
from scipy import signal
from scipy.stats import kurtosis, skew
import tqdm
from multiprocessing import Pool, cpu_count

class AdvancedBiquadEQ:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def apply_eq_band(self, audio, freq, gain_db, q=1.0, filter_type='peak'):
        """Advanced EQ with multiple filter types"""
        if abs(gain_db) < 0.1:
            return audio

        freq = np.clip(freq, 20, min(20000, self.sr//2 - 100))
        q = np.clip(q, 0.1, 10.0)
        gain_db = np.clip(gain_db, -24, 24)

        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / self.sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * q)

        if filter_type == 'peak':
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        elif filter_type == 'low_shelf':
            A = 10**(gain_db / 20.0)
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        elif filter_type == 'high_shelf':
            A = 10**(gain_db / 20.0)
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        if abs(a0) < 1e-10:
            return audio

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])

        try:
            return signal.lfilter(b, a, audio)
        except:
            return audio

    def apply_multi_eq(self, audio, eq_params):
        """Apply 5-band EQ with mixed filter types"""
        result = audio.copy()
        filter_types = ['peak', 'low_shelf', 'peak', 'peak', 'high_shelf']
        
        for i in range(5):
            freq = eq_params[i*3]
            q = eq_params[i*3 + 1]
            gain = eq_params[i*3 + 2]
            result = self.apply_eq_band(result, freq, gain, q, filter_types[i])
        return result

class AdvancedEQCorruptionGenerator:
    def __init__(self):
        # Professional frequency bands
        self.pro_bands = [
            (30, 120),    # Sub-bass
            (120, 400),   # Bass
            (400, 2000),  # Low-mids
            (2000, 8000), # High-mids
            (8000, 18000) # Highs
        ]

    def generate_realistic_eq_issues(self):
        """Generate realistic EQ problems that professionals encounter"""
        eq_params = []
        
        # Base frequencies for professional EQ
        base_freqs = [60, 250, 1000, 4000, 12000]
        
        for i, base_freq in enumerate(base_freqs):
            # Frequency with musical variations (octave-based)
            freq_variation = random.choice([0.5, 1.0, 2.0])  # Octave shifts
            freq = base_freq * freq_variation * random.uniform(0.8, 1.2)
            freq = np.clip(freq, 20, 18000)
            
            # Q values based on frequency range
            if freq < 100:
                q = random.uniform(0.5, 2.0)  # Wider for bass
            elif freq < 1000:
                q = random.uniform(1.0, 4.0)  # Medium for mids
            else:
                q = random.uniform(2.0, 8.0)  # Tighter for highs
            
            # Realistic gain ranges based on common issues
            if random.random() < 0.4:  # 40% no correction needed
                gain = 0.0
            else:
                # Common EQ adjustment ranges
                if freq < 150:  # Bass issues
                    gain = random.uniform(-8, 6)  # More cut than boost
                elif freq < 800:  # Low-mid issues
                    gain = random.uniform(-6, 4)
                elif freq < 3000:  # Presence range
                    gain = random.uniform(-4, 6)
                else:  # High frequencies
                    gain = random.uniform(-6, 8)
            
            eq_params.extend([freq, q, gain])
        
        return np.array(eq_params, dtype=np.float32)

class AdvancedAudioFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def extract_eq_features(self, audio):
        """Advanced features specifically for EQ correction"""
        if len(audio) == 0:
            return np.zeros(15, dtype=np.float32)

        try:
            # Spectral analysis with more bands
            stft = np.abs(librosa.stft(audio, n_fft=2048))
            spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sr)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=self.sr)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=self.sr)[0].mean()
            
            # Multi-band energy ratios (more granular)
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freq_bins = np.fft.rfftfreq(len(audio), 1/self.sr)
            
            bands = [
                (20, 60), (60, 120), (120, 250),    # Sub, Bass, Low-bass
                (250, 500), (500, 1000), (1000, 2000), # Low-mids
                (2000, 4000), (4000, 8000),         # Mids
                (8000, 16000), (16000, 20000)       # Highs
            ]
            
            band_energies = []
            for low, high in bands:
                mask = (freq_bins >= low) & (freq_bins <= high)
                if np.any(mask):
                    band_energies.append(np.sum(magnitude[mask]))
                else:
                    band_energies.append(0.0)
            
            total_energy = sum(band_energies) + 1e-8
            band_ratios = [e / total_energy for e in band_energies]
            
            # Time domain features
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / (rms + 1e-8)
            dynamic_range = 20 * np.log10(peak / (rms + 1e-8))
            
            # Statistical features
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            audio_skew = skew(audio)
            audio_kurtosis = kurtosis(audio)
            
            # Combine all features
            features = band_ratios + [
                spectral_centroid / 1000,
                spectral_bandwidth / 1000, 
                spectral_rolloff / 1000,
                rms, crest_factor, dynamic_range,
                zcr, audio_skew, audio_kurtosis
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(15, dtype=np.float32)

def process_audio_file(audio_file):
    """Process audio file with advanced EQ features"""
    try:
        eq_gen = AdvancedEQCorruptionGenerator()
        eq_processor = AdvancedBiquadEQ()
        feature_extractor = AdvancedAudioFeatureExtractor()
        
        # Load audio with error handling
        try:
            audio, sr = librosa.load(audio_file, sr=44100, mono=True, duration=30)
        except:
            return [], []
        
        if len(audio) < 44100:  # At least 1 second
            return [], []
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        
        file_features = []
        file_targets = []
        
        # Create multiple variations from each file
        for _ in range(3):
            # Generate realistic EQ issue
            corruption_eq = eq_gen.generate_realistic_eq_issues()
            corrupted_audio = eq_processor.apply_multi_eq(audio, corruption_eq)
            
            # Extract advanced features
            features = feature_extractor.extract_eq_features(corrupted_audio)
            
            # Target is the inverse EQ
            correction_eq = corruption_eq.copy()
            correction_eq[2::3] *= -1  # Invert gains
            
            file_features.append(features)
            file_targets.append(correction_eq)
        
        return file_features, file_targets
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return [], []

def main():
    dataset_dir = "fma_medium"
    output_dir = "./precomputed_features"
    os.makedirs(output_dir, exist_ok=True)

    print("Finding audio files...")
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process files in parallel
    num_processes = min(cpu_count(), 16)
    print(f"Using {num_processes} processes")
    
    with Pool(num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_audio_file, audio_files),
            total=len(audio_files),
            desc="Extracting EQ features"
        ))
    
    # Combine results
    all_features, all_targets = [], []
    for features, targets in results:
        all_features.extend(features)
        all_targets.extend(targets)
    
    if not all_features:
        print("No features extracted!")
        return
    
    features_array = np.stack(all_features)
    targets_array = np.stack(all_targets)
    
    print(f"Final dataset: {features_array.shape} features, {targets_array.shape} targets")
    
    # Save with compression
    np.save(os.path.join(output_dir, "features.npy"), features_array)
    np.save(os.path.join(output_dir, "targets.npy"), targets_array)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()