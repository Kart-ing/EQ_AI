#!/usr/bin/env python3
"""
File-based EQ Demo - Play audio file with real-time EQ processing
"""

import numpy as np
import sys
import os
import time
import threading
import queue

# Import MemryX
try:
    import memryx as mx
except ImportError as e:
    print(f"MemryX SDK not found: {e}")
    sys.exit(1)

# Import audio libraries
try:
    import librosa
    import soundfile as sf
    import pyaudio
    HAS_AUDIO = True
except ImportError as e:
    print(f"Audio libraries not available: {e}")
    sys.exit(1)

# Genre mapping
GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock',
    8: 'Blues', 9: 'Jazz', 10: 'Classical', 11: 'Old-Time / Historic',
    12: 'Country', 13: 'Easy Listening', 14: 'Soul-RnB', 15: 'Spoken'
}

class FileEQProcessor:
    def __init__(self, genre_model_path, eq_model_path, audio_file_path, sample_rate=22050):
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.update_interval = 1.0  # EQ update every 1 second
        
        # Store paths
        self.genre_model_path = genre_model_path
        self.eq_model_path = eq_model_path
        self.audio_file_path = audio_file_path
        
        # Audio data
        self.audio_data = None
        self.audio_position = 0
        self.total_samples = 0
        
        # State
        self.current_genre = "Unknown"
        self.current_eq = np.zeros(15)
        self.is_playing = False
        self.genre_detected = False
        
        # Audio output
        self.audio = pyaudio.PyAudio()
        self.output_stream = None
        
        # Load audio file
        self.load_audio_file()
        
        print(f"File EQ processor initialized")
        print(f"Audio duration: {self.total_samples / sample_rate:.1f} seconds")
    
    def load_audio_file(self):
        """Load the audio file"""
        try:
            print(f"Loading audio file: {self.audio_file_path}")
            self.audio_data, sr = librosa.load(self.audio_file_path, sr=self.sample_rate, mono=True)
            self.total_samples = len(self.audio_data)
            print(f"Loaded {len(self.audio_data)} samples at {sr}Hz")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            sys.exit(1)
    
    def extract_genre_features(self, audio_segment):
        """Extract 60 features for genre classification"""
        try:
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
            
            # Chroma and tempo
            chroma = librosa.feature.chroma_stft(y=audio_segment, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=self.sample_rate)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_segment)[0]
            
            # Aggregate features
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth)),
                float(np.mean(zero_crossing_rate)), float(np.std(zero_crossing_rate))
            ])
            
            # MFCC stats
            for mfcc in mfccs:
                features.extend([float(np.mean(mfcc)), float(np.std(mfcc))])
            
            # Chroma stats
            features.extend([float(np.mean(chroma)), float(np.std(chroma))])
            
            # Tempo
            features.append(float(tempo) if np.isscalar(tempo) else float(tempo[0]))
            
            # RMS stats
            features.extend([float(np.mean(rms)), float(np.std(rms))])
            
            # Ensure exactly 60 features
            feature_vector = np.array(features, dtype=np.float32)
            if len(feature_vector) > 60:
                feature_vector = feature_vector[:60]
            elif len(feature_vector) < 60:
                feature_vector = np.pad(feature_vector, (0, 60 - len(feature_vector)), 'constant')
            
            # Normalize
            feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
            
            return feature_vector
            
        except Exception as e:
            print(f"Genre feature extraction error: {e}")
            return np.random.uniform(-2, 2, 60).astype(np.float32)
    
    def extract_eq_features(self, audio_segment):
        """Extract 12 features for EQ model"""
        try:
            features = []
            
            # Spectral features (4)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.mean(spectral_rolloff)), 
                float(np.mean(spectral_bandwidth)),
                float(np.mean(zero_crossing_rate))
            ])
            
            # MFCC features (4)
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=4)
            for mfcc in mfccs:
                features.append(float(np.mean(mfcc)))
            
            # Chroma and tempo (2)
            chroma = librosa.feature.chroma_stft(y=audio_segment, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=self.sample_rate)
            
            features.extend([
                float(np.mean(chroma)),
                float(tempo) if np.isscalar(tempo) else float(tempo[0])
            ])
            
            # RMS energy (2)
            rms = librosa.feature.rms(y=audio_segment)[0]
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms))
            ])
            
            # Ensure exactly 12 features
            feature_vector = np.array(features[:12], dtype=np.float32)
            if len(feature_vector) < 12:
                feature_vector = np.pad(feature_vector, (0, 12 - len(feature_vector)), 'constant')
            
            # Normalize
            feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
            
            return feature_vector
            
        except Exception as e:
            print(f"EQ feature extraction error: {e}")
            return np.random.uniform(-2, 2, 12).astype(np.float32)
    
    def predict_genre(self, features):
        """Predict genre using MemryX model"""
        try:
            genre_accl = mx.MultiStreamAsyncAccl(dfp=self.genre_model_path)
            
            prediction_result = None
            inference_complete = False
            
            def input_callback(stream_idx):
                return features.reshape(1, 1, 60)
            
            def output_callback(stream_idx, *mxa_output):
                nonlocal prediction_result, inference_complete
                
                if len(mxa_output) > 0:
                    output = mxa_output[0]
                    if len(output.shape) > 1:
                        predictions = output[0]
                    else:
                        predictions = output
                    
                    predicted_class = np.argmax(predictions)
                    exp_output = np.exp(predictions - np.max(predictions))
                    probabilities = exp_output / np.sum(exp_output)
                    confidence = probabilities[predicted_class]
                    genre_name = GENRES.get(predicted_class, f"Unknown ({predicted_class})")
                    
                    prediction_result = (genre_name, confidence)
                    inference_complete = True
            
            genre_accl.connect_streams(input_callback, output_callback, 1)
            
            start_time = time.time()
            while not inference_complete and (time.time() - start_time) < 2.0:
                time.sleep(0.01)
            
            genre_accl.stop()
            
            return prediction_result
            
        except Exception as e:
            print(f"Genre prediction error: {e}")
            return ("Unknown", 0.0)
    
    def predict_eq(self, features):
        """Predict EQ adjustments using MemryX model"""
        try:
            eq_accl = mx.MultiStreamAsyncAccl(dfp=self.eq_model_path)
            
            eq_result = None
            inference_complete = False
            
            def input_callback(stream_idx):
                return features.reshape(1, 12)
            
            def output_callback(stream_idx, *mxa_output):
                nonlocal eq_result, inference_complete
                
                if len(mxa_output) > 0:
                    output = mxa_output[0]
                    if len(output.shape) > 1:
                        eq_adjustments = output[0]
                    else:
                        eq_adjustments = output
                    
                    eq_result = eq_adjustments
                    inference_complete = True
            
            eq_accl.connect_streams(input_callback, output_callback, 1)
            
            start_time = time.time()
            while not inference_complete and (time.time() - start_time) < 1.0:
                time.sleep(0.01)
            
            eq_accl.stop()
            
            return eq_result if eq_result is not None else np.zeros(15)
            
        except Exception as e:
            print(f"EQ prediction error: {e}")
            return np.zeros(15)
    
    def apply_eq(self, audio_chunk, eq_adjustments):
        """Apply simple EQ adjustments to audio chunk"""
        if np.all(eq_adjustments == 0):
            return audio_chunk
        
        # Very simple EQ - just slight gain adjustments
        processed_audio = audio_chunk.copy()
        
        # Apply a mild overall adjustment based on EQ
        overall_gain = 1.0 + 0.05 * np.mean(eq_adjustments)  # Very small adjustment
        processed_audio = processed_audio * overall_gain
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0.95:
            processed_audio = processed_audio * (0.95 / max_val)
        
        return processed_audio
    
    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """Output audio callback"""
        if not self.is_playing or self.audio_position >= self.total_samples:
            # End of audio or stopped
            silence = np.zeros(frame_count, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paComplete)
        
        # Get next chunk of audio
        end_pos = min(self.audio_position + frame_count, self.total_samples)
        audio_chunk = self.audio_data[self.audio_position:end_pos]
        
        # Pad if necessary
        if len(audio_chunk) < frame_count:
            audio_chunk = np.pad(audio_chunk, (0, frame_count - len(audio_chunk)), 'constant')
        
        # Apply EQ processing
        processed_chunk = self.apply_eq(audio_chunk, self.current_eq)
        
        # Update position
        self.audio_position = end_pos
        
        return (processed_chunk.astype(np.float32).tobytes(), pyaudio.paContinue)
    
    def processing_loop(self):
        """Background processing for genre detection and EQ updates"""
        print("Starting background processing...")
        
        # Initial genre detection (first 5 seconds)
        if not self.genre_detected and self.total_samples >= 5 * self.sample_rate:
            print("Analyzing genre from first 5 seconds...")
            genre_segment = self.audio_data[:5 * self.sample_rate]
            features = self.extract_genre_features(genre_segment)
            genre_result = self.predict_genre(features)
            
            if genre_result:
                self.current_genre, confidence = genre_result
                self.genre_detected = True
                print(f"Genre detected: {self.current_genre} (confidence: {confidence:.3f})")
        
        # EQ updates during playback
        last_eq_update = time.time()
        
        while self.is_playing and self.audio_position < self.total_samples:
            current_time = time.time()
            
            if current_time - last_eq_update >= self.update_interval:
                # Get current audio segment for EQ analysis
                current_pos = self.audio_position
                segment_start = max(0, current_pos - self.sample_rate)  # Last 1 second
                segment_end = min(current_pos + self.sample_rate, self.total_samples)
                
                if segment_end > segment_start:
                    audio_segment = self.audio_data[segment_start:segment_end]
                    eq_features = self.extract_eq_features(audio_segment)
                    new_eq = self.predict_eq(eq_features)
                    
                    # Smooth EQ changes
                    alpha = 0.3
                    self.current_eq = alpha * new_eq + (1 - alpha) * self.current_eq
                    
                    current_time_pos = self.audio_position / self.sample_rate
                    print(f"[{current_time_pos:.1f}s] EQ Update: {self.current_eq[:5].round(2)}... (Genre: {self.current_genre})")
                
                last_eq_update = current_time
            
            time.sleep(0.1)
    
    def start(self):
        """Start file playback with EQ processing"""
        print("Starting file EQ demo...")
        
        try:
            self.is_playing = True
            self.audio_position = 0
            
            # Start output stream
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_output_callback
            )
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.output_stream.start_stream()
            
            print("Playback started!")
            print("Press Ctrl+C to stop")
            
            # Wait for playback to finish or interruption
            while self.is_playing and self.audio_position < self.total_samples:
                time.sleep(0.1)
            
            print("Playback finished")
            self.stop()
            
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
        except Exception as e:
            print(f"Error during playback: {e}")
            self.stop()
    
    def stop(self):
        """Stop playback and cleanup"""
        self.is_playing = False
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("File EQ demo stopped")

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_eq_demo.py <genre_model.dfp> <eq_model.dfp> <audio_file>")
        sys.exit(1)
    
    genre_model_path = sys.argv[1]
    eq_model_path = sys.argv[2]
    audio_file_path = sys.argv[3]
    
    # Check files exist
    for path in [genre_model_path, eq_model_path, audio_file_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
    
    processor = FileEQProcessor(genre_model_path, eq_model_path, audio_file_path)
    processor.start()

if __name__ == "__main__":
    main()