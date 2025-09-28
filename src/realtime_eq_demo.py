#!/usr/bin/env python3
"""
Real-time Genre Classification + Live EQ Demo - Fixed Version
- Initial 5-second window for genre classification  
- Then 1-second windows for continuous EQ adjustments
- Real-time audio playback with EQ applied
"""

import numpy as np
import sys
import os
import time
import threading
import queue
from collections import deque

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
    from scipy import signal
    HAS_AUDIO = True
except ImportError as e:
    print(f"Audio libraries not available: {e}")
    print("Install with: pip install librosa soundfile pyaudio scipy")
    sys.exit(1)

# Genre mapping
GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock',
    8: 'Blues', 9: 'Jazz', 10: 'Classical', 11: 'Old-Time / Historic',
    12: 'Country', 13: 'Easy Listening', 14: 'Soul-RnB', 15: 'Spoken'
}

class RealTimeEQProcessor:
    def __init__(self, genre_model_path, eq_model_path, sample_rate=22050):
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.buffer_duration = 5.0
        self.update_interval = 1.0
        
        # Store model paths
        self.genre_model_path = genre_model_path
        self.eq_model_path = eq_model_path
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * self.buffer_duration))
        self.output_buffer = queue.Queue(maxsize=10)
        
        # State
        self.current_genre = "Unknown"
        self.current_eq = np.zeros(15)
        self.is_running = False
        self.genre_detected = False
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        # Verify models exist
        if not os.path.exists(genre_model_path):
            print(f"Genre model not found: {genre_model_path}")
            sys.exit(1)
        if not os.path.exists(eq_model_path):
            print(f"EQ model not found: {eq_model_path}")
            sys.exit(1)
        
        print("Models verified")
    
    def extract_genre_features(self, audio_data):
        """Extract 60 features for genre classification"""
        if len(audio_data) == 0:
            return np.random.uniform(-2, 2, 60).astype(np.float32)
        
        audio_array = np.array(audio_data, dtype=np.float32)
        
        try:
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_array, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=13)
            
            # Chroma and tempo
            chroma = librosa.feature.chroma_stft(y=audio_array, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_array)[0]
            
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
    
    def extract_eq_features(self, audio_data):
        """Extract 12 features for EQ model"""
        if len(audio_data) == 0:
            return np.random.uniform(-2, 2, 12).astype(np.float32)
        
        audio_array = np.array(audio_data, dtype=np.float32)
        
        try:
            features = []
            
            # Spectral features (4)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_array, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.mean(spectral_rolloff)), 
                float(np.mean(spectral_bandwidth)),
                float(np.mean(zero_crossing_rate))
            ])
            
            # MFCC features (4 - first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=4)
            for mfcc in mfccs:
                features.append(float(np.mean(mfcc)))
            
            # Chroma and tempo (2)
            chroma = librosa.feature.chroma_stft(y=audio_array, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
            
            features.extend([
                float(np.mean(chroma)),
                float(tempo) if np.isscalar(tempo) else float(tempo[0])
            ])
            
            # RMS energy (2)
            rms = librosa.feature.rms(y=audio_array)[0]
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
        """Predict genre using one-time accelerator"""
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
        """Predict EQ adjustments using one-time accelerator"""
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
        """Apply EQ adjustments to audio chunk"""
        if len(audio_chunk) == 0 or np.all(eq_adjustments == 0):
            return audio_chunk
        
        # EQ bands (Hz): simplified implementation
        processed_audio = audio_chunk.copy()
        
        # Simple gain adjustment for demo purposes
        # In practice, you'd use proper digital filters
        for i, gain_db in enumerate(eq_adjustments):
            if abs(gain_db) > 0.1:
                gain_linear = 10 ** (gain_db / 20)
                processed_audio = processed_audio * (1.0 + 0.1 * gain_linear)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0.95:
            processed_audio = processed_audio * (0.95 / max_val)
        
        return processed_audio
    
    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Input audio callback"""
        if not self.is_running:
            return (in_data, pyaudio.paComplete)
        
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """Output audio callback"""
        if not self.output_buffer.empty():
            try:
                output_data = self.output_buffer.get_nowait()
                if len(output_data) >= frame_count:
                    return (output_data[:frame_count].astype(np.float32).tobytes(), pyaudio.paContinue)
            except:
                pass
        
        silence = np.zeros(frame_count, dtype=np.float32)
        return (silence.tobytes(), pyaudio.paContinue)
    
    def processing_loop(self):
        """Main processing loop"""
        print("Starting audio processing...")
        print("Analyzing initial 5-second window for genre detection...")
        
        last_eq_update = time.time()
        
        while self.is_running:
            try:
                if len(self.audio_buffer) >= int(self.sample_rate * 1.0):
                    
                    current_audio = list(self.audio_buffer)
                    
                    # Genre classification (only if not detected yet)
                    if not self.genre_detected and len(current_audio) >= int(self.sample_rate * self.buffer_duration):
                        print("Analyzing genre...")
                        features = self.extract_genre_features(current_audio)
                        genre_result = self.predict_genre(features)
                        
                        if genre_result:
                            self.current_genre, confidence = genre_result
                            self.genre_detected = True
                            print(f"Genre detected: {self.current_genre} (confidence: {confidence:.3f})")
                    
                    # EQ prediction and application
                    current_time = time.time()
                    if current_time - last_eq_update >= self.update_interval:
                        
                        # Get recent audio for EQ analysis
                        recent_audio = current_audio[-int(self.sample_rate * 1.0):]
                        eq_features = self.extract_eq_features(recent_audio)
                        
                        # Predict EQ adjustments
                        new_eq = self.predict_eq(eq_features)
                        
                        # Smooth EQ changes
                        alpha = 0.3
                        self.current_eq = alpha * new_eq + (1 - alpha) * self.current_eq
                        
                        print(f"EQ Update: {self.current_eq[:5].round(2)}... (Genre: {self.current_genre})")
                        
                        last_eq_update = current_time
                    
                    # Apply EQ to current audio chunk
                    if len(current_audio) >= self.chunk_size:
                        audio_chunk = np.array(current_audio[-self.chunk_size:], dtype=np.float32)
                        processed_chunk = self.apply_eq(audio_chunk, self.current_eq)
                        
                        if not self.output_buffer.full():
                            self.output_buffer.put(processed_chunk)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start real-time processing"""
        print("Starting real-time EQ demo...")
        
        try:
            self.is_running = True
            
            # Start input stream
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_input_callback
            )
            
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
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            print("Real-time processing started!")
            print("Speak, play music, or make sounds...")
            print("Initial 5-second analysis, then 1-second EQ updates")
            print("Press Ctrl+C to stop")
            
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
        except Exception as e:
            print(f"Error during processing: {e}")
            self.stop()
    
    def stop(self):
        """Stop processing and cleanup"""
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("Real-time EQ demo stopped")

def main():
    if len(sys.argv) != 3:
        print("Usage: python realtime_eq_fixed.py <genre_model.dfp> <eq_model.dfp>")
        sys.exit(1)
    
    genre_model_path = sys.argv[1]
    eq_model_path = sys.argv[2]
    
    processor = RealTimeEQProcessor(genre_model_path, eq_model_path)
    processor.start()

if __name__ == "__main__":
    main()