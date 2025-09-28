#!/usr/bin/env python3
"""
AI Audio Mastering Desktop Application with PyQt5
"""

import sys
import os
import numpy as np
import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, List

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QPushButton, QWidget, QComboBox,
                             QGroupBox, QProgressBar, QFileDialog, QMessageBox,
                             QTabWidget, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    import memryx as mx
    import pyaudio
    HAS_AUDIO = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_AUDIO = False

# Genre mapping - 8 genres from your model
GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock'
}

# Enhanced EQ presets
GENRE_EQ_PRESETS = {
    'Electronic': [2, 1, 0, 1, 3, 2, 1, 0, 2, 1, 3, 2, 1, 0, 1],
    'Electronic Instrumental': [1, 0, 1, 2, 2, 1, 0, 1, 2, 1, 1, 0, 2, 1, 2],
    'Electronic Pop': [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2],
    'Instrumental': [0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1],
    'Pop': [1, 0, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2],
    'Pop Rock': [1, 0, 2, 1, 2, 1, 0, 2, 1, 0, 1, 2, 1, 1, 1],
    'Rock': [1, 0, 2, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 0],
    'Hip-Hop': [4, 2, -1, 0, 2, 1, 0, -1, 1, 2, 1, 0, 2, 1, 0],
    'Folk': [1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 1, 0, 1],
    'International': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Experimental': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Analyzing': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Unknown': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

class AudioProcessor:
    """Audio processing functionality"""
    
    def __init__(self, genre_model_path, eq_model_path):
        self.genre_model_path = genre_model_path
        self.eq_model_path = eq_model_path
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_genre_features(self, audio_data, sample_rate=22050):
        """Extract 60 features for genre classification"""
        try:
            features = []
            
            if len(audio_data) < 2048:
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)), 'constant')
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)[0]
            
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            for mfcc in mfccs:
                features.extend([np.mean(mfcc), np.std(mfcc)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                tempo_val = tempo if np.isscalar(tempo) else tempo[0] if len(tempo) > 0 else 120.0
            except:
                tempo_val = 120.0
            features.append(tempo_val)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # Spectral contrast
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
                for band in spectral_contrast:
                    features.extend([np.mean(band), np.std(band)])
            except:
                features.extend([0.0] * 14)
            
            # Ensure exactly 60 features
            features = features[:60]
            if len(features) < 60:
                features.extend([0.0] * (60 - len(features)))
            
            feature_vector = np.array(features, dtype=np.float32)
            
            # Normalize
            if np.std(feature_vector) > 0:
                feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(60, dtype=np.float32)
    
    def predict_genre(self, features):
        """Predict genre with aggressive electronic filtering"""
        try:
            model_input = features.reshape(1, -1).astype(np.float32)
            
            if model_input.shape != (1, 60):
                if model_input.shape[1] > 60:
                    model_input = model_input[:, :60]
                elif model_input.shape[1] < 60:
                    model_input = np.pad(model_input, ((0, 0), (0, 60 - model_input.shape[1])), 'constant')
            
            genre_accl = mx.MultiStreamAsyncAccl(dfp=self.genre_model_path)
            
            prediction_result = None
            inference_complete = False
            
            def input_callback(stream_idx):
                return model_input
            
            def output_callback(stream_idx, *mxa_output):
                nonlocal prediction_result, inference_complete
                
                if len(mxa_output) > 0:
                    output = mxa_output[0]
                    
                    if len(output.shape) == 1:
                        predictions = output
                    else:
                        predictions = output[0]
                    
                    exp_output = np.exp(predictions - np.max(predictions))
                    probabilities = exp_output / np.sum(exp_output)
                    
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    top_probs = probabilities[top_indices]
                    top_genres = [GENRES.get(idx, "Unknown") for idx in top_indices]
                    
                    print(f"Raw predictions: {list(zip(top_genres, top_probs))}")
                    
                    electronic_idx = 0
                    electronic_prob = probabilities[electronic_idx]
                    
                    # Aggressive electronic filtering
                    if (electronic_prob > 0.5 and 
                        len(top_indices) > 1 and 
                        electronic_prob - top_probs[1] > 0.15):
                        final_genre = "Electronic"
                        confidence = electronic_prob
                    
                    elif top_indices[0] == electronic_idx:
                        if len(top_indices) > 1:
                            final_genre = GENRES.get(top_indices[1], "Unknown")
                            confidence = top_probs[1]
                            
                            if final_genre in ["Electronic", "Instrumental"] and len(top_indices) > 2:
                                final_genre = GENRES.get(top_indices[2], "Unknown")
                                confidence = top_probs[2]
                        else:
                            final_genre = "Unknown"
                            confidence = 0.3
                    
                    else:
                        for i, genre_idx in enumerate(top_indices):
                            genre_name = GENRES.get(genre_idx, "Unknown")
                            if genre_name != "Electronic":
                                final_genre = genre_name
                                confidence = top_probs[i]
                                break
                        else:
                            final_genre = GENRES.get(top_indices[0], "Unknown")
                            confidence = top_probs[0]
                    
                    if final_genre == "Electronic" and confidence < 0.4:
                        if len(top_indices) > 1:
                            final_genre = GENRES.get(top_indices[1], "Analyzing")
                            confidence = top_probs[1]
                        else:
                            final_genre = "Analyzing"
                            confidence = 0.3
                    
                    prediction_result = (final_genre, confidence)
                    inference_complete = True
            
            genre_accl.connect_streams(input_callback, output_callback, 1)
            
            start_time = time.time()
            while not inference_complete and (time.time() - start_time) < 5.0:
                time.sleep(0.01)
            
            genre_accl.stop()
            
            if prediction_result:
                return prediction_result
            else:
                return "Analyzing", 0.3
            
        except Exception as e:
            print(f"Genre prediction error: {e}")
            return "Unknown", 0.3
    
    def extract_eq_features(self, audio_data, sample_rate=22050):
        """Extract 12 features for EQ model"""
        try:
            features = []
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)),
                float(np.mean(spectral_rolloff)), 
                float(np.mean(spectral_bandwidth)),
                float(np.mean(zero_crossing_rate))
            ])
            
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=4)
            for mfcc in mfccs:
                features.append(float(np.mean(mfcc)))
            
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                tempo_val = tempo if np.isscalar(tempo) else tempo[0] if len(tempo) > 0 else 120.0
            except:
                tempo_val = 120.0
                
            features.extend([
                float(np.mean(chroma)),
                float(tempo_val)
            ])
            
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms))
            ])
            
            feature_vector = np.array(features[:12], dtype=np.float32)
            if len(feature_vector) < 12:
                feature_vector = np.pad(feature_vector, (0, 12 - len(feature_vector)), 'constant')
            
            if np.std(feature_vector) > 0:
                feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            print(f"EQ feature extraction failed: {str(e)}")
            return np.zeros(12, dtype=np.float32)
    
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
                    
                    eq_result = eq_adjustments.tolist()
                    inference_complete = True
            
            eq_accl.connect_streams(input_callback, output_callback, 1)
            
            start_time = time.time()
            while not inference_complete and (time.time() - start_time) < 2.0:
                time.sleep(0.01)
            
            eq_accl.stop()
            
            if eq_result:
                return eq_result
            else:
                return [0.0] * 15
            
        except Exception as e:
            print(f"EQ prediction error: {e}")
            return [0.0] * 15

class LiveAudioThread(QThread):
    """Thread for live audio processing"""
    genre_updated = pyqtSignal(str)
    eq_updated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, audio_processor, sample_rate=22050):
        super().__init__()
        self.audio_processor = audio_processor
        self.sample_rate = sample_rate
        self.running = False
        self.audio_interface = pyaudio.PyAudio()
        
    def run(self):
        self.running = True
        stream = None
        
        try:
            # Open audio stream
            stream = self.audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=2048,
                stream_callback=self.audio_callback
            )
            
            stream.start_stream()
            
            while self.running and stream.is_active():
                time.sleep(0.1)  # Process in chunks
                
        except Exception as e:
            self.error_occurred.emit(f"Audio stream error: {str(e)}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Audio stream status: {status}")
        
        # Convert audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Process audio if we have enough data
        if len(audio_data) >= 2048:
            try:
                # Extract features and predict
                genre_features = self.audio_processor.extract_genre_features(audio_data, self.sample_rate)
                genre, confidence = self.audio_processor.predict_genre(genre_features)
                
                eq_features = self.audio_processor.extract_eq_features(audio_data, self.sample_rate)
                eq_adjustments = self.audio_processor.predict_eq(eq_features)
                
                # Emit signals to update GUI
                self.genre_updated.emit(genre)
                self.eq_updated.emit(eq_adjustments)
                
            except Exception as e:
                print(f"Audio processing error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, genre_model_path, eq_model_path):
        super().__init__()
        self.audio_processor = AudioProcessor(genre_model_path, eq_model_path)
        self.live_audio_thread = None
        self.is_live = False
        
        self.setWindowTitle("AI Audio Mastering Studio")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("AI Audio Mastering Studio")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #4CAF50;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196F3, stop:1 #4CAF50);
                border-radius: 10px;
                color: white;
            }
        """)
        main_layout.addWidget(header_label)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Live Processing Tab
        live_tab = QWidget()
        tabs.addTab(live_tab, "🎵 Live Processing")
        self.setup_live_tab(live_tab)
        
        # File Processing Tab
        file_tab = QWidget()
        tabs.addTab(file_tab, "📁 File Processing")
        self.setup_file_tab(file_tab)
        
        # Status bar
        self.status_label = QLabel("Ready to process audio")
        self.status_label.setStyleSheet("QLabel { padding: 5px; background: #2b2b2b; color: #ccc; }")
        main_layout.addWidget(self.status_label)
    
    def setup_live_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Genre display
        genre_group = QGroupBox("Detected Genre")
        genre_layout = QVBoxLayout(genre_group)
        
        self.genre_label = QLabel("Ready")
        self.genre_label.setAlignment(Qt.AlignCenter)
        self.genre_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
                padding: 30px;
                background: #1a1a1a;
                border: 2px solid #4CAF50;
                border-radius: 10px;
            }
        """)
        genre_layout.addWidget(self.genre_label)
        layout.addWidget(genre_group)
        
        # EQ Controls
        eq_group = QGroupBox("5-Band Parametric EQ")
        eq_layout = QHBoxLayout(eq_group)
        
        self.eq_sliders = []
        bands = [
            ('60Hz', 'bg-red-500'),
            ('250Hz', 'bg-orange-500'), 
            ('1kHz', 'bg-yellow-500'),
            ('4kHz', 'bg-green-500'),
            ('12kHz', 'bg-blue-500')
        ]
        
        for band_name, color in bands:
            band_widget = self.create_eq_band(band_name, color)
            eq_layout.addWidget(band_widget)
            self.eq_sliders.append(band_widget)
        
        layout.addWidget(eq_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.live_button = QPushButton("🎤 Start Live Processing")
        self.live_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #45a049;
            }
            QPushButton:pressed {
                background: #3d8b40;
            }
        """)
        self.live_button.clicked.connect(self.toggle_live_processing)
        control_layout.addWidget(self.live_button)
        
        layout.addLayout(control_layout)
    
    def create_eq_band(self, band_name, color):
        band_widget = QWidget()
        layout = QVBoxLayout(band_widget)
        
        # Band name
        name_label = QLabel(band_name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(name_label)
        
        # Slider
        slider = QSlider(Qt.Vertical)
        slider.setRange(-100, 100)  # -10dB to +10dB
        slider.setValue(0)
        slider.setEnabled(False)  # Read-only in live mode
        slider.setStyleSheet("""
            QSlider::groove:vertical {
                border: 1px solid #555;
                background: #333;
                width: 10px;
                border-radius: 5px;
            }
            QSlider::handle:vertical {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                height: 20px;
                margin: 0 -5px;
                border-radius: 10px;
            }
        """)
        layout.addWidget(slider)
        
        # Value display
        value_label = QLabel("0.0 dB")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("color: #ccc;")
        layout.addWidget(value_label)
        
        # Store references
        band_widget.slider = slider
        band_widget.value_label = value_label
        
        return band_widget
    
    def setup_file_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # File selection
        file_group = QGroupBox("Audio File Processing")
        file_layout = QVBoxLayout(file_group)
        
        file_control_layout = QHBoxLayout()
        self.file_button = QPushButton("Select Audio File")
        self.file_button.clicked.connect(self.select_audio_file)
        file_control_layout.addWidget(self.file_button)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #ccc;")
        file_control_layout.addWidget(self.file_label)
        
        file_layout.addLayout(file_control_layout)
        
        # Process button
        self.process_button = QPushButton("Process Audio")
        self.process_button.clicked.connect(self.process_audio_file)
        self.process_button.setEnabled(False)
        file_layout.addWidget(self.process_button)
        
        layout.addWidget(file_group)
    
    def toggle_live_processing(self):
        if not self.is_live:
            # Start live processing
            self.live_audio_thread = LiveAudioThread(self.audio_processor)
            self.live_audio_thread.genre_updated.connect(self.update_genre_display)
            self.live_audio_thread.eq_updated.connect(self.update_eq_display)
            self.live_audio_thread.error_occurred.connect(self.show_error)
            
            self.live_audio_thread.start()
            self.live_button.setText("🛑 Stop Live Processing")
            self.live_button.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 15px 30px;
                    background: #f44336;
                    color: white;
                    border: none;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background: #da190b;
                }
            """)
            self.status_label.setText("Live audio processing started")
            self.is_live = True
        else:
            # Stop live processing
            if self.live_audio_thread:
                self.live_audio_thread.stop()
                self.live_audio_thread = None
            
            self.live_button.setText("🎤 Start Live Processing")
            self.live_button.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 15px 30px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                }
            """)
            self.status_label.setText("Live processing stopped")
            self.is_live = False
    
    def update_genre_display(self, genre):
        self.genre_label.setText(genre)
    
    def update_eq_display(self, eq_values):
        for i, band_widget in enumerate(self.eq_sliders):
            if i * 3 < len(eq_values):
                gain_value = eq_values[i * 3]
                # Convert to slider scale (-10 to +10 dB)
                slider_value = int(gain_value * 10)
                band_widget.slider.setValue(slider_value)
                band_widget.value_label.setText(f"{gain_value:.1f} dB")
    
    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_button.setEnabled(True)
    
    def process_audio_file(self):
        if hasattr(self, 'selected_file'):
            try:
                self.status_label.setText("Processing audio file...")
                
                # Load and process audio
                audio_data, sr = librosa.load(self.selected_file, sr=22050, mono=True)
                
                # Analyze genre
                genre_features = self.audio_processor.extract_genre_features(audio_data, sr)
                genre, confidence = self.audio_processor.predict_genre(genre_features)
                
                # Analyze EQ
                eq_features = self.audio_processor.extract_eq_features(audio_data, sr)
                eq_adjustments = self.audio_processor.predict_eq(eq_features)
                
                # Update display
                self.update_genre_display(genre)
                self.update_eq_display(eq_adjustments)
                
                self.status_label.setText(f"File processed: {genre} detected")
                
            except Exception as e:
                self.show_error(f"File processing error: {str(e)}")
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")
    
    def closeEvent(self, event):
        # Clean up resources
        if self.live_audio_thread and self.live_audio_thread.isRunning():
            self.live_audio_thread.stop()
            self.live_audio_thread.wait()
        
        if hasattr(self.audio_processor, 'audio_interface'):
            self.audio_processor.audio_interface.terminate()
        
        event.accept()

def main():
    # Check for model paths
    if len(sys.argv) != 3:
        print("Usage: python mastering_app.py <genre_model.dfp> <eq_model.dfp>")
        sys.exit(1)
    
    genre_model_path = sys.argv[1]
    eq_model_path = sys.argv[2]
    
    if not os.path.exists(genre_model_path) or not os.path.exists(eq_model_path):
        print("Error: Model files not found")
        sys.exit(1)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Create and show main window
    window = MainWindow(genre_model_path, eq_model_path)
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()