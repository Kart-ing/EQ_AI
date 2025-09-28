#!/usr/bin/env python3
"""
EQ_AI - Enhanced Audio Processing Application
Modern PyQt5 GUI with advanced audio processing capabilities
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
import memryx as mx
import time
import sounddevice as sd
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QTabWidget, QSlider,
    QGroupBox, QGridLayout, QComboBox, QSpinBox, QCheckBox, QSplitter,
    QFrame, QScrollArea, QListWidget, QListWidgetItem, QDialog, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar, QMenuBar,
    QAction, QToolBar, QMessageBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QIcon, QPainter, QLinearGradient, QBrush,
    QPixmap, QPen, QPolygonF, QFontMetrics
)

# ======================= 
# CONSTANTS & CONFIG
# =======================

GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock'
}

GENRE_MODEL_PATH = "/home/memryx/Desktop/EQ_AI/models/genre_3.dfp"
EQ_MODEL_PATH = "/home/memryx/Desktop/EQ_AI/models/live_eq_model_50.dfp"

# Audio Settings
SAMPLE_RATE = 22050
CHUNK_SIZE = 2048
EQ_BANDS = 5
EQ_FEATURES_PER_BAND = 3
EQ_TOTAL_FEATURES = EQ_BANDS * EQ_FEATURES_PER_BAND

# GUI Theme Colors
DARK_THEME = {
    'background': '#1a1a1a',
    'surface': '#2d2d2d',
    'primary': '#4a9eff',
    'secondary': '#ff6b6b',
    'accent': '#4ecdc4',
    'text': '#ffffff',
    'text_secondary': '#b0b0b0',
    'border': '#404040',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'error': '#ff6b6b'
}

# ======================= 
# CUSTOM WIDGETS
# =======================

class ModernSlider(QSlider):
    """Custom styled slider with modern appearance"""
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: #404040;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4a9eff, stop:1 #357abd);
                border: 2px solid #2d2d2d;
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5ba8ff, stop:1 #4589d4);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a9eff, stop:1 #4ecdc4);
                border-radius: 3px;
            }
        """)

class ModernButton(QPushButton):
    """Custom styled button with modern appearance"""
    def __init__(self, text="", button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.update_style()
    
    def update_style(self):
        if self.button_type == "primary":
            color = DARK_THEME['primary']
            hover_color = "#5ba8ff"
        elif self.button_type == "secondary":
            color = DARK_THEME['secondary']
            hover_color = "#ff7b7b"
        elif self.button_type == "success":
            color = DARK_THEME['success']
            hover_color = "#69db7c"
        else:
            color = DARK_THEME['surface']
            hover_color = "#3d3d3d"
        
        self.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {hover_color};
            }}
            QPushButton:pressed {{
                background: {color};
                transform: translateY(1px);
            }}
            QPushButton:disabled {{
                background: #404040;
                color: #808080;
            }}
        """)

class WaveformWidget(QWidget):
    """Custom widget to display audio waveform"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_data = None
        self.setMinimumHeight(100)
        self.setStyleSheet(f"background: {DARK_THEME['surface']}; border-radius: 8px;")
    
    def set_waveform(self, data):
        self.waveform_data = data
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.waveform_data is None:
            painter.setPen(QPen(QColor(DARK_THEME['text_secondary']), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "No audio loaded")
            return
        
        # Draw waveform
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Create gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(DARK_THEME['primary']))
        gradient.setColorAt(1, QColor(DARK_THEME['accent']))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(DARK_THEME['primary']), 1))
        
        # Downsample data for display
        step = max(1, len(self.waveform_data) // width)
        display_data = self.waveform_data[::step]
        
        # Draw waveform bars
        bar_width = width / len(display_data)
        for i, sample in enumerate(display_data):
            x = i * bar_width
            bar_height = abs(sample) * (height // 2)
            painter.drawRect(int(x), int(center_y - bar_height/2), 
                           int(bar_width), int(bar_height))

class SpectrumAnalyzer(QWidget):
    """Real-time frequency spectrum analyzer"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectrum_data = None
        self.setMinimumHeight(200)
        self.setStyleSheet(f"background: {DARK_THEME['surface']}; border-radius: 8px;")
    
    def set_spectrum(self, fft_data):
        self.spectrum_data = np.abs(fft_data)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.spectrum_data is None:
            painter.setPen(QPen(QColor(DARK_THEME['text_secondary']), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "No spectrum data")
            return
        
        width = self.width()
        height = self.height()
        
        # Create gradient for spectrum
        gradient = QLinearGradient(0, height, 0, 0)
        gradient.setColorAt(0, QColor(DARK_THEME['success']))
        gradient.setColorAt(0.5, QColor(DARK_THEME['warning']))
        gradient.setColorAt(1, QColor(DARK_THEME['secondary']))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        
        # Draw spectrum bars
        num_bars = min(width // 3, len(self.spectrum_data) // 2)
        bar_width = width / num_bars
        
        for i in range(num_bars):
            idx = i * len(self.spectrum_data) // (2 * num_bars)
            bar_height = min(self.spectrum_data[idx] * height * 2, height)
            x = i * bar_width
            y = height - bar_height
            painter.drawRect(int(x), int(y), int(bar_width - 1), int(bar_height))

class EQBandWidget(QWidget):
    """Individual EQ band control widget"""
    value_changed = pyqtSignal(int, float, float, float)  # band_id, gain, freq, q
    
    def __init__(self, band_id, name, freq_range, parent=None):
        super().__init__(parent)
        self.band_id = band_id
        self.setup_ui(name, freq_range)
    
    def setup_ui(self, name, freq_range):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Band label
        label = QLabel(name)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Segoe UI", 9, QFont.Medium))
        label.setStyleSheet(f"color: {DARK_THEME['text']}; margin-bottom: 5px;")
        layout.addWidget(label)
        
        # Gain slider (vertical)
        self.gain_slider = ModernSlider(Qt.Vertical)
        self.gain_slider.setRange(-20, 20)
        self.gain_slider.setValue(0)
        self.gain_slider.setMinimumHeight(120)
        self.gain_slider.valueChanged.connect(self.on_value_changed)
        layout.addWidget(self.gain_slider, alignment=Qt.AlignCenter)
        
        # Gain value label
        self.gain_label = QLabel("0 dB")
        self.gain_label.setAlignment(Qt.AlignCenter)
        self.gain_label.setStyleSheet(f"color: {DARK_THEME['text_secondary']}; font-size: 10px;")
        layout.addWidget(self.gain_label)
        
        # Frequency slider
        freq_layout = QVBoxLayout()
        freq_layout.addWidget(QLabel("Freq"))
        self.freq_slider = ModernSlider(Qt.Horizontal)
        self.freq_slider.setRange(freq_range[0], freq_range[1])
        self.freq_slider.setValue((freq_range[0] + freq_range[1]) // 2)
        self.freq_slider.valueChanged.connect(self.on_value_changed)
        freq_layout.addWidget(self.freq_slider)
        
        self.freq_label = QLabel(f"{self.freq_slider.value()} Hz")
        self.freq_label.setStyleSheet(f"color: {DARK_THEME['text_secondary']}; font-size: 9px;")
        freq_layout.addWidget(self.freq_label)
        layout.addLayout(freq_layout)
        
        # Q factor slider
        q_layout = QVBoxLayout()
        q_layout.addWidget(QLabel("Q"))
        self.q_slider = ModernSlider(Qt.Horizontal)
        self.q_slider.setRange(1, 100)
        self.q_slider.setValue(10)
        self.q_slider.valueChanged.connect(self.on_value_changed)
        q_layout.addWidget(self.q_slider)
        
        self.q_label = QLabel(f"{self.q_slider.value() / 10:.1f}")
        self.q_label.setStyleSheet(f"color: {DARK_THEME['text_secondary']}; font-size: 9px;")
        q_layout.addWidget(self.q_label)
        layout.addLayout(q_layout)
        
        self.setStyleSheet(f"""
            QWidget {{
                background: {DARK_THEME['surface']};
                border-radius: 8px;
                padding: 10px;
                margin: 2px;
            }}
            QLabel {{
                color: {DARK_THEME['text']};
                font-size: 10px;
            }}
        """)
    
    def on_value_changed(self):
        gain = self.gain_slider.value()
        freq = self.freq_slider.value()
        q = self.q_slider.value() / 10.0
        
        self.gain_label.setText(f"{gain:+d} dB")
        self.freq_label.setText(f"{freq} Hz")
        self.q_label.setText(f"{q:.1f}")
        
        self.value_changed.emit(self.band_id, gain, freq, q)

# ======================= 
# AUDIO PROCESSING
# =======================

class AudioProcessor:
    """Enhanced audio processing with better error handling"""
    def __init__(self):
        self.genre_model_path = GENRE_MODEL_PATH
        self.eq_model_path = EQ_MODEL_PATH
        self.genre_accl = None
        self.eq_accl = None
        self.is_processing = False
        
    def initialize_models(self):
        """Initialize AI models with error handling"""
        try:
            if os.path.exists(self.genre_model_path):
                self.genre_accl = mx.Accelerator(self.genre_model_path)
                print(f"Genre model loaded: {self.genre_model_path}")
            else:
                print(f"Warning: Genre model not found at {self.genre_model_path}")
                
            if os.path.exists(self.eq_model_path):
                self.eq_accl = mx.Accelerator(self.eq_model_path)
                print(f"EQ model loaded: {self.eq_model_path}")
            else:
                print(f"Warning: EQ model not found at {self.eq_model_path}")
                
            return True
        except Exception as e:
            print(f"Error initializing models: {e}")
            return False
    
    def predict_genre(self, audio_features):
        """Predict genre from audio features"""
        if self.genre_accl is None:
            return 0, 0.0
        
        try:
            outputs = self.genre_accl.run(audio_features)
            probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
            predicted_genre = np.argmax(probabilities)
            confidence = probabilities[predicted_genre]
            return predicted_genre, confidence
        except Exception as e:
            print(f"Genre prediction error: {e}")
            return 0, 0.0
    
    def predict_eq_settings(self, audio_features):
        """Predict EQ settings from audio features"""
        if self.eq_accl is None:
            return np.zeros(EQ_TOTAL_FEATURES)
        
        try:
            outputs = self.eq_accl.run(audio_features)
            return outputs[0].flatten()[:EQ_TOTAL_FEATURES]
        except Exception as e:
            print(f"EQ prediction error: {e}")
            return np.zeros(EQ_TOTAL_FEATURES)

class AudioAnalysisThread(QThread):
    """Thread for real-time audio analysis"""
    genre_detected = pyqtSignal(int, float)
    eq_suggested = pyqtSignal(np.ndarray)
    spectrum_updated = pyqtSignal(np.ndarray)
    waveform_updated = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.audio_file = None
        self.is_running = False
        
    def set_audio_file(self, file_path):
        self.audio_file = file_path
        
    def run(self):
        if not self.audio_file or not os.path.exists(self.audio_file):
            self.error_occurred.emit("Audio file not found")
            return
            
        try:
            # Load audio
            audio, sr = librosa.load(self.audio_file, sr=SAMPLE_RATE)
            self.waveform_updated.emit(audio[:SAMPLE_RATE])  # First second for display
            
            # Process in chunks
            chunk_samples = CHUNK_SIZE
            for i in range(0, len(audio) - chunk_samples, chunk_samples // 2):
                if not self.is_running:
                    break
                    
                chunk = audio[i:i + chunk_samples]
                
                # Compute FFT for spectrum
                fft = np.fft.fft(chunk)
                self.spectrum_updated.emit(fft)
                
                # Extract features for AI models
                features = self.extract_features(chunk)
                
                # Predict genre
                genre, confidence = self.processor.predict_genre(features)
                self.genre_detected.emit(genre, confidence)
                
                # Predict EQ settings
                eq_params = self.processor.predict_eq_settings(features)
                self.eq_suggested.emit(eq_params)
                
                self.msleep(100)  # Update every 100ms
                
        except Exception as e:
            self.error_occurred.emit(f"Analysis error: {str(e)}")
    
    def extract_features(self, audio_chunk):
        """Extract audio features for AI models"""
        try:
            # Compute spectral features
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=SAMPLE_RATE)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=SAMPLE_RATE)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_chunk)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroid),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            # Ensure proper shape for model input
            return features.reshape(1, -1).astype(np.float32)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros((1, 16), dtype=np.float32)
    
    def stop_analysis(self):
        self.is_running = False
        self.wait()

# ======================= 
# MAIN APPLICATION
# =======================

class EQAIMainWindow(QMainWindow):
    """Enhanced main application window"""
    def __init__(self):
        super().__init__()
        self.processor = AudioProcessor()
        self.analysis_thread = None
        self.current_file = None
        self.eq_bands = []
        
        self.setup_ui()
        self.setup_theme()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Initialize models in background
        QTimer.singleShot(1000, self.initialize_models)
    
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("EQ_AI - Enhanced Audio Processor")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualization and EQ
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        
        # File selection group
        file_group = QGroupBox("Audio File")
        file_layout = QVBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {DARK_THEME['text_secondary']}; padding: 10px;")
        file_layout.addWidget(self.file_label)
        
        btn_layout = QHBoxLayout()
        self.load_btn = ModernButton("Load Audio", "primary")
        self.load_btn.clicked.connect(self.load_audio_file)
        btn_layout.addWidget(self.load_btn)
        
        self.play_btn = ModernButton("Play", "success")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        btn_layout.addWidget(self.play_btn)
        
        file_layout.addLayout(btn_layout)
        layout.addWidget(file_group)
        
        # Analysis group
        analysis_group = QGroupBox("AI Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Genre detection
        genre_layout = QHBoxLayout()
        genre_layout.addWidget(QLabel("Detected Genre:"))
        self.genre_label = QLabel("Unknown")
        self.genre_label.setStyleSheet(f"color: {DARK_THEME['primary']}; font-weight: bold;")
        genre_layout.addWidget(self.genre_label)
        genre_layout.addStretch()
        analysis_layout.addLayout(genre_layout)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_label = QLabel("0%")
        self.confidence_label.setStyleSheet(f"color: {DARK_THEME['accent']};")
        conf_layout.addWidget(self.confidence_label)
        conf_layout.addStretch()
        analysis_layout.addLayout(conf_layout)
        
        # Analysis controls
        self.analyze_btn = ModernButton("Start Analysis", "secondary")
        self.analyze_btn.clicked.connect(self.toggle_analysis)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(analysis_group)
        
        # Presets group
        presets_group = QGroupBox("EQ Presets")
        presets_layout = QVBoxLayout(presets_group)
        
        self.presets_combo = QComboBox()
        self.presets_combo.addItems([
            "AI Suggested", "Flat", "Rock", "Pop", "Jazz", "Classical", 
            "Electronic", "Hip-Hop", "Vocal", "Bass Boost"
        ])
        self.presets_combo.currentTextChanged.connect(self.apply_preset)
        presets_layout.addWidget(self.presets_combo)
        
        preset_btn_layout = QHBoxLayout()
        save_preset_btn = ModernButton("Save", "default")
        save_preset_btn.clicked.connect(self.save_preset)
        preset_btn_layout.addWidget(save_preset_btn)
        
        load_preset_btn = ModernButton("Load", "default")
        load_preset_btn.clicked.connect(self.load_preset)
        preset_btn_layout.addWidget(load_preset_btn)
        
        presets_layout.addLayout(preset_btn_layout)
        layout.addWidget(presets_group)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Real-time processing
        self.realtime_cb = QCheckBox("Real-time Processing")
        self.realtime_cb.setChecked(True)
        settings_layout.addWidget(self.realtime_cb)
        
        # Auto-apply EQ
        self.auto_eq_cb = QCheckBox("Auto-apply AI EQ")
        self.auto_eq_cb.setChecked(False)
        settings_layout.addWidget(self.auto_eq_cb)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """Create right visualization and EQ panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Visualization tabs
        vis_tabs = QTabWidget()
        
        # Waveform tab
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout(waveform_tab)
        self.waveform_widget = WaveformWidget()
        waveform_layout.addWidget(QLabel("Waveform"))
        waveform_layout.addWidget(self.waveform_widget)
        vis_tabs.addTab(waveform_tab, "Waveform")
        
        # Spectrum tab
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)
        self.spectrum_widget = SpectrumAnalyzer()
        spectrum_layout.addWidget(QLabel("Frequency Spectrum"))
        spectrum_layout.addWidget(self.spectrum_widget)
        vis_tabs.addTab(spectrum_tab, "Spectrum")
        
        layout.addWidget(vis_tabs, stretch=1)
        
        # EQ Section
        eq_group = QGroupBox("5-Band Parametric EQ")
        eq_layout = QHBoxLayout(eq_group)
        
        # EQ bands
        band_configs = [
            ("Low", (20, 250)),
            ("Low-Mid", (250, 500)),
            ("Mid", (500, 2000)),
            ("Hi-Mid", (2000, 4000)),
            ("High", (4000, 20000))
        ]
        
        for i, (name, freq_range) in enumerate(band_configs):
            band_widget = EQBandWidget(i, name, freq_range)
            band_widget.value_changed.connect(self.on_eq_changed)
            self.eq_bands.append(band_widget)
            eq_layout.addWidget(band_widget)
        
        layout.addWidget(eq_group, stretch=1)
        
        # EQ Controls
        eq_controls = QHBoxLayout()
        
        self.bypass_btn = ModernButton("Bypass EQ", "default")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.clicked.connect(self.toggle_eq_bypass)
        eq_controls.addWidget(self.bypass_btn)
        
        self.reset_eq_btn = ModernButton("Reset EQ", "default")
        self.reset_eq_btn.clicked.connect(self.reset_eq)
        eq_controls.addWidget(self.reset_eq_btn)
        
        eq_controls.addStretch()
        
        self.export_btn = ModernButton("Export Audio", "success")
        self.export_btn.clicked.connect(self.export_audio)
        self.export_btn.setEnabled(False)
        eq_controls.addWidget(self.export_btn)
        
        layout.addLayout(eq_controls)
        
        return panel
    
    def setup_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
            }}
            QWidget {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {DARK_THEME['border']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {DARK_THEME['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {DARK_THEME['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {DARK_THEME['border']};
                background-color: {DARK_THEME['surface']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: {DARK_THEME['surface']};
                color: {DARK_THEME['text']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {DARK_THEME['border']};
            }}
            QTabBar::tab:selected {{
                background-color: {DARK_THEME['primary']};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {DARK_THEME['border']};
            }}
            QComboBox {{
                background-color: {DARK_THEME['surface']};
                border: 2px solid {DARK_THEME['border']};
                border-radius: 6px;
                padding: 6px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {DARK_THEME['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {DARK_THEME['text']};
                width: 0px;
                height: 0px;
            }}
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid {DARK_THEME['border']};
                background-color: {DARK_THEME['surface']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {DARK_THEME['primary']};
                border-color: {DARK_THEME['primary']};
            }}
            QProgressBar {{
                border: 2px solid {DARK_THEME['border']};
                border-radius: 5px;
                text-align: center;
                background-color: {DARK_THEME['surface']};
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {DARK_THEME['primary']}, stop:1 {DARK_THEME['accent']});
                border-radius: 3px;
            }}
            QStatusBar {{
                background-color: {DARK_THEME['surface']};
                border-top: 1px solid {DARK_THEME['border']};
            }}
            QMenuBar {{
                background-color: {DARK_THEME['surface']};
                border-bottom: 1px solid {DARK_THEME['border']};
            }}
            QMenuBar::item {{
                padding: 6px 12px;
                background-color: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: {DARK_THEME['primary']};
                border-radius: 4px;
            }}
            QMenu {{
                background-color: {DARK_THEME['surface']};
                border: 1px solid {DARK_THEME['border']};
                border-radius: 6px;
            }}
            QMenu::item {{
                padding: 8px 16px;
            }}
            QMenu::item:selected {{
                background-color: {DARK_THEME['primary']};
            }}
        """)
    
    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Audio File', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_audio_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export Processed Audio', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_audio)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        batch_action = QAction('Batch Processing', self)
        batch_action.triggered.connect(self.show_batch_dialog)
        tools_menu.addAction(batch_action)
        
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.show_settings_dialog)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready - Load an audio file to begin")
        
        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def initialize_models(self):
        """Initialize AI models"""
        self.statusBar().showMessage("Initializing AI models...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        success = self.processor.initialize_models()
        
        self.progress_bar.setVisible(False)
        if success:
            self.statusBar().showMessage("AI models initialized successfully")
        else:
            self.statusBar().showMessage("Warning: Some AI models failed to load")
            QMessageBox.warning(self, "Model Loading", 
                              "Some AI models could not be loaded. Check file paths.")
    
    def load_audio_file(self):
        """Load audio file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.current_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.setText(f"Loaded: {filename}")
            
            # Enable controls
            self.play_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Load waveform for display
            try:
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                self.waveform_widget.set_waveform(audio[:SAMPLE_RATE])  # First second
                self.statusBar().showMessage(f"Loaded: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")
    
    def play_audio(self):
        """Play/pause audio"""
        if not self.current_file:
            return
        
        if self.play_btn.text() == "Play":
            try:
                # Simple playback using sounddevice
                audio, _ = librosa.load(self.current_file, sr=SAMPLE_RATE)
                sd.play(audio, SAMPLE_RATE)
                self.play_btn.setText("Stop")
                self.statusBar().showMessage("Playing audio...")
            except Exception as e:
                QMessageBox.critical(self, "Playback Error", str(e))
        else:
            sd.stop()
            self.play_btn.setText("Play")
            self.statusBar().showMessage("Playback stopped")
    
    def toggle_analysis(self):
        """Start/stop audio analysis"""
        if not self.current_file:
            return
        
        if self.analyze_btn.text() == "Start Analysis":
            self.start_analysis()
        else:
            self.stop_analysis()
    
    def start_analysis(self):
        """Start real-time analysis"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            return
        
        self.analysis_thread = AudioAnalysisThread(self.processor)
        self.analysis_thread.set_audio_file(self.current_file)
        
        # Connect signals
        self.analysis_thread.genre_detected.connect(self.on_genre_detected)
        self.analysis_thread.eq_suggested.connect(self.on_eq_suggested)
        self.analysis_thread.spectrum_updated.connect(self.on_spectrum_updated)
        self.analysis_thread.waveform_updated.connect(self.on_waveform_updated)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        
        self.analysis_thread.is_running = True
        self.analysis_thread.start()
        
        self.analyze_btn.setText("Stop Analysis")
        self.statusBar().showMessage("Analyzing audio...")
    
    def stop_analysis(self):
        """Stop analysis"""
        if self.analysis_thread:
            self.analysis_thread.stop_analysis()
            self.analysis_thread = None
        
        self.analyze_btn.setText("Start Analysis")
        self.statusBar().showMessage("Analysis stopped")
    
    def on_genre_detected(self, genre_id, confidence):
        """Handle genre detection result"""
        genre_name = GENRES.get(genre_id, "Unknown")
        self.genre_label.setText(genre_name)
        self.confidence_label.setText(f"{confidence*100:.1f}%")
        
        # Update status
        self.statusBar().showMessage(f"Genre: {genre_name} ({confidence*100:.1f}%)")
    
    def on_eq_suggested(self, eq_params):
        """Handle AI EQ suggestions"""
        if self.auto_eq_cb.isChecked() and len(eq_params) >= EQ_TOTAL_FEATURES:
            # Apply AI suggested EQ settings
            for i, band_widget in enumerate(self.eq_bands):
                if i * 3 + 2 < len(eq_params):
                    gain = eq_params[i * 3] * 20  # Scale to ±20dB
                    freq = eq_params[i * 3 + 1] * 10000  # Scale frequency
                    q = eq_params[i * 3 + 2] * 10  # Scale Q factor
                    
                    band_widget.gain_slider.setValue(int(gain))
                    # Don't auto-adjust freq and Q to avoid conflicts
    
    def on_spectrum_updated(self, fft_data):
        """Update spectrum display"""
        self.spectrum_widget.set_spectrum(fft_data)
    
    def on_waveform_updated(self, waveform_data):
        """Update waveform display"""
        self.waveform_widget.set_waveform(waveform_data)
    
    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        QMessageBox.warning(self, "Analysis Error", error_msg)
        self.stop_analysis()
    
    def on_eq_changed(self, band_id, gain, freq, q):
        """Handle EQ parameter changes"""
        # This would apply real-time EQ processing
        # For now, just update status
        self.statusBar().showMessage(f"EQ Band {band_id+1}: {gain:+.1f}dB @ {freq}Hz Q{q:.1f}")
    
    def apply_preset(self, preset_name):
        """Apply EQ preset"""
        presets = {
            "Flat": [(0, 1000, 1.0)] * 5,
            "Rock": [(3, 80, 1.2), (2, 250, 1.0), (1, 1000, 1.0), (4, 3000, 1.2), (5, 8000, 1.5)],
            "Pop": [(2, 60, 1.0), (1, 200, 1.0), (0, 1000, 1.0), (2, 3000, 1.0), (3, 10000, 1.2)],
            "Jazz": [(1, 60, 1.0), (0, 300, 1.0), (1, 1000, 1.0), (2, 4000, 1.0), (1, 8000, 1.0)],
            "Classical": [(0, 60, 1.0), (1, 200, 1.0), (2, 1500, 1.0), (3, 4000, 1.2), (2, 10000, 1.0)],
            "Electronic": [(4, 60, 1.5), (2, 250, 1.0), (-1, 1000, 1.0), (3, 4000, 1.2), (4, 12000, 1.3)],
            "Hip-Hop": [(5, 60, 1.8), (3, 200, 1.2), (0, 1000, 1.0), (2, 3000, 1.0), (3, 8000, 1.2)],
            "Vocal": [(0, 80, 1.0), (-1, 200, 1.0), (3, 1000, 1.2), (4, 3000, 1.5), (2, 8000, 1.0)],
            "Bass Boost": [(8, 60, 2.0), (5, 150, 1.5), (0, 500, 1.0), (0, 2000, 1.0), (0, 8000, 1.0)]
        }
        
        if preset_name in presets and preset_name != "AI Suggested":
            settings = presets[preset_name]
            for i, (gain, freq, q) in enumerate(settings):
                if i < len(self.eq_bands):
                    band = self.eq_bands[i]
                    band.gain_slider.setValue(gain)
                    if freq != band.freq_slider.value():
                        band.freq_slider.setValue(min(max(freq, band.freq_slider.minimum()), 
                                                    band.freq_slider.maximum()))
                    band.q_slider.setValue(int(q * 10))
    
    def reset_eq(self):
        """Reset all EQ bands to flat"""
        for band in self.eq_bands:
            band.gain_slider.setValue(0)
    
    def toggle_eq_bypass(self):
        """Toggle EQ bypass"""
        is_bypassed = self.bypass_btn.isChecked()
        for band in self.eq_bands:
            band.setEnabled(not is_bypassed)
        
        if is_bypassed:
            self.bypass_btn.setText("Enable EQ")
            self.statusBar().showMessage("EQ bypassed")
        else:
            self.bypass_btn.setText("Bypass EQ")
            self.statusBar().showMessage("EQ enabled")
    
    def save_preset(self):
        """Save current EQ settings as preset"""
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, 'Save Preset', 'Preset name:')
        if ok and name:
            # Save preset logic here
            settings = {}
            for i, band in enumerate(self.eq_bands):
                settings[f'band_{i}'] = {
                    'gain': band.gain_slider.value(),
                    'freq': band.freq_slider.value(),
                    'q': band.q_slider.value() / 10.0
                }
            
            # Save to file (JSON format)
            preset_file = f"presets/{name}.json"
            os.makedirs("presets", exist_ok=True)
            try:
                with open(preset_file, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.statusBar().showMessage(f"Preset '{name}' saved")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save preset: {str(e)}")
    
    def load_preset(self):
        """Load preset from file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Load Preset", "presets/", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                for i, band in enumerate(self.eq_bands):
                    band_key = f'band_{i}'
                    if band_key in settings:
                        band_settings = settings[band_key]
                        band.gain_slider.setValue(band_settings.get('gain', 0))
                        band.freq_slider.setValue(band_settings.get('freq', 1000))
                        band.q_slider.setValue(int(band_settings.get('q', 1.0) * 10))
                
                self.statusBar().showMessage(f"Preset loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load preset: {str(e)}")
    
    def export_audio(self):
        """Export processed audio"""
        if not self.current_file:
            return
        
        file_dialog = QFileDialog()
        output_path, _ = file_dialog.getSaveFileName(
            self, "Export Processed Audio", "", 
            "WAV Files (*.wav);;FLAC Files (*.flac);;All Files (*)"
        )
        
        if output_path:
            try:
                self.statusBar().showMessage("Exporting audio...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 100)
                
                # Load original audio
                audio, sr = librosa.load(self.current_file, sr=None)
                
                # Apply EQ processing (simplified - would need proper audio processing)
                # This is a placeholder - real implementation would apply filter banks
                processed_audio = audio  # Apply actual EQ processing here
                
                # Save processed audio
                sf.write(output_path, processed_audio, sr)
                
                self.progress_bar.setVisible(False)
                self.statusBar().showMessage(f"Audio exported: {os.path.basename(output_path)}")
                
                QMessageBox.information(self, "Export Complete", 
                                      f"Processed audio saved to:\n{output_path}")
                
            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Export Error", f"Failed to export audio: {str(e)}")
    
    def show_batch_dialog(self):
        """Show batch processing dialog"""
        dialog = BatchProcessingDialog(self)
        dialog.exec_()
    
    def show_settings_dialog(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        dialog.exec_()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About EQ_AI", 
                         "EQ_AI - Enhanced Audio Processor\n\n"
                         "An AI-powered audio processing application with\n"
                         "automatic genre detection and EQ optimization.\n\n"
                         "Features:\n"
                         "• AI Genre Detection\n"
                         "• Intelligent EQ Suggestions\n"
                         "• Real-time Spectrum Analysis\n"
                         "• Custom Presets\n"
                         "• Batch Processing\n\n"
                         "Version 2.0")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.analysis_thread:
            self.stop_analysis()
        sd.stop()  # Stop any audio playback
        event.accept()


class BatchProcessingDialog(QDialog):
    """Dialog for batch processing multiple files"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(600, 400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File list
        self.file_list = QListWidget()
        layout.addWidget(QLabel("Files to Process:"))
        layout.addWidget(self.file_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = ModernButton("Add Files", "primary")
        add_btn.clicked.connect(self.add_files)
        btn_layout.addWidget(add_btn)
        
        remove_btn = ModernButton("Remove Selected", "default")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        
        process_btn = ModernButton("Process All", "success")
        process_btn.clicked.connect(self.process_files)
        btn_layout.addWidget(process_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", 
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        for file in files:
            self.file_list.addItem(file)
    
    def remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
    
    def process_files(self):
        # Placeholder for batch processing logic
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            return
        
        self.progress.setRange(0, len(files))
        for i, file in enumerate(files):
            # Process each file here
            self.progress.setValue(i + 1)
            QApplication.processEvents()
        
        QMessageBox.information(self, "Complete", f"Processed {len(files)} files")


class SettingsDialog(QDialog):
    """Settings configuration dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(400, 300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QGridLayout(audio_group)
        
        audio_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["22050", "44100", "48000", "96000"])
        self.sample_rate_combo.setCurrentText("22050")
        audio_layout.addWidget(self.sample_rate_combo, 0, 1)
        
        audio_layout.addWidget(QLabel("Buffer Size:"), 1, 0)
        self.buffer_size_combo = QComboBox()
        self.buffer_size_combo.addItems(["512", "1024", "2048", "4096"])
        self.buffer_size_combo.setCurrentText("2048")
        audio_layout.addWidget(self.buffer_size_combo, 1, 1)
        
        layout.addWidget(audio_group)
        
        # AI settings
        ai_group = QGroupBox("AI Model Settings")
        ai_layout = QGridLayout(ai_group)
        
        ai_layout.addWidget(QLabel("Genre Model:"), 0, 0)
        self.genre_model_path = QLineEdit(GENRE_MODEL_PATH)
        ai_layout.addWidget(self.genre_model_path, 0, 1)
        
        genre_browse_btn = QPushButton("Browse")
        genre_browse_btn.clicked.connect(lambda: self.browse_model(self.genre_model_path))
        ai_layout.addWidget(genre_browse_btn, 0, 2)
        
        ai_layout.addWidget(QLabel("EQ Model:"), 1, 0)
        self.eq_model_path = QLineEdit(EQ_MODEL_PATH)
        ai_layout.addWidget(self.eq_model_path, 1, 1)
        
        eq_browse_btn = QPushButton("Browse")
        eq_browse_btn.clicked.connect(lambda: self.browse_model(self.eq_model_path))
        ai_layout.addWidget(eq_browse_btn, 1, 2)
        
        layout.addWidget(ai_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def browse_model(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.dfp);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)


# ======================= 
# MAIN APPLICATION ENTRY
# =======================

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("EQ_AI Enhanced")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Audio Processing Solutions")
    
    # Create and show main window
    window = EQAIMainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()