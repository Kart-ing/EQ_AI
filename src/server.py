#!/usr/bin/env python3
"""
FastAPI Audio Processing Server with WebSocket support
"""

import os
import sys
import tempfile
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    import memryx as mx
    HAS_AUDIO = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_AUDIO = False

# Genre mapping - 8 genres from your model
GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock'
}

# Enhanced EQ presets for blended genres
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

# Pydantic models
class GenreResponse(BaseModel):
    genre: str
    confidence: float
    all_predictions: Dict[str, float]

class EQResponse(BaseModel):
    eq_adjustments: List[float]
    genre: Optional[str] = None

class ProcessedAudioResponse(BaseModel):
    file_id: str
    genre: str
    confidence: float
    eq_adjustments: List[float]
    original_filename: str
    processed_filename: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    audio_support: bool

class LiveAudioRequest(BaseModel):
    audio_data: List[float]
    sample_rate: int = 22050

# Global variables
app = FastAPI(title="AI Audio Mastering API", version="1.0.0")
genre_model_path = None
eq_model_path = None
temp_dir = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def startup_check():
    """Check if everything is properly configured"""
    if not HAS_AUDIO:
        raise RuntimeError("Audio processing libraries not available")
    
    if not genre_model_path or not eq_model_path:
        raise RuntimeError("Model paths not configured")
    
    if not os.path.exists(genre_model_path) or not os.path.exists(eq_model_path):
        raise RuntimeError("Model files not found")

class AudioProcessor:
    """Audio processing functionality"""
    
    @staticmethod
    def extract_genre_features(audio_data, sample_rate=22050):
        """Extract 60 features for genre classification"""
        try:
            features = []
            
            # Ensure audio is not too short
            if len(audio_data) < 2048:
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)), 'constant')
            
            # Spectral features (8 features)
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
            
            # MFCC features (26 features: 13 mfccs × 2 stats)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            for mfcc in mfccs:
                features.extend([np.mean(mfcc), np.std(mfcc)])
            
            # Chroma features (2 features)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            # Tempo (1 feature)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                tempo_val = tempo if np.isscalar(tempo) else tempo[0] if len(tempo) > 0 else 120.0
            except:
                tempo_val = 120.0
            features.append(tempo_val)
            
            # RMS energy (2 features)
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # Spectral contrast (14 features: 7 bands × 2 stats)
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
    
    @staticmethod
    def extract_eq_features(audio_data, sample_rate=22050):
        """Extract 12 features for EQ model"""
        try:
            features = []
            
            # Spectral features (4)
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
            
            # MFCC features (4)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=4)
            for mfcc in mfccs:
                features.append(float(np.mean(mfcc)))
            
            # Chroma and tempo (2)
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
            
            # RMS energy (2)
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([
                float(np.mean(rms)),
                float(np.std(rms))
            ])
            
            # Ensure exactly 12 features
            feature_vector = np.array(features[:12], dtype=np.float32)
            if len(feature_vector) < 12:
                feature_vector = np.pad(feature_vector, (0, 12 - len(feature_vector)), 'constant')
            
            # Normalize
            if np.std(feature_vector) > 0:
                feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            print(f"EQ feature extraction failed: {str(e)}")
            return np.zeros(12, dtype=np.float32)
    @staticmethod
    def predict_genre(features):
        """Predict genre with aggressive electronic filtering"""
        try:
            # Reshape features to match model input [1, 60]
            model_input = features.reshape(1, -1).astype(np.float32)
            
            # Check input shape
            if model_input.shape != (1, 60):
                print(f"Warning: Input shape {model_input.shape} doesn't match expected (1, 60)")
                if model_input.shape[1] > 60:
                    model_input = model_input[:, :60]
                elif model_input.shape[1] < 60:
                    model_input = np.pad(model_input, ((0, 0), (0, 60 - model_input.shape[1])), 'constant')
            
            genre_accl = mx.MultiStreamAsyncAccl(dfp=genre_model_path)
            
            prediction_result = None
            inference_complete = False
            
            def input_callback(stream_idx):
                return model_input
            
            def output_callback(stream_idx, *mxa_output):
                nonlocal prediction_result, inference_complete
                
                if len(mxa_output) > 0:
                    output = mxa_output[0]
                    
                    # Handle different output shapes
                    if len(output.shape) == 1:
                        predictions = output
                    else:
                        predictions = output[0]
                    
                    # Apply softmax to get probabilities
                    exp_output = np.exp(predictions - np.max(predictions))
                    probabilities = exp_output / np.sum(exp_output)
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    top_probs = probabilities[top_indices]
                    top_genres = [GENRES.get(idx, "Unknown") for idx in top_indices]
                    
                    print(f"Raw predictions: {list(zip(top_genres, top_probs))}")  # Debug output
                    
                    electronic_idx = 0  # Electronic is index 0
                    electronic_prob = probabilities[electronic_idx]
                    
                    # AGGRESSIVE ELECTRONIC FILTERING
                    # Only allow Electronic if it's clearly dominant (>50% and significantly higher than others)
                    if (electronic_prob > 0.5 and 
                        len(top_indices) > 1 and 
                        electronic_prob - top_probs[1] > 0.15):
                        final_genre = "Electronic"
                        confidence = electronic_prob
                    
                    # If Electronic is top but doesn't meet strict criteria, use second genre
                    elif top_indices[0] == electronic_idx:
                        if len(top_indices) > 1:
                            # Use second genre if available
                            final_genre = GENRES.get(top_indices[1], "Unknown")
                            confidence = top_probs[1]
                            
                            # Special case: if second is also Electronic-like, try third
                            if final_genre in ["Electronic", "Instrumental"] and len(top_indices) > 2:
                                final_genre = GENRES.get(top_indices[2], "Unknown")
                                confidence = top_probs[2]
                        else:
                            final_genre = "Unknown"
                            confidence = 0.3
                    
                    # Default case: use top non-electronic genre
                    else:
                        # Find the first non-electronic genre in top predictions
                        for i, genre_idx in enumerate(top_indices):
                            genre_name = GENRES.get(genre_idx, "Unknown")
                            if genre_name != "Electronic":
                                final_genre = genre_name
                                confidence = top_probs[i]
                                break
                        else:
                            # If all are electronic (shouldn't happen), use first
                            final_genre = GENRES.get(top_indices[0], "Unknown")
                            confidence = top_probs[0]
                    
                    # Ensure we don't show Electronic with low confidence
                    if final_genre == "Electronic" and confidence < 0.4:
                        # Force switch to second genre
                        if len(top_indices) > 1:
                            final_genre = GENRES.get(top_indices[1], "Analyzing")
                            confidence = top_probs[1]
                        else:
                            final_genre = "Analyzing"
                            confidence = 0.3
                    
                    # Create simplified predictions (no percentages)
                    all_preds = {}
                    for i in top_indices[:3]:
                        genre_display = GENRES.get(i, f"Unknown ({i})")
                        if genre_display != "Electronic" or probabilities[i] > 0.6:
                            all_preds[genre_display] = "Detected"
                    
                    # If we filtered out all predictions, show "Analyzing"
                    if not all_preds:
                        all_preds = {"Analyzing": "Processing"}
                    
                    prediction_result = (final_genre, confidence, all_preds)
                    inference_complete = True
            
            genre_accl.connect_streams(input_callback, output_callback, 1)
            
            # Wait for inference with timeout
            start_time = time.time()
            while not inference_complete and (time.time() - start_time) < 5.0:
                time.sleep(0.01)
            
            genre_accl.stop()
            
            if prediction_result:
                return prediction_result
            else:
                print("Genre prediction timed out, using fallback")
                return "Analyzing", 0.3, {"Analyzing": "Processing"}
            
        except Exception as e:
            print(f"Genre prediction error: {e}")
            return "Unknown", 0.3, {"Unknown": "Processing"}
        
    @staticmethod
    def predict_eq(features):
        """Predict EQ adjustments using MemryX model"""
        try:
            eq_accl = mx.MultiStreamAsyncAccl(dfp=eq_model_path)
            
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
    
    @staticmethod
    def apply_simple_eq(audio_data, eq_adjustments, sample_rate=22050):
        """Apply simple EQ adjustments to audio"""
        if np.all(np.array(eq_adjustments) == 0):
            return audio_data
        
        processed_audio = audio_data.copy()
        overall_gain = 1.0 + 0.05 * np.mean(eq_adjustments)
        processed_audio = processed_audio * overall_gain
        
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0.95:
            processed_audio = processed_audio * (0.95 / max_val)
        
        return processed_audio

def cleanup_temp_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=bool(genre_model_path and eq_model_path),
        audio_support=HAS_AUDIO
    )

@app.get("/genres")
async def get_available_genres():
    """Get list of available genres"""
    return {"genres": list(GENRES.values())}

@app.get("/eq-presets")
async def get_eq_presets():
    """Get predefined EQ presets for all genres"""
    return {"presets": GENRE_EQ_PRESETS}

@app.get("/eq-presets/{genre}")
async def get_genre_eq_preset(genre: str):
    """Get EQ preset for specific genre"""
    if genre not in GENRE_EQ_PRESETS:
        raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found")
    
    return {
        "genre": genre,
        "eq_adjustments": GENRE_EQ_PRESETS[genre]
    }

@app.post("/analyze-genre", response_model=GenreResponse)
async def analyze_genre(file: UploadFile = File(...)):
    """Analyze audio file and return genre classification"""
    if not HAS_AUDIO:
        raise HTTPException(status_code=503, detail="Audio processing not available")
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        audio_data, sr = librosa.load(temp_file.name, sr=22050, mono=True)
        features = AudioProcessor.extract_genre_features(audio_data, sr)
        genre_name, confidence, all_predictions = AudioProcessor.predict_genre(features)
        
        os.unlink(temp_file.name)
        
        return GenreResponse(
            genre=genre_name,
            confidence=confidence,
            all_predictions=all_predictions
        )
        
    except Exception as e:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-genre-live")
async def analyze_genre_live(request: LiveAudioRequest):
    """Analyze raw audio data for live processing"""
    if not HAS_AUDIO:
        raise HTTPException(status_code=503, detail="Audio processing not available")
    
    try:
        audio_data = np.array(request.audio_data, dtype=np.float32)
        
        if len(audio_data) == 0:
            return {"genre": "Unknown", "confidence": 0.0, "all_predictions": {"Unknown": 1.0}}
        
        features = AudioProcessor.extract_genre_features(audio_data, request.sample_rate)
        genre_name, confidence, all_predictions = AudioProcessor.predict_genre(features)
        
        return {
            "genre": genre_name,
            "confidence": confidence,
            "all_predictions": all_predictions
        }
        
    except Exception as e:
        print(f"Live genre analysis error: {e}")
        return {"genre": "Unknown", "confidence": 0.0, "all_predictions": {"Unknown": 1.0}}

@app.post("/predict-eq", response_model=EQResponse)
async def predict_eq(file: UploadFile = File(...)):
    """Predict EQ adjustments for audio file"""
    if not HAS_AUDIO:
        raise HTTPException(status_code=503, detail="Audio processing not available")
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        audio_data, sr = librosa.load(temp_file.name, sr=22050, mono=True)
        eq_features = AudioProcessor.extract_eq_features(audio_data, sr)
        eq_adjustments = AudioProcessor.predict_eq(eq_features)
        
        os.unlink(temp_file.name)
        
        return EQResponse(eq_adjustments=eq_adjustments)
        
    except Exception as e:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-audio", response_model=ProcessedAudioResponse)
async def process_audio_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Complete audio processing: analyze genre, predict EQ, and apply processing"""
    if not HAS_AUDIO:
        raise HTTPException(status_code=503, detail="Audio processing not available")
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        audio_data, sr = librosa.load(temp_file.name, sr=22050, mono=True)
        
        genre_features = AudioProcessor.extract_genre_features(audio_data, sr)
        genre_name, confidence, _ = AudioProcessor.predict_genre(genre_features)
        
        eq_features = AudioProcessor.extract_eq_features(audio_data, sr)
        eq_adjustments = AudioProcessor.predict_eq(eq_features)
        
        processed_audio = AudioProcessor.apply_simple_eq(audio_data, eq_adjustments, sr)
        
        file_id = str(uuid.uuid4())
        processed_filename = f"processed_{file_id}.wav"
        processed_path = os.path.join(temp_dir, processed_filename)
        
        sf.write(processed_path, processed_audio, sr)
        
        background_tasks.add_task(cleanup_temp_file, temp_file.name)
        
        return ProcessedAudioResponse(
            file_id=file_id,
            genre=genre_name,
            confidence=confidence,
            eq_adjustments=eq_adjustments,
            original_filename=file.filename,
            processed_filename=processed_filename
        )
        
    except Exception as e:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    

@staticmethod
def analyze_spectral_content(audio_data, sample_rate=22050):
    """Real-time spectral analysis for EQ adjustments"""
    try:
        # Calculate FFT
        fft = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Define frequency bands (matching your 5-band EQ)
        bands = {
            'sub_bass': (20, 60),      # 20-60Hz
            'bass': (60, 250),         # 60-250Hz  
            'low_mid': (250, 1000),    # 250-1000Hz
            'mid': (1000, 4000),       # 1-4kHz
            'high_mid': (4000, 8000),  # 4-8kHz
            'high': (8000, 12000)      # 8-12kHz
        }
        
        band_energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequencies in this band
            band_mask = (np.abs(frequencies) >= low_freq) & (np.abs(frequencies) <= high_freq)
            if np.any(band_mask):
                band_energy = np.mean(magnitudes[band_mask])
                band_energies[band_name] = float(band_energy)
            else:
                band_energies[band_name] = 0.0
        
        # Normalize energies
        total_energy = sum(band_energies.values())
        if total_energy > 0:
            band_energies = {k: v/total_energy for k, v in band_energies.items()}
        
        # Calculate EQ adjustments based on spectral balance
        eq_adjustments = []
        
        # Target balanced frequency response (adjust these based on your preferences)
        target_balance = {
            'sub_bass': 0.12,
            'bass': 0.18, 
            'low_mid': 0.20,
            'mid': 0.25,
            'high_mid': 0.15,
            'high': 0.10
        }
        
        for band_name in ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']:
            current_energy = band_energies.get(band_name, 0)
            target_energy = target_balance.get(band_name, 0.15)
            
            # Calculate adjustment (positive = boost, negative = cut)
            if current_energy > 0:
                adjustment_db = 10 * np.log10(target_energy / current_energy)
                # Limit adjustments to reasonable range
                adjustment_db = np.clip(adjustment_db, -6, 6)
            else:
                adjustment_db = 0
            
            eq_adjustments.append(float(adjustment_db))
        
        return {
            'band_energies': band_energies,
            'eq_adjustments': eq_adjustments,
            'overall_energy': float(np.mean(magnitudes))
        }
        
    except Exception as e:
        print(f"Spectral analysis error: {e}")
        return {
            'band_energies': {},
            'eq_adjustments': [0.0] * 6,
            'overall_energy': 0.0
        }

# Add this to your server.py - WebSocket endpoint with proper error handling
@app.websocket("/ws/live-audio")
async def websocket_live_audio(websocket: WebSocket):
    """WebSocket endpoint for live audio processing"""
    await manager.connect(websocket)
    try:
        while True:
            # Try to receive data with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await manager.send_personal_message({
                    "type": "keepalive",
                    "timestamp": time.time()
                }, websocket)
                continue
            
            if 'audio_data' in data and 'sample_rate' in data:
                audio_array = np.array(data['audio_data'], dtype=np.float32)
                
                # Only process if we have meaningful audio data (not just silence/noise)
                if len(audio_array) > 1000 and np.max(np.abs(audio_array)) > 0.01:
                    
                    # Analyze genre
                    features = AudioProcessor.extract_genre_features(audio_array, data['sample_rate'])
                    genre_name, confidence, _ = AudioProcessor.predict_genre(features)
                    
                    # Predict EQ based on spectral analysis
                    eq_features = AudioProcessor.extract_eq_features(audio_array, data['sample_rate'])
                    eq_adjustments = AudioProcessor.predict_eq(eq_features)
                    
                    # Add real-time spectral analysis for EQ
                    spectral_eq = AudioProcessor.analyze_spectral_content(audio_array, data['sample_rate'])
                    
                    # Send results back to client
                    await manager.send_personal_message({
                        "type": "analysis_result",
                        "genre": genre_name,
                        "eq_adjustments": eq_adjustments,
                        "spectral_analysis": spectral_eq,
                        "timestamp": time.time()
                    }, websocket)
                else:
                    # Send silence detection
                    await manager.send_personal_message({
                        "type": "silence_detected",
                        "timestamp": time.time()
                    }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/download/{file_id}")
async def download_processed_file(file_id: str, background_tasks: BackgroundTasks):
    """Download processed audio file"""
    processed_filename = f"processed_{file_id}.wav"
    file_path = os.path.join(temp_dir, processed_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    return FileResponse(
        path=file_path,
        filename=processed_filename,
        media_type='audio/wav'
    )

@app.delete("/cleanup/{file_id}")
async def cleanup_processed_file(file_id: str, background_tasks: BackgroundTasks):
    """Cleanup processed file after use"""
    processed_filename = f"processed_{file_id}.wav"
    file_path = os.path.join(temp_dir, processed_filename)
    
    background_tasks.add_task(cleanup_temp_file, file_path)
    return {"message": "File cleanup scheduled"}

def configure_models(genre_model: str, eq_model: str):
    """Configure model paths"""
    global genre_model_path, eq_model_path, temp_dir
    
    genre_model_path = genre_model
    eq_model_path = eq_model
    temp_dir = tempfile.mkdtemp()
    
    startup_check()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Audio Mastering API Server")
    parser.add_argument("--genre-model", required=True, help="Path to genre classification model (.dfp)")
    parser.add_argument("--eq-model", required=True, help="Path to EQ model (.dfp)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    configure_models(args.genre_model, args.eq_model)
    
    print(f"Starting AI Audio Mastering API on {args.host}:{args.port}")
    print(f"Genre model: {args.genre_model}")
    print(f"EQ model: {args.eq_model}")
    print(f"Supported genres: {list(GENRES.values())}")
    print(f"WebSocket endpoint: ws://{args.host}:{args.port}/ws/live-audio")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )