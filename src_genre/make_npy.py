#!/usr/bin/env python3
"""
Extract spectral features from FMA-Medium dataset - FIXED VERSION
Maps tracks to root genres and saves features/genres to .npy files
"""

import os
import glob
import numpy as np
import librosa
import pandas as pd
import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import gc
import subprocess

# -----------------------------
# CONFIGURATION
# -----------------------------
MAX_FILES = None  # Set to None to process all files

# -----------------------------
# FIXED: Proper root genre mapping for FMA
# -----------------------------
ROOT_GENRES = {
    'Electronic': 0,
    'Experimental': 1, 
    'Folk': 2,
    'Hip-Hop': 3,
    'Instrumental': 4,
    'International': 5,
    'Pop': 6,
    'Rock': 7,
    'Blues': 8,
    'Jazz': 9,
    'Classical': 10,
    'Old-Time / Historic': 11,
    'Country': 12,
    'Easy Listening': 13,
    'Soul-RnB': 14,
    'Spoken': 15
}

# -----------------------------
# IMPROVED Audio Feature Extractor
# -----------------------------
class SpectralFeatureExtractor:
    def __init__(self, sample_rate=22050):  # Reduced for efficiency
        self.sr = sample_rate

    def extract_features(self, audio):
        """Extract comprehensive spectral features optimized for genre classification"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        if len(audio) < 1000:
            return np.zeros(58, dtype=np.float32)

        try:
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Time domain features
            rms = np.sqrt(np.mean(audio**2))
            zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()
            
            # Spectral features
            stft = np.abs(librosa.stft(audio))
            spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sr)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=self.sr)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=self.sr)[0].mean()
            
            # MFCCs - Most important for music genre classification
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            mfcc_mean = mfccs.mean(axis=1)
            mfcc_std = mfccs.std(axis=1)
            
            # Chroma features - Pitch class profiles
            chroma = librosa.feature.chroma_stft(S=stft, sr=self.sr)
            chroma_mean = chroma.mean(axis=1)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=self.sr)
            spectral_contrast_mean = spectral_contrast.mean(axis=1)
            
            # Tonnetz features - Tonal centroid features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sr)
            tonnetz_mean = tonnetz.mean(axis=1)
            
            # Mel-scaled spectrogram (reduced)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_mean = mel_spec_db.mean(axis=1)[:10]  # First 10 bands
            mel_spec_std = mel_spec_db.std(axis=1)[:10]    # First 10 stds
            
            # Statistical features
            audio_mean = np.mean(audio)
            audio_std = np.std(audio)
            audio_skew = np.mean(((audio - audio_mean) / (audio_std + 1e-8)) ** 3)
            audio_kurtosis = np.mean(((audio - audio_mean) / (audio_std + 1e-8)) ** 4)
            
            # Combine all features
            features = []
            
            # MFCCs (13 mean + 13 std = 26 features)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # Chroma features (12)
            features.extend(chroma_mean)
            
            # Spectral features (3)
            features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff])
            
            # Spectral contrast (7)
            features.extend(spectral_contrast_mean)
            
            # Tonnetz (6)
            features.extend(tonnetz_mean)
            
            # Mel spectrogram (10 mean + 10 std = 20)
            features.extend(mel_spec_mean)
            features.extend(mel_spec_std)
            
            # Time domain (2)
            features.extend([rms, zcr])
            
            # Statistical (4)
            features.extend([audio_mean, audio_std, audio_skew, audio_kurtosis])
            
            feature_vector = np.array(features, dtype=np.float32)
            
            # Replace any NaN or Inf values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(58, dtype=np.float32)

# -----------------------------
# Audio Loading with FFmpeg
# -----------------------------
def load_audio_ffmpeg(path, sr=22050, duration=30):
    """Decode audio using ffmpeg for better format support"""
    try:
        max_len = sr * duration
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(sr),
            "-t", str(duration),
            "-"
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        audio = np.frombuffer(proc.stdout, np.float32)
        
        # Ensure correct length
        if len(audio) > max_len:
            audio = audio[:max_len]
        elif len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
            
        return audio, sr
        
    except Exception as e:
        # Fallback to librosa
        print(f"FFmpeg failed for {path}, using librosa fallback: {e}")
        try:
            audio, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            return audio, sr
        except Exception as lib_e:
            print(f"Librosa also failed for {path}: {lib_e}")
            return np.zeros(max_len), sr

# -----------------------------
# Genre Mapping Functions
# -----------------------------
def get_track_id_from_path(audio_path):
    """Extract track ID from FMA file path"""
    try:
        filename = os.path.basename(audio_path)
        track_id = int(filename.split('.')[0])
        return track_id
    except:
        return None

def get_root_genre_fast(track_id, tracks_df, genres_df):
    """Proper genre mapping for FMA dataset with robust error handling"""
    if tracks_df is None or genres_df is None:
        print(f"No metadata available for track {track_id}")
        return None
    
    try:
        # Check if track exists in metadata
        if track_id not in tracks_df.index:
            print(f"Track {track_id} not found in metadata")
            return None
            
        genre_top = None
        
        # Try different column access patterns for FMA multi-index columns
        try:
            # Method 1: Multi-level index (standard FMA format)
            genre_top = tracks_df.loc[track_id, ('track', 'genre_top')]
        except (KeyError, TypeError):
            try:
                # Method 2: Flattened columns
                for col in tracks_df.columns:
                    if 'genre_top' in str(col):
                        genre_top = tracks_df.loc[track_id, col]
                        break
            except:
                try:
                    # Method 3: Direct access
                    if 'genre_top' in tracks_df.columns:
                        genre_top = tracks_df.loc[track_id, 'genre_top']
                except:
                    pass
        
        if pd.isna(genre_top) or genre_top is None:
            print(f"No genre found for track {track_id}")
            return None
        
        # Clean genre name
        if isinstance(genre_top, str):
            genre_top = genre_top.strip()
        
        # Map to our root genre index
        result = ROOT_GENRES.get(genre_top, None)
        
        if result is None:
            print(f"Genre '{genre_top}' not in our mapping for track {track_id}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Genre lookup error for track {track_id}: {e}")
        return None

# -----------------------------
# Multiprocessing Setup
# -----------------------------
tracks_df_global = None
genres_df_global = None

def init_worker(tracks_df, genres_df):
    """Initialize worker process with shared metadata"""
    global tracks_df_global, genres_df_global
    tracks_df_global = tracks_df
    genres_df_global = genres_df

def process_audio_file(audio_file):
    """Process single audio file with comprehensive error handling"""
    global tracks_df_global, genres_df_global
    
    try:
        feature_extractor = SpectralFeatureExtractor()
        
        # Extract track ID
        track_id = get_track_id_from_path(audio_file)
        if track_id is None:
            return None
        
        # Get genre
        root_genre = get_root_genre_fast(track_id, tracks_df_global, genres_df_global)
        if root_genre is None:
            return None

        # Load audio
        audio, sr = load_audio_ffmpeg(audio_file, sr=22050, duration=30)
        
        # Check audio quality
        if len(audio) < 1000 or np.max(np.abs(audio)) < 0.01:
            print(f"Low quality audio for track {track_id}")
            return None

        # Extract features
        feature_vector = feature_extractor.extract_features(audio)
        
        # Validate features
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            print(f"Invalid features for track {track_id}")
            return None
            
        if np.all(feature_vector == 0):
            print(f"Zero features for track {track_id}")
            return None

        return {
            'features': feature_vector,
            'genre': root_genre,
            'track_id': track_id,
            'file_path': audio_file
        }

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# -----------------------------
# Parallel Processing
# -----------------------------
def process_files_in_chunks(audio_files, tracks_df, genres_df, chunk_size=1000):
    """Optimized parallel processing with progress tracking"""
    features_list = []
    genres_list = []
    track_ids_list = []
    successful_files = 0
    
    num_cores = min(mp.cpu_count(), 16)  # Conservative core count
    print(f"Using {num_cores} workers for parallel processing")
    
    total_files = len(audio_files)
    total_chunks = (total_files + chunk_size - 1) // chunk_size
    
    with tqdm.tqdm(total=total_files, desc="Processing audio files", 
                   unit="files", unit_scale=True) as pbar:
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_files)
            chunk_files = audio_files[start_idx:end_idx]
            
            with ProcessPoolExecutor(
                max_workers=num_cores,
                initializer=init_worker,
                initargs=(tracks_df, genres_df)
            ) as executor:
                # Process chunk
                chunk_results = list(executor.map(process_audio_file, chunk_files))
            
            # Collect results
            for result in chunk_results:
                if result is not None:
                    features_list.append(result['features'])
                    genres_list.append(result['genre'])
                    track_ids_list.append(result['track_id'])
                    successful_files += 1
                
                pbar.update(1)
            
            # Progress reporting
            if chunk_idx % 10 == 0 and chunk_idx > 0:
                success_rate = (successful_files / pbar.n) * 100
                pbar.set_postfix({
                    'success_rate': f'{success_rate:.1f}%',
                    'successful': successful_files
                })
            
            # Clean up memory
            if chunk_idx % 20 == 0:
                gc.collect()
    
    return features_list, genres_list, track_ids_list, successful_files

# -----------------------------
# Metadata Loading
# -----------------------------
def load_metadata(metadata_dir):
    """Load FMA metadata with robust error handling"""
    tracks_df = None
    genres_df = None
    
    try:
        tracks_path = os.path.join(metadata_dir, 'tracks.csv')
        genres_path = os.path.join(metadata_dir, 'genres.csv')
        
        if os.path.exists(tracks_path):
            print("Loading tracks metadata...")
            # Try different loading strategies for FMA CSV format
            try:
                tracks_df = pd.read_csv(tracks_path, header=[0, 1], index_col=0, low_memory=False)
            except:
                try:
                    tracks_df = pd.read_csv(tracks_path, index_col=0, low_memory=False)
                except:
                    tracks_df = pd.read_csv(tracks_path, low_memory=False)
            
            print(f"Loaded tracks metadata: {len(tracks_df)} tracks")
            print(f"Track index sample: {tracks_df.index[:5]}")
            
            # Check for genre columns
            genre_cols = [col for col in tracks_df.columns if 'genre' in str(col).lower()]
            print(f"Genre-related columns: {genre_cols}")
            
        else:
            print(f"Warning: tracks.csv not found at {tracks_path}")
            
        if os.path.exists(genres_path):
            print("Loading genres metadata...")
            genres_df = pd.read_csv(genres_path)
            print(f"Loaded genres metadata: {len(genres_df)} genres")
            
            # Display root genres
            root_genres = genres_df[pd.isna(genres_df['parent']) | (genres_df['parent'] == 0)]
            print(f"Root genres found: {len(root_genres)}")
            for _, row in root_genres.iterrows():
                print(f"  {row['genre_id']}: {row['title']}")
                
        else:
            print(f"Warning: genres.csv not found at {genres_path}")
            
    except Exception as e:
        print(f"Error loading metadata: {e}")
        print("Will proceed with fallback genre estimation")
        
    return tracks_df, genres_df

# -----------------------------
# Main Function
# -----------------------------
def main():
    """Main processing function with comprehensive error handling"""
    print("FMA Spectral Feature Extraction - FIXED VERSION")
    print("=" * 50)
    
    # Configuration
    dataset_dir = "./fma_medium"
    metadata_dir = "./fma_metadata"
    output_dir = "./spectral_features"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    tracks_df, genres_df = load_metadata(metadata_dir)
    
    if tracks_df is None:
        print("Warning: No tracks metadata found. Genre mapping will be limited.")
    
    # Find audio files
    print("Searching for audio files...")
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
        pattern = os.path.join(dataset_dir, "**", ext)
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    if MAX_FILES is not None:
        audio_files = audio_files[:MAX_FILES]
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found! Check dataset directory.")
        return
    
    # Process files
    print("Starting feature extraction...")
    features_list, genres_list, track_ids_list, successful_files = process_files_in_chunks(
        audio_files, tracks_df, genres_df, chunk_size=1000
    )
    
    if len(features_list) == 0:
        print("No features extracted successfully!")
        return
    
    # Convert to arrays
    print("Converting to arrays...")
    features_array = np.stack(features_list)
    genres_array = np.array(genres_list, dtype=np.int32)
    track_ids_array = np.array(track_ids_list, dtype=np.int32)
    
    # Analyze results
    print("\nFeature Extraction Complete!")
    print("=" * 50)
    print(f"Successfully processed: {successful_files}/{len(audio_files)} files ({successful_files/len(audio_files)*100:.1f}%)")
    print(f"Final features shape: {features_array.shape}")
    print(f"Genres distribution: {np.bincount(genres_array)}")
    
    # Save results
    print("Saving results...")
    np.savez_compressed(os.path.join(output_dir, "optimized_features.npz"),
                       features=features_array,
                       genres=genres_array,
                       track_ids=track_ids_array)
    
    # Also save individual files for compatibility
    np.save(os.path.join(output_dir, "spectral_features.npy"), features_array)
    np.save(os.path.join(output_dir, "root_genres.npy"), genres_array)
    np.save(os.path.join(output_dir, "track_ids.npy"), track_ids_array)
    
    # Save feature information
    feature_info = {
        'feature_dim': features_array.shape[1],
        'num_samples': features_array.shape[0],
        'genre_distribution': np.bincount(genres_array).tolist(),
        'success_rate': successful_files / len(audio_files)
    }
    
    import json
    with open(os.path.join(output_dir, "feature_info.json"), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print("Feature extraction complete!")

if __name__ == "__main__":
    # Set multiprocessing start method
    if mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main()