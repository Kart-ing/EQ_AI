#!/usr/bin/env python3
"""
Test genre classification on multiple 5-second segments
"""

import numpy as np
import sys
import os
import time
import tempfile

# Import MemryX
try:
    import memryx as mx
except ImportError as e:
    print(f"❌ MemryX SDK not found: {e}")
    sys.exit(1)

# Import audio libraries
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    print("❌ Audio libraries not available")
    sys.exit(1)

# Genre mapping
GENRES = {
    0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
    4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock'
}

def split_audio_into_segments(audio_path, segment_duration=5, num_segments=6):
    """Split audio into specified number of segments"""
    print(f"🎵 Loading and splitting audio: {os.path.basename(audio_path)}")
    
    # Load full audio
    audio_data, sr = librosa.load(audio_path, sr=22050, mono=True)
    total_duration = len(audio_data) / sr
    
    print(f"📊 Total audio duration: {total_duration:.2f} seconds")
    
    # Calculate segment parameters
    samples_per_segment = int(segment_duration * sr)
    total_samples_needed = samples_per_segment * num_segments
    
    if len(audio_data) < total_samples_needed:
        print(f"⚠️ Audio too short. Need {total_samples_needed/sr:.1f}s, have {total_duration:.1f}s")
        # Take what we can get
        actual_segments = int(len(audio_data) / samples_per_segment)
        print(f"📉 Will process {actual_segments} segments instead")
        num_segments = actual_segments
    
    segments = []
    segment_files = []
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp()
    
    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        
        # Extract segment
        segment = audio_data[start_sample:end_sample]
        segments.append(segment)
        
        # Save segment to temporary file
        segment_file = os.path.join(temp_dir, f"segment_{i+1}.wav")
        sf.write(segment_file, segment, sr)
        segment_files.append(segment_file)
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        print(f"  📂 Segment {i+1}: {start_time:.1f}s - {end_time:.1f}s -> {segment_file}")
    
    return segment_files, temp_dir

def extract_features(audio_path):
    """Extract audio features for a single segment"""
    audio_data, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    features = []
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # Chroma and tempo
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    
    # RMS energy
    rms = librosa.feature.rms(y=audio_data)[0]
    
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

def predict_segment(accelerator, features):
    """Predict genre for a single segment"""
    prediction_result = None
    inference_complete = False
    
    def input_callback(stream_idx):
        return features.reshape(1, 60)
    
    def output_callback(stream_idx, *mxa_output):
        nonlocal prediction_result, inference_complete
        
        if len(mxa_output) > 0:
            output = mxa_output[0]
            
            # Process output
            if len(output.shape) > 1:
                predictions = output[0]
            else:
                predictions = output
            
            # Get prediction
            predicted_class = np.argmax(predictions)
            
            # Apply softmax for confidence
            exp_output = np.exp(predictions - np.max(predictions))
            probabilities = exp_output / np.sum(exp_output)
            confidence = probabilities[predicted_class]
            
            genre_name = GENRES.get(predicted_class, f"Unknown ({predicted_class})")
            
            prediction_result = (predicted_class, confidence, genre_name, predictions)
            inference_complete = True
    
    # Connect and run
    accelerator.connect_streams(input_callback, output_callback, 1)
    
    # Wait for result
    start_time = time.time()
    timeout = 2.0  # 2 second timeout per segment
    
    while not inference_complete and (time.time() - start_time) < timeout:
        time.sleep(0.01)
    
    return prediction_result

def main():
    if len(sys.argv) != 3:
        print("Usage: python multi_segment_test.py <model.dfp> <audio_file>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Split audio into segments
    segment_files, temp_dir = split_audio_into_segments(audio_path, segment_duration=5, num_segments=6)
    
    # Load model once
    print(f"\n🔄 Loading model...")
    try:
        accelerator = mx.MultiStreamAsyncAccl(dfp=model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print(f"✅ Model loaded successfully")
    
    # Test each segment
    print(f"\n🎯 TESTING {len(segment_files)} SEGMENTS")
    print("=" * 60)
    
    results = []
    blues_count = 0
    
    for i, segment_file in enumerate(segment_files, 1):
        print(f"\n🎵 Segment {i} ({(i-1)*5}s - {i*5}s):")
        
        try:
            # Extract features
            features = extract_features(segment_file)
            
            # Predict
            result = predict_segment(accelerator, features)
            
            if result:
                predicted_class, confidence, genre_name, predictions = result
                results.append((i, genre_name, confidence))
                
                # Check if it's Blues
                if genre_name == 'Blues':
                    blues_count += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"  {status} Predicted: {genre_name} (confidence: {confidence:.3f})")
                
                # Show top 3 for this segment
                top_indices = np.argsort(predictions)[::-1][:3]
                exp_output = np.exp(predictions - np.max(predictions))
                probabilities = exp_output / np.sum(exp_output)
                
                print(f"    Top 3:")
                for j, idx in enumerate(top_indices, 1):
                    genre = GENRES.get(idx, f"Unknown ({idx})")
                    prob = probabilities[idx]
                    marker = "🔵" if genre == 'Blues' else "  "
                    print(f"    {marker} {j}. {genre}: {prob:.3f}")
            else:
                print(f"  ❌ Prediction failed for segment {i}")
                results.append((i, "Error", 0.0))
        
        except Exception as e:
            print(f"  ❌ Error processing segment {i}: {e}")
            results.append((i, "Error", 0.0))
    
    # Stop accelerator
    try:
        accelerator.stop()
    except:
        pass
    
    # Summary
    print(f"\n" + "🎵" * 60)
    print("SUMMARY")
    print("🎵" * 60)
    
    print(f"📊 Total segments tested: {len(segment_files)}")
    print(f"🔵 Detected as Blues: {blues_count}/{len(segment_files)} ({blues_count/len(segment_files)*100:.1f}%)")
    
    print(f"\n📋 Detailed results:")
    for segment_num, genre, confidence in results:
        status = "✅" if genre == 'Blues' else "❌" if genre != "Error" else "💥"
        print(f"  Segment {segment_num}: {status} {genre} ({confidence:.3f})")
    
    if blues_count == len(segment_files):
        print(f"\n🎉 SUCCESS! All segments detected as Blues!")
    elif blues_count > len(segment_files) // 2:
        print(f"\n✅ MOSTLY CORRECT! {blues_count}/{len(segment_files)} segments detected as Blues")
    else:
        print(f"\n⚠️ MIXED RESULTS: Only {blues_count}/{len(segment_files)} segments detected as Blues")
    
    # Cleanup temporary files
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print(f"\n👋 Analysis complete!")

if __name__ == "__main__":
    main()