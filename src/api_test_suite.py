#!/usr/bin/env python3
"""
Test suite for AI Audio Mastering API
"""

import pytest
import requests
import io
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import json

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_AUDIO_DURATION = 5  # seconds
SAMPLE_RATE = 22050

class TestAudioMasteringAPI:
    """Test suite for the audio mastering API"""
    
    @classmethod
    def setup_class(cls):
        """Setup test class"""
        cls.base_url = API_BASE_URL
        
    def create_test_audio_file(self, duration=5, sample_rate=22050, format='wav'):
        """Create a test audio file"""
        # Generate simple sine wave
        t = np.linspace(0, duration, int(duration * sample_rate))
        frequency = 440  # A4 note
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
        
        if format == 'wav':
            import soundfile as sf
            sf.write(temp_file.name, audio_data, sample_rate)
        
        temp_file.close()
        return temp_file.name
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "audio_support" in data
        assert data["status"] == "healthy"
    
    def test_get_genres(self):
        """Test get available genres endpoint"""
        response = requests.get(f"{self.base_url}/genres")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "genres" in data
        assert isinstance(data["genres"], list)
        assert len(data["genres"]) == 16  # Should have 16 genres
        assert "Blues" in data["genres"]
        assert "Rock" in data["genres"]
    
    def test_get_eq_presets(self):
        """Test get all EQ presets endpoint"""
        response = requests.get(f"{self.base_url}/eq-presets")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "presets" in data
        assert isinstance(data["presets"], dict)
        assert "Blues" in data["presets"]
        assert "Electronic" in data["presets"]
        
        # Check that each preset has 15 values
        for genre, preset in data["presets"].items():
            assert isinstance(preset, list)
            assert len(preset) == 15
    
    def test_get_genre_eq_preset(self):
        """Test get specific genre EQ preset"""
        # Test valid genre
        response = requests.get(f"{self.base_url}/eq-presets/Blues")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "genre" in data
        assert "eq_adjustments" in data
        assert data["genre"] == "Blues"
        assert isinstance(data["eq_adjustments"], list)
        assert len(data["eq_adjustments"]) == 15
        
        # Test invalid genre
        response = requests.get(f"{self.base_url}/eq-presets/NonExistentGenre")
        assert response.status_code == 404
    
    def test_analyze_genre_valid_file(self):
        """Test genre analysis with valid audio file"""
        # Create test audio file
        test_file_path = self.create_test_audio_file()
        
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/analyze-genre", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "genre" in data
            assert "confidence" in data
            assert "all_predictions" in data
            
            assert isinstance(data["genre"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["all_predictions"], dict)
            
            assert 0.0 <= data["confidence"] <= 1.0
            assert len(data["all_predictions"]) == 16  # All genres
            
            # Check that all predictions sum to approximately 1.0
            total_prob = sum(data["all_predictions"].values())
            assert abs(total_prob - 1.0) < 0.01
            
        finally:
            os.unlink(test_file_path)
    
    def test_analyze_genre_invalid_file(self):
        """Test genre analysis with invalid file"""
        # Test with text file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_file.write(b"This is not an audio file")
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{self.base_url}/analyze-genre", files=files)
            
            assert response.status_code == 400
            
        finally:
            os.unlink(temp_file.name)
    
    def test_predict_eq_valid_file(self):
        """Test EQ prediction with valid audio file"""
        test_file_path = self.create_test_audio_file()
        
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/predict-eq", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "eq_adjustments" in data
            assert isinstance(data["eq_adjustments"], list)
            assert len(data["eq_adjustments"]) == 15
            
            # Check that all EQ values are numbers
            for eq_val in data["eq_adjustments"]:
                assert isinstance(eq_val, (int, float))
                
        finally:
            os.unlink(test_file_path)
    
    def test_predict_eq_invalid_file(self):
        """Test EQ prediction with invalid file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_file.write(b"Not audio")
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{self.base_url}/predict-eq", files=files)
            
            assert response.status_code == 400
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_audio_complete_workflow(self):
        """Test complete audio processing workflow"""
        test_file_path = self.create_test_audio_file()
        
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/process-audio", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            required_fields = [
                'file_id', 'genre', 'confidence', 'eq_adjustments',
                'original_filename', 'processed_filename'
            ]
            for field in required_fields:
                assert field in data
            
            # Validate data types
            assert isinstance(data["file_id"], str)
            assert isinstance(data["genre"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["eq_adjustments"], list)
            assert isinstance(data["original_filename"], str)
            assert isinstance(data["processed_filename"], str)
            
            # Check constraints
            assert 0.0 <= data["confidence"] <= 1.0
            assert len(data["eq_adjustments"]) == 15
            assert data["original_filename"] == "test_audio.wav"
            assert data["processed_filename"].startswith("processed_")
            assert data["processed_filename"].endswith(".wav")
            
            return data["file_id"]  # Return for download test
            
        finally:
            os.unlink(test_file_path)
    
    def test_download_processed_file(self):
        """Test downloading processed audio file"""
        # First process a file
        test_file_path = self.create_test_audio_file()
        
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                process_response = requests.post(f"{self.base_url}/process-audio", files=files)
            
            assert process_response.status_code == 200
            process_data = process_response.json()
            file_id = process_data["file_id"]
            
            # Now download the processed file
            download_response = requests.get(f"{self.base_url}/download/{file_id}")
            
            assert download_response.status_code == 200
            assert download_response.headers['content-type'] == 'audio/wav'
            assert len(download_response.content) > 0
            
        finally:
            os.unlink(test_file_path)
    
    def test_download_nonexistent_file(self):
        """Test downloading non-existent file"""
        fake_file_id = "nonexistent-file-id"
        response = requests.get(f"{self.base_url}/download/{fake_file_id}")
        
        assert response.status_code == 404
    
    def test_file_upload_size_limits(self):
        """Test file upload with various sizes"""
        # Test very short audio (should work)
        short_file_path = self.create_test_audio_file(duration=0.5)
        
        try:
            with open(short_file_path, 'rb') as f:
                files = {'file': ('short_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/analyze-genre", files=files)
            
            # Should work but might have lower confidence
            assert response.status_code == 200
            
        finally:
            os.unlink(short_file_path)
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            test_file_path = self.create_test_audio_file()
            try:
                with open(test_file_path, 'rb') as f:
                    files = {'file': ('test_audio.wav', f, 'audio/wav')}
                    response = requests.post(f"{self.base_url}/analyze-genre", files=files)
                results.append(response.status_code)
            finally:
                os.unlink(test_file_path)
        
        # Create multiple threads
        threads = []
        for i in range(3):  # Test with 3 concurrent requests
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        assert all(status == 200 for status in results)
    
    def test_error_handling_corrupted_audio(self):
        """Test error handling with corrupted audio files"""
        # Create a file with audio extension but corrupted content
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(b"CORRUPTED_AUDIO_DATA_NOT_REAL_WAV_FILE")
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                files = {'file': ('corrupted.wav', f, 'audio/wav')}
                response = requests.post(f"{self.base_url}/analyze-genre", files=files)
            
            # Should return an error
            assert response.status_code == 500
            
        finally:
            os.unlink(temp_file.name)

class TestAPIPerformance:
    """Performance tests for the API"""
    
    def test_response_times(self):
        """Test that API responses are within acceptable time limits"""
        import time
        
        # Create test file
        test_file_path = TestAudioMasteringAPI().create_test_audio_file()
        
        try:
            # Test genre analysis performance
            start_time = time.time()
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{API_BASE_URL}/analyze-genre", files=files)
            genre_time = time.time() - start_time
            
            assert response.status_code == 200
            assert genre_time < 10.0  # Should complete within 10 seconds
            
            # Test EQ prediction performance
            start_time = time.time()
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = requests.post(f"{API_BASE_URL}/predict-eq", files=files)
            eq_time = time.time() - start_time
            
            assert response.status_code == 200
            assert eq_time < 5.0  # Should complete within 5 seconds
            
            print(f"Genre analysis time: {genre_time:.2f}s")
            print(f"EQ prediction time: {eq_time:.2f}s")
            
        finally:
            os.unlink(test_file_path)

# Integration test script
def run_integration_tests():
    """Run integration tests that require the server to be running"""
    
    print("Running API Integration Tests...")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Start it with:")
        print(f"python audio_api_server.py --genre-model <path> --eq-model <path>")
        return False
    
    print("✅ Server is running")
    
    # Create test instance
    test_instance = TestAudioMasteringAPI()
    test_instance.setup_class()
    
    # Run tests
    tests = [
        ("Health Check", test_instance.test_health_check),
        ("Get Genres", test_instance.test_get_genres),
        ("Get EQ Presets", test_instance.test_get_eq_presets),
        ("Genre EQ Preset", test_instance.test_get_genre_eq_preset),
        ("Analyze Genre (Valid)", test_instance.test_analyze_genre_valid_file),
        ("Analyze Genre (Invalid)", test_instance.test_analyze_genre_invalid_file),
        ("Predict EQ (Valid)", test_instance.test_predict_eq_valid_file),
        ("Predict EQ (Invalid)", test_instance.test_predict_eq_invalid_file),
        ("Process Audio", test_instance.test_process_audio_complete_workflow),
        ("Download File", test_instance.test_download_processed_file),
        ("Download Non-existent", test_instance.test_download_nonexistent_file),
        ("File Size Limits", test_instance.test_file_upload_size_limits),
        ("Error Handling", test_instance.test_error_handling_corrupted_audio),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...", end=" ")
            test_func()
            print("✅ PASS")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL - {str(e)}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI Audio Mastering API")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    
    args = parser.parse_args()
    
    # Update API URL if provided
    API_BASE_URL = args.url.rstrip('/')
    TestAudioMasteringAPI.base_url = API_BASE_URL
    
    # Run integration tests
    success = run_integration_tests()
    
    # Run performance tests if requested
    if args.performance:
        print("\nRunning Performance Tests...")
        perf_test = TestAPIPerformance()
        try:
            perf_test.test_response_times()
            print("✅ Performance tests passed")
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            success = False
    
    exit(0 if success else 1)