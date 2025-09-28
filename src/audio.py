#!/usr/bin/env python3
"""
Simple audio test - just pass through audio to identify the static issue
"""

import numpy as np
import sys
import os
import time
import threading
import queue
from collections import deque

# Import audio libraries
try:
    import pyaudio
    HAS_AUDIO = True
except ImportError as e:
    print(f"Audio libraries not available: {e}")
    sys.exit(1)

class SimpleAudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        
        # Audio buffers
        self.audio_buffer = queue.Queue(maxsize=20)
        
        # State
        self.is_running = False
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        print("Simple audio processor initialized")
    
    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Input audio callback - just capture audio"""
        if not self.is_running:
            return (in_data, pyaudio.paComplete)
        
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Put in buffer for output
        if not self.audio_buffer.full():
            self.audio_buffer.put(audio_data.copy())
        
        return (in_data, pyaudio.paContinue)
    
    def audio_output_callback(self, in_data, frame_count, time_info, status):
        """Output audio callback - just pass through"""
        try:
            if not self.audio_buffer.empty():
                output_data = self.audio_buffer.get_nowait()
                if len(output_data) == frame_count:
                    # Simple pass-through - no processing
                    return (output_data.astype(np.float32).tobytes(), pyaudio.paContinue)
        except queue.Empty:
            pass
        
        # Return silence if no audio available
        silence = np.zeros(frame_count, dtype=np.float32)
        return (silence.tobytes(), pyaudio.paContinue)
    
    def start(self):
        """Start simple audio pass-through"""
        print("Starting simple audio test...")
        
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
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            print("Audio pass-through started!")
            print("You should hear your voice/audio with minimal delay")
            print("Press Ctrl+C to stop")
            
            # Keep running until interrupted
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
        
        print("Audio test stopped")

def main():
    processor = SimpleAudioProcessor()
    processor.start()

if __name__ == "__main__":
    main()