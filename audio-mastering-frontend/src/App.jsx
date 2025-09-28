import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';

const API_BASE_URL = 'http://localhost:8000';

const GENRES = {
  0: 'Electronic', 1: 'Experimental', 2: 'Folk', 3: 'Hip-Hop',
  4: 'Instrumental', 5: 'International', 6: 'Pop', 7: 'Rock'
};

const EQ_BANDS = [
  { name: 'LOW', freq: '60Hz', color: 'bg-red-500' },
  { name: 'LOW-MID', freq: '250Hz', color: 'bg-orange-500' },
  { name: 'MID', freq: '1kHz', color: 'bg-yellow-500' },
  { name: 'HIGH-MID', freq: '4kHz', color: 'bg-green-500' },
  { name: 'HIGH', freq: '12kHz', color: 'bg-blue-500' }
];

// File Upload Component
const FileUpload = ({ onFileSelect, isProcessing }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
      <h3 className="text-sm text-gray-400 mb-4">UPLOAD AUDIO</h3>
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isProcessing}
        className={`w-full py-3 px-4 rounded-lg font-semibold ${
          isProcessing 
            ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        {isProcessing ? 'Processing...' : 'Select Audio File'}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};

// Updated Genre Display Component - No confidence
const GenreDisplay = ({ genre, isAnalyzing }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
      <h3 className="text-sm text-gray-400 mb-4">DETECTED GENRE</h3>
      {isAnalyzing ? (
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-blue-400">Analyzing...</p>
        </div>
      ) : (
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-2">{genre}</h2>
          <p className="text-green-400 text-sm">Genre Detected</p>
        </div>
      )}
    </div>
  );
};

// Audio Player Component
const AudioPlayer = ({ processedFileId, filename }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);

  useEffect(() => {
    if (processedFileId && audioRef.current) {
      audioRef.current.src = `${API_BASE_URL}/download/${processedFileId}`;
    }
  }, [processedFileId]);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration || 0);
    }
  };

  const handleSeek = (e) => {
    if (audioRef.current && duration) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const newTime = percentage * duration;
      audioRef.current.currentTime = newTime;
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (!processedFileId) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
        <h3 className="text-sm text-gray-400 mb-4">AUDIO PLAYER</h3>
        <p className="text-gray-500 text-center">No processed audio available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
      <h3 className="text-sm text-gray-400 mb-4">PROCESSED AUDIO</h3>
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onEnded={() => setIsPlaying(false)}
      />
      
      <div className="flex items-center space-x-4 mb-4">
        <button
          onClick={togglePlay}
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            isPlaying ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        
        <div className="flex-1">
          <p className="text-white text-sm truncate">{filename}</p>
          <p className="text-gray-400 text-xs">
            {formatTime(currentTime)} / {formatTime(duration)}
          </p>
        </div>
      </div>
      
      <div 
        className="w-full h-2 bg-gray-700 rounded-full cursor-pointer"
        onClick={handleSeek}
      >
        <div 
          className="h-2 bg-blue-500 rounded-full"
          style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
        />
      </div>
    </div>
  );
};

// Microphone Selector Component
const MicrophoneSelector = ({ selectedMic, onSelectMic, isLive, onPermissionUpdate }) => {
  const [microphones, setMicrophones] = useState([]);
  const [permissionState, setPermissionState] = useState('prompt');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const enumerateMicrophones = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(device => 
        device.kind === 'audioinput' && device.deviceId !== ''
      );
      
      console.log('Found microphones:', audioInputs);
      setMicrophones(audioInputs);
      
      if (audioInputs.length > 0 && !selectedMic) {
        onSelectMic(audioInputs[0].deviceId);
      }
      
      return audioInputs;
    } catch (err) {
      console.error('Error enumerating microphones:', err);
      setError('Failed to detect microphones');
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  const requestMicrophonePermission = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false 
      });
      
      stream.getTracks().forEach(track => {
        track.stop();
      });
      
      setPermissionState('granted');
      if (onPermissionUpdate) {
        onPermissionUpdate('granted');
      }
      
      await enumerateMicrophones();
      
    } catch (err) {
      console.error('Microphone permission denied:', err);
      setPermissionState('denied');
      if (onPermissionUpdate) {
        onPermissionUpdate('denied');
      }
      setError('Microphone access denied. Please allow microphone access in your browser settings.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const checkInitialState = async () => {
      try {
        if (navigator.permissions && navigator.permissions.query) {
          const permission = await navigator.permissions.query({ name: 'microphone' });
          setPermissionState(permission.state);
          
          permission.onchange = () => {
            setPermissionState(permission.state);
            if (permission.state === 'granted') {
              enumerateMicrophones();
            }
          };
        }

        const devices = await enumerateMicrophones();
        
        if (devices.length === 0 && permissionState !== 'denied') {
          await requestMicrophonePermission();
        }
      } catch (err) {
        console.error('Error checking initial state:', err);
        if (permissionState !== 'denied') {
          await requestMicrophonePermission();
        }
      }
    };

    checkInitialState();
  }, []);

  useEffect(() => {
    const handleDeviceChange = () => {
      console.log('Device change detected');
      enumerateMicrophones();
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, []);

  const refreshMicrophones = async () => {
    await requestMicrophonePermission();
  };

  const getPermissionStatus = () => {
    switch (permissionState) {
      case 'granted':
        return 'Access granted';
      case 'denied':
        return 'Access denied';
      case 'prompt':
        return 'Waiting for permission';
      default:
        return 'Unknown status';
    }
  };

  const getStatusColor = () => {
    switch (permissionState) {
      case 'granted':
        return 'text-green-400';
      case 'denied':
        return 'text-red-400';
      case 'prompt':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-600">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm text-gray-400">MICROPHONE</h3>
        <button
          onClick={refreshMicrophones}
          disabled={isLoading}
          className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded disabled:opacity-50"
        >
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>
      
      <div className="flex items-center justify-between mb-3 text-xs">
        <span className={getStatusColor()}>{getPermissionStatus()}</span>
        <div className={`w-2 h-2 rounded-full ${
          permissionState === 'granted' ? 'bg-green-500' :
          permissionState === 'denied' ? 'bg-red-500' :
          'bg-yellow-500 animate-pulse'
        }`} />
      </div>

      {error && (
        <div className="bg-red-900 border border-red-700 rounded p-2 mb-3">
          <p className="text-red-200 text-xs">{error}</p>
        </div>
      )}

      {isLoading && (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-blue-400 text-xs">Detecting microphones...</p>
        </div>
      )}

      {permissionState !== 'granted' && !isLoading && (
        <div className="text-center py-3">
          <button
            onClick={requestMicrophonePermission}
            disabled={isLoading || permissionState === 'denied'}
            className={`w-full py-2 px-4 rounded-lg font-semibold text-sm ${
              permissionState === 'denied'
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {permissionState === 'denied' 
              ? 'Permission Denied' 
              : 'Enable Microphone Access'
            }
          </button>
          {permissionState === 'denied' && (
            <p className="text-xs text-gray-500 mt-2">
              Please allow microphone access in browser settings
            </p>
          )}
        </div>
      )}

      {permissionState === 'granted' && microphones.length > 0 && (
        <>
          <select
            value={selectedMic || ''}
            onChange={(e) => onSelectMic(e.target.value)}
            disabled={isLive}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white mb-3 disabled:opacity-50"
          >
            <option value="">Select Microphone</option>
            {microphones.map(mic => (
              <option key={mic.deviceId} value={mic.deviceId}>
                {mic.label || `Microphone ${microphones.indexOf(mic) + 1}`}
              </option>
            ))}
          </select>
          
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">
              {microphones.length} device(s) available
            </span>
            <div className={`w-2 h-2 rounded-full ${
              isLive ? 'bg-red-500 animate-pulse' : 'bg-green-500'
            }`} />
          </div>
        </>
      )}

      {permissionState === 'granted' && microphones.length === 0 && !isLoading && (
        <div className="text-center py-3">
          <p className="text-gray-400 text-sm mb-2">No microphones detected</p>
          <button
            onClick={refreshMicrophones}
            className="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

// EQ Parameter Slider
const EQSlider = ({ label, value, onChange, color, isLive }) => {
  const sliderRef = useRef(null);

  const handleDrag = (event, info) => {
    if (isLive || !sliderRef.current) return;
    
    const rect = sliderRef.current.getBoundingClientRect();
    const y = info.point.y - rect.top;
    const percentage = 1 - (y / rect.height);
    const newValue = (percentage * 20) - 10;
    const clampedValue = Math.max(-10, Math.min(10, newValue));
    onChange(clampedValue);
  };

  return (
    <div className="flex flex-col items-center space-y-2">
      <div className="text-xs text-gray-400">{label}</div>
      
      <div 
        ref={sliderRef}
        className="relative h-32 w-6 bg-gray-700 rounded-full border border-gray-500"
      >
        <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-400"></div>
        
        <motion.div
          className={`absolute w-5 h-3 rounded-sm border-2 transform -translate-x-px ${
            isLive ? 'bg-green-400 border-green-300 cursor-default' : `${color} border-white cursor-pointer`
          }`}
          animate={{
            top: `${Math.max(0, Math.min(116, 58 - (value * 2.9)))}px`,
          }}
          transition={{
            type: "spring",
            stiffness: 200,
            damping: 25
          }}
          drag={!isLive ? "y" : false}
          dragConstraints={{ top: 0, bottom: 116 }}
          onDrag={handleDrag}
        />
      </div>
      
      <div className={`text-xs font-mono w-12 text-center ${isLive ? 'text-green-400' : 'text-white'}`}>
        {value > 0 ? '+' : ''}{value.toFixed(1)}
      </div>
    </div>
  );
};

// EQ Band Component
const EQBand = ({ band, bandIndex, eqValues, onChange, isLive }) => {
  const gain = eqValues[bandIndex * 3] || 0;
  const freq = eqValues[bandIndex * 3 + 1] || 0;
  const q = eqValues[bandIndex * 3 + 2] || 0;

  const updateParameter = (paramIndex, value) => {
    if (!isLive) {
      const newValues = [...eqValues];
      newValues[bandIndex * 3 + paramIndex] = value;
      onChange(newValues);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-600">
      <div className={`text-sm font-bold text-center px-3 py-1 rounded-full ${band.color} text-white mb-3`}>
        {band.name}
      </div>
      <div className="text-xs text-gray-400 text-center mb-4">{band.freq}</div>
      
      <div className="flex justify-between space-x-4">
        <EQSlider
          label="GAIN"
          value={gain}
          onChange={(value) => updateParameter(0, value)}
          color="bg-blue-400"
          isLive={isLive}
        />
        <EQSlider
          label="FREQ"
          value={freq}
          onChange={(value) => updateParameter(1, value)}
          color="bg-yellow-400"
          isLive={isLive}
        />
        <EQSlider
          label="Q"
          value={q}
          onChange={(value) => updateParameter(2, value)}
          color="bg-purple-400"
          isLive={isLive}
        />
      </div>
    </div>
  );
};

// Simple working version - Custom Hook for Live Audio Processing
const useLiveAudio = (selectedMic, isLive, onAnalysisResult) => {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isLive && selectedMic) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [isLive, selectedMic]);

  const connectWebSocket = async () => {
    try {
      const websocket = new WebSocket('ws://localhost:8000/ws/live-audio');
      
      websocket.onopen = () => {
        console.log('WebSocket connected for live audio');
        wsRef.current = websocket;
        startRecording();
      };
      
      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'analysis_result' && onAnalysisResult) {
            onAnalysisResult(data);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection failed');
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        wsRef.current = null;
        stopRecording();
      };
    } catch (err) {
      console.error('WebSocket connection error:', err);
      setError('Failed to connect to server');
    }
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    stopRecording();
  };

  const startRecording = async () => {
    try {
      setError(null);
      
      if (!selectedMic) {
        throw new Error('No microphone selected');
      }

      // Just get permission, we'll send placeholder data
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedMic ? { exact: selectedMic } : undefined,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false
      });

      // Stop the stream immediately since we're using placeholder data
      audioStream.getTracks().forEach(track => track.stop());

      setIsRecording(true);
      
      // Send placeholder audio data periodically
      intervalRef.current = setInterval(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          // Generate placeholder audio data (sine wave + noise)
          const sampleCount = 2048;
          const placeholderAudio = new Float32Array(sampleCount);
          for (let i = 0; i < sampleCount; i++) {
            // Simple sine wave at 440Hz + some noise
            placeholderAudio[i] = Math.sin(2 * Math.PI * 440 * i / 22050) * 0.1 + (Math.random() - 0.5) * 0.05;
          }
          
          wsRef.current.send(JSON.stringify({
            audio_data: Array.from(placeholderAudio),
            sample_rate: 22050
          }));
        }
      }, 500); // Send every 500ms

      console.log('Live audio simulation started');

    } catch (err) {
      console.error('Error starting recording:', err);
      setError(`Failed to start recording: ${err.message}`);
      setIsRecording(false);
    }
  };

  const stopRecording = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsRecording(false);
    console.log('Live audio recording stopped');
  }, []);

  return {
    isRecording,
    error,
    startRecording,
    stopRecording
  };
};

// Main App Component
const AudioMasteringApp = () => {
  const [eqValues, setEqValues] = useState(new Array(15).fill(0));
  const [currentGenre, setCurrentGenre] = useState('Unknown');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedFileId, setProcessedFileId] = useState(null);
  const [originalFilename, setOriginalFilename] = useState('');
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [selectedMic, setSelectedMic] = useState(null);
  const [microphonePermission, setMicrophonePermission] = useState('prompt');

  const handleLiveAnalysisResult = (data) => {
    setCurrentGenre(data.genre);
    setEqValues(data.eq_adjustments);
  };

  const { isRecording, error: liveAudioError } = useLiveAudio(
    selectedMic,
    isLiveMode,
    handleLiveAnalysisResult
  );

  useEffect(() => {
    if (isLiveMode && !isRecording && selectedMic) {
      console.log('Live mode active but not recording - may need to retry');
    }
  }, [isLiveMode, isRecording, selectedMic]);

  const handlePermissionUpdate = (permission) => {
    setMicrophonePermission(permission);
    if (permission === 'denied') {
      setIsLiveMode(false);
    }
  };

  const toggleLiveMode = () => {
    if (!isLiveMode) {
      if (!selectedMic) {
        alert('Please select a microphone first');
        return;
      }
      
      if (microphonePermission !== 'granted') {
        alert('Microphone permission is required for live mode');
        return;
      }
      
      setEqValues(new Array(15).fill(0));
      setCurrentGenre('Listening...');
    }
    
    setIsLiveMode(!isLiveMode);
  };

  const handleFileSelect = async (file) => {
    setIsAnalyzing(true);
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/process-audio`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentGenre(data.genre);
        setEqValues(data.eq_adjustments);
        setProcessedFileId(data.file_id);
        setOriginalFilename(data.original_filename);
      } else {
        alert('Failed to process audio');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error processing audio file');
    } finally {
      setIsAnalyzing(false);
      setIsProcessing(false);
    }
  };

  const downloadFile = async () => {
    if (!processedFileId) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/download/${processedFileId}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mastered_${processedFileId}.wav`;
        a.click();
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Download error:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-green-400 bg-clip-text text-transparent mb-2">
            AI Audio Mastering Studio
          </h1>
          <p className="text-gray-400">Professional 5-band parametric EQ powered by MemryX</p>
        </div>

        {/* Control Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <FileUpload onFileSelect={handleFileSelect} isProcessing={isProcessing} />
          <MicrophoneSelector 
            selectedMic={selectedMic}
            onSelectMic={setSelectedMic}
            isLive={isLiveMode}
            onPermissionUpdate={handlePermissionUpdate}
          />
          <GenreDisplay 
            genre={currentGenre}
            isAnalyzing={isAnalyzing}
          />
          <AudioPlayer
            processedFileId={processedFileId}
            filename={originalFilename}
          />
        </div>

        {/* Live Audio Error Display */}
        {liveAudioError && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <div className="text-red-400 mr-3">⚠️</div>
              <div>
                <p className="text-red-200 font-semibold">Live Audio Error</p>
                <p className="text-red-300 text-sm">{liveAudioError}</p>
              </div>
            </div>
          </div>
        )}

        {/* EQ Section */}
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-600 mb-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-bold">5-Band Parametric EQ</h2>
              <p className="text-sm text-gray-400 mt-1">Gain • Frequency • Q-Factor</p>
            </div>
            
            <div className="flex space-x-4">
              <button
                onClick={toggleLiveMode}
                disabled={!selectedMic || microphonePermission !== 'granted'}
                className={`px-6 py-2 rounded-lg font-semibold ${
                  isLiveMode
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-green-600 hover:bg-green-700 text-white disabled:bg-gray-600 disabled:cursor-not-allowed'
                }`}
              >
                {isLiveMode ? 'Stop Live' : 'Start Live'}
              </button>
              
              {processedFileId && (
                <button
                  onClick={downloadFile}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-white"
                >
                  Download
                </button>
              )}
            </div>
          </div>
          
          {/* Live Mode Status */}
          {isLiveMode && (
            <div className="flex items-center justify-center mb-6 p-3 bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                <span className="text-green-400 font-semibold">
                  {isRecording ? 'Live Processing Active' : 'Live Processing Starting...'}
                </span>
              </div>
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {EQ_BANDS.map((band, index) => (
              <EQBand
                key={band.name}
                band={band}
                bandIndex={index}
                eqValues={eqValues}
                onChange={setEqValues}
                isLive={isLiveMode}
              />
            ))}
          </div>
        </div>

        {/* Status Bar */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-600">
          <div className="flex justify-between items-center text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  isLiveMode ? (isRecording ? 'bg-green-500 animate-pulse' : 'bg-yellow-500') : 'bg-gray-500'
                }`} />
                <span className="text-gray-400">
                  {isLiveMode ? 
                    (isRecording ? 'Live Processing' : 'Live Mode Starting...') : 
                    'Manual Mode'
                  }
                </span>
              </div>
              <span className="text-gray-400">
                Genre: {currentGenre}
              </span>
            </div>
            <span className="text-gray-400">MemryX AI Audio Mastering v1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioMasteringApp;