# SoundSage: Real-Time Music Intelligence

## 🎯 About The Project

SoundSage represents a breakthrough in real-time audio processing, combining sophisticated machine learning with cutting-edge hardware acceleration. This dual-model system delivers instant music genre recognition and intelligent audio enhancement, all running efficiently on MemryX NPU hardware.

### The Challenge
Traditional music analysis systems suffer from several limitations:
- Cloud dependency causing latency and privacy concerns
- Class imbalance in genre classification leading to biased predictions
- Separate systems for analysis and audio enhancement
- High computational requirements unsuitable for edge devices

### Our Solution
SoundSage addresses these challenges through an integrated approach:

**Dual-Model Architecture:**
- **Genre Prediction Model**: Accurately identifies music genres from audio features
- **Live EQ Model**: Automatically optimizes equalizer settings based on genre context

**Hardware-Optimized Deployment:**
- Trained on Modal's scalable infrastructure
- Converted to ONNX for interoperability
- Deployed as DFP files on MemryX NPU for maximum performance

**Balanced Training Methodology:**
- Processed 22,000+ sound clips from FMA dataset
- Strategic class balancing (8 genres × 1,000 samples each)
- Feature selection optimizing for both accuracy and efficiency

### Technical Innovation

**Data Pipeline Excellence:**

# Balanced dataset creation
class BalancedFMADataset:
    - Filters genres with ≥1000 samples
    - Subsamples to exactly 1000 per class
    - Selects top 60 features via ANOVA F-test
    - Applies standardized normalization
Model Architecture:

python
GenreClassifier(
    input(60) → Linear(256) → ReLU → Dropout(0.3)
            → Linear(128) → ReLU → Dropout(0.3) 
            → Linear(64) → ReLU → Dropout(0.2)
            → Linear(8)  # 8 balanced genres
)
Performance Optimizations:

Feature selection reducing from 80 to 60 most relevant features

Regularization techniques (dropout, weight decay, label smoothing)

Cosine annealing learning rate scheduling

Early stopping based on overfitting gap monitoring

Key Features
🎵 Real-time Processing

<5ms inference time on MemryX NPU

Continuous audio stream analysis

Instant genre detection and EQ adjustment

🔒 Privacy-First Architecture

Fully local processing - no cloud dependency

On-device model execution

Secure audio processing pipeline

⚡ Hardware Efficiency

MemryX NPU optimization via DFP format

Low power consumption suitable for mobile/embedded

Memory-efficient model architecture

🎛️ Intelligent Audio Enhancement

Genre-aware EQ parameter optimization

Contextual audio improvements

Preserves original audio quality

Supported Genres
The system currently recognizes and optimizes for 8 major music genres:

Rock • Electronic • Hip-Hop • Classical

Jazz • Folk • Pop • Experimental

Impact & Applications
Consumer Electronics:

Smart speakers with auto-EQ adjustment

Car audio systems with genre-optimized sound profiles

Headphones with intelligent sound enhancement

Professional Audio:

Live sound engineering assistance

Broadcast audio optimization

Music production tools

Developer Platform:

API for genre detection services

Custom EQ model training framework

Hardware acceleration SDK

Performance Metrics
Genre Accuracy: >85% on balanced test set

Latency: <5ms end-to-end processing

Model Size: Optimized for edge deployment

Power Efficiency: NPU-optimized for continuous operation

Future Roadmap
Expand to 16+ genre classifications

Multi-modal input (audio + metadata)

Personalized EQ preferences

Cross-platform SDK release

Real-time audio effect chain optimization

