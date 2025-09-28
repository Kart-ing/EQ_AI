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
```python
# Balanced dataset creation
class BalancedFMADataset:
    - Filters genres with ≥1000 samples
    - Subsamples to exactly 1000 per class
    - Selects top 60 features via ANOVA F-test
    - Applies standardized normalization
