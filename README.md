# 🤟 Real-Time Sign Language Detection System

A high-fidelity American Sign Language (ASL) translation system that uses **MediaPipe Holistic** for landmark detection and **LSTM neural networks** for real-time sign recognition.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Details](#model-details)

---

## 🎯 Overview

This project implements a real-time sign language detection system capable of recognizing ASL signs from a webcam feed. The system extracts body, hand, and facial landmarks using Google's MediaPipe, processes them through an LSTM neural network, and provides immediate predictions with a clean Streamlit interface.

### Key Capabilities
- ✅ **Real-time detection** from webcam at ~30 FPS
- ✅ **High accuracy** using deep LSTM networks
- ✅ **Scalable architecture** - easily add new signs
- ✅ **Motion-based recognition** - captures dynamic signing
- ✅ **Prediction stabilization** - reduces flickering
- ✅ **Distance & position invariant** - works at any distance/position

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.9** - Core programming language

### Machine Learning & Computer Vision
- **TensorFlow 2.20** / **Keras** - Deep learning framework for LSTM models
- **MediaPipe 0.10.9** - Google's ML solution for landmark detection
  - Holistic model (pose + hands + face)
  - 1,662 feature points per frame
- **NumPy 1.26** - Numerical computations and array operations

### Computer Vision & Video Processing
- **OpenCV 4.10** - Camera capture and image processing
- **Pillow** - Image manipulation

### Web Interface
- **Streamlit 1.50** - Interactive web UI for real-time predictions
- **streamlit-webrtc** - WebRTC support for camera streaming

### Data Science & Utilities
- **scikit-learn 1.6** - Train/test splitting and metrics
- **Matplotlib 3.9** - Training visualizations
- **SciPy 1.13** - Scientific computations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER WORKFLOW                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DATA COLLECTION (collect_data.py)                      │
│     ├─ Webcam capture                                      │
│     ├─ MediaPipe landmark extraction                       │
│     └─ Save 30-frame sequences (.npy files)                │
│                                                             │
│  2. MODEL TRAINING (train_model.py)                        │
│     ├─ Load collected sequences                            │
│     ├─ Build LSTM model                                    │
│     ├─ Train with validation split                         │
│     └─ Save trained model (.h5)                            │
│                                                             │
│  3. REAL-TIME INFERENCE (app.py)                           │
│     ├─ Live webcam feed                                    │
│     ├─ Continuous landmark extraction                      │
│     ├─ 30-frame rolling buffer                             │
│     ├─ LSTM prediction                                     │
│     ├─ Prediction stabilization                            │
│     └─ Streamlit UI display                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Webcam Frame → MediaPipe → Landmarks (1662 features) 
    → Normalization → 30-Frame Sequence → LSTM Model 
    → Softmax Predictions → Stabilization → Display
```

---

## ✨ Features

### 1. **Landmark Extraction (`LandmarkExtractor.py`)**
- Extracts **1,662 features per frame**:
  - **Left hand**: 21 landmarks × 3 coords = 63 features
  - **Right hand**: 21 landmarks × 3 coords = 63 features
  - **Pose**: 33 landmarks × 3 coords = 99 features
  - **Face**: 468 landmarks × 3 coords = 1,404 features
  - **Pose visibility**: 33 landmarks = 33 features

- **Normalization Strategy**:
  - Position-invariant: All coordinates relative to nose position
  - Scale-invariant: Normalized by shoulder width
  - Enables recognition regardless of distance from camera

### 2. **LSTM Neural Network (`SignModel.py`)**
- **Architecture**:
  ```
  Input (30, 1662) 
    → LSTM(64) + Dropout(0.2)
    → LSTM(128) + Dropout(0.3) 
    → Dense(64, ReLU) + Dropout(0.2)
    → Dense(num_classes, Softmax)
  ```
  
- **Training Features**:
  - Adam optimizer with learning rate 0.001
  - Categorical crossentropy loss
  - Early stopping (patience: 15 epochs)
  - Learning rate reduction on plateau
  - Model checkpointing (saves best model)

### 3. **Prediction Stabilization**
- Requires **10 consecutive frames** with same prediction
- Minimum **90% confidence** threshold
- Prevents flickering and false positives
- Builds sentence by adding stable predictions

### 4. **Data Collection Tools**
- **Single sign collection** (`collect_data.py`)
- **Batch collection** (`batch_collect_data.py`)
- Visual countdown and progress bars
- Customizable vocabulary

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Webcam
- macOS / Linux / Windows

### Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd sign-language-detector

# 2. Create virtual environment
python3.9 -m venv .venv39
source .venv39/bin/activate  # On Windows: .venv39\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Start

```bash
# Simply run the script
./run_app.sh
```

### Step-by-Step Workflow

#### 1️⃣ **Collect Training Data**

```bash
# For a single sign
python collect_data.py

# For multiple signs (batch)
# First, edit batch_collect_data.py to add your vocabulary
python batch_collect_data.py
```

**Tips for data collection:**
- Use good lighting
- Perform each sign 30 times
- Vary speed and style slightly
- Keep hands visible in frame

#### 2️⃣ **Train the Model**

```bash
python train_model.py
```

This will:
- Load all collected data from `training_data/`
- Build and train the LSTM model
- Save the model to `models/asl_model.h5`
- Generate `labels.json` mapping
- Display training metrics

#### 3️⃣ **Run Real-Time Detection**

```bash
# Simple camera test
streamlit run app_simple.py

# Full app with predictions
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📁 Project Structure

```
sign-language-detector/
├── LandmarkExtractor.py      # MediaPipe landmark extraction
├── SignModel.py               # LSTM model architecture & training
├── collect_data.py            # Single sign data collection
├── batch_collect_data.py      # Batch data collection
├── train_model.py             # Model training script
├── app.py                     # Full Streamlit app with predictions
├── app_simple.py              # Simple camera test
├── run_app.sh                 # Convenience runner script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── QUICKSTART.md             # Quick start guide
├── README_SCALING.md         # Guide for scaling to 100s of signs
├── training_data/            # Collected sign sequences
│   ├── 0000_hello/
│   │   ├── sequence_0000.npy
│   │   └── ...
│   └── 0001_goodbye/
│       └── ...
├── models/
│   └── asl_model.h5          # Trained model
└── labels.json               # Sign label mappings
```

---

## 🧠 How It Works

### 1. **Landmark Detection**

MediaPipe Holistic processes each frame to detect:
- **Pose landmarks**: Body position (shoulders, elbows, wrists, etc.)
- **Hand landmarks**: Detailed finger positions for both hands
- **Face landmarks**: Facial expressions and head orientation

All landmarks are normalized to be invariant to:
- **Distance**: Works whether you're close or far from camera
- **Position**: Works regardless of where you stand in frame
- **Scale**: Normalized by shoulder width

### 2. **Sequence Processing**

Signs are recognized as **30-frame sequences** (~1 second at 30 FPS):
- Captures the **motion dynamics** of signing
- Uses a **rolling buffer** for continuous detection
- Each sequence: `(30 frames, 1662 features) = (30, 1662)` array

### 3. **LSTM Classification**

The LSTM network:
- Learns **temporal patterns** in sign movements
- Processes entire sequences, not individual frames
- Outputs probability distribution over all known signs
- Uses **softmax activation** for multi-class prediction

### 4. **Prediction Stabilization**

To prevent false positives:
1. Only accept predictions with ≥90% confidence
2. Require same prediction for 10 consecutive frames
3. Add to sentence when stable
4. Reset buffer after accepting a sign

---

## 📊 Model Details

### Input Shape
- **(batch_size, 30, 1662)**
  - 30 frames per sequence
  - 1,662 features per frame

### Architecture Summary
```
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)               (None, 30, 64)            442,112
dropout (Dropout)           (None, 30, 64)            0
lstm_2 (LSTM)               (None, 128)               98,816
dropout_1 (Dropout)         (None, 128)               0
dense_1 (Dense)             (None, 64)                8,256
dropout_2 (Dropout)         (None, 64)                0
output (Dense)              (None, num_classes)       varies
=================================================================
Total params: ~549,000+ (varies with num_classes)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Top-5 Accuracy
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20%

### Performance Characteristics
- **Inference Time**: ~10-30ms per prediction
- **FPS**: ~30 frames per second
- **Memory**: ~500MB (model + MediaPipe)

---

## 🎓 Methods & Algorithms

### Feature Extraction
- **MediaPipe Holistic**: Pre-trained ML models for landmark detection
- **Normalization**: Position and scale invariance through geometric transformations

### Deep Learning
- **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in sign sequences
- **Dropout Regularization**: Prevents overfitting (20-30% dropout)
- **Early Stopping**: Automatic training termination when validation loss plateaus

### Signal Processing
- **Rolling Buffer**: Continuous 30-frame window for real-time processing
- **Prediction Smoothing**: Temporal consistency through multi-frame voting

---

## 🔧 Configuration

### Adjusting Model Capacity
Edit `SignModel.py`:
```python
# Increase for larger vocabularies
layers.LSTM(128, ...)  # Increase from 64
layers.LSTM(256, ...)  # Increase from 128
```

### Adjusting Confidence Threshold
Edit `app.py`:
```python
stabilizer = PredictionStabilizer(
    min_confidence=0.85,  # Lower for easier acceptance
    stability_frames=8     # Lower for faster response
)
```

---

## 📈 Scaling to More Signs

See **[README_SCALING.md](README_SCALING.md)** for detailed guide on:
- Collecting data for 100+ signs
- Using pre-trained datasets (WLASL, MS-ASL)
- Optimizing model for large vocabularies
- Performance tuning

---

## 🐛 Troubleshooting

### Camera not working
- Check camera permissions in System Preferences
- Try different camera index (0, 1, 2)

### Low accuracy
- Collect more training sequences (50+ per sign)
- Ensure good lighting during data collection
- Increase model capacity
- Train for more epochs

### Slow inference
- Reduce MediaPipe model complexity
- Use GPU acceleration
- Lower camera resolution

---

## 📝 License

MIT License - feel free to use for educational and commercial purposes.

---

## 🙏 Acknowledgments

- **Google MediaPipe** - Landmark detection
- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web interface

---

## 📬 Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Built with ❤️ using Python, TensorFlow, and MediaPipe**
