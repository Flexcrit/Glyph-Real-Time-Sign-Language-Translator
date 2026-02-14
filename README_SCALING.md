# 🚀 Scaling Your ASL Detector to All Sign Words

This guide explains how to expand your sign language detector to recognize hundreds or thousands of ASL signs.

## 📋 Table of Contents
1. [Current System Overview](#current-system-overview)
2. [Three Scaling Approaches](#three-scaling-approaches)
3. [Quick Start: Adding More Signs](#quick-start-adding-more-signs)
4. [Using Pre-trained Datasets](#using-pre-trained-datasets)
5. [Training Tips for Large Vocabularies](#training-tips-for-large-vocabularies)
6. [Performance Optimization](#performance-optimization)

---

## 🎯 Current System Overview

Your system is **already designed to scale**! Here's what you have:

- ✅ **LSTM-based sequence model** (handles variable vocabulary sizes)
- ✅ **MediaPipe Holistic landmarks** (1662 features: pose + face + hands)
- ✅ **30-frame sequences** (captures motion dynamics)
- ✅ **One-hot encoding** (supports unlimited classes)
- ✅ **Modular architecture** (easy to add new signs)

**Current limitation:** You need training data for each sign word.

---

## 🛤️ Three Scaling Approaches

### Approach 1: Manual Data Collection
**Best for:** Custom vocabulary, high accuracy, small-to-medium vocabularies (10-100 signs)

#### Steps:
```bash
# Option A: Collect one sign at a time
python collect_data.py
# Enter sign label: "hello"
# Enter sign index: 0

# Option B: Batch collection (recommended)
python batch_collect_data.py
```

#### Editing the Vocabulary:
Open `batch_collect_data.py` and modify the `VOCABULARY` list:
```python
VOCABULARY = [
    "hello",
    "goodbye", 
    "thank_you",
    "please",
    # Add your signs here...
]
```

#### Time Estimate:
- ~5-10 minutes per sign (30 sequences)
- 50 signs = ~5-8 hours
- 100 signs = ~10-16 hours

---

### Approach 2: Use Pre-trained Datasets
**Best for:** Large vocabularies (100-2000 signs), faster deployment

#### Popular ASL Datasets:

| Dataset | # Signs | # Videos | Format | Link |
|---------|---------|----------|--------|------|
| **WLASL** | 2,000 | 21,000+ | MP4 videos | [GitHub](https://github.com/dxli94/WLASL) |
| **MS-ASL** | 1,000 | 25,000+ | MP4 videos | [Microsoft](https://www.microsoft.com/en-us/research/project/ms-asl/) |
| **ASL Citizen** | 80+ | 80,000+ | MP4 videos | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) |
| **ChicagoFSWild** | 500+ | 7,300+ | MP4 videos | [UChicago](https://home.ttic.edu/~klivescu/ChicagoFSWild.html) |

#### How to Use Pre-trained Dataset:

**Step 1: Download Dataset**
```bash
# Example: WLASL
git clone https://github.com/dxli94/WLASL.git
cd WLASL
# Follow their download instructions
```

**Step 2: Convert to Your Format**
Create a script to process videos:
```python
# See convert_dataset.py (create this script)
import cv2
import numpy as np
from LandmarkExtractor import LandmarkExtractor

def video_to_sequences(video_path, num_sequences=5):
    """Convert a video to multiple 30-frame sequences with landmarks."""
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(video_path)
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks, _ = extractor.process_frame_with_drawing(frame)
        all_frames.append(landmarks)
    
    cap.release()
    extractor.close()
    
    # Split into 30-frame sequences
    sequences = []
    for i in range(0, len(all_frames) - 30, 10):  # Overlapping windows
        sequence = all_frames[i:i+30]
        if len(sequence) == 30:
            sequences.append(np.array(sequence))
    
    return sequences[:num_sequences]  # Limit number of sequences
```

**Step 3: Organize Data**
```bash
# Convert all videos and save in your format
python convert_dataset.py --input WLASL/videos --output training_data
```

**Step 4: Train**
```bash
python train_model.py
```

---

### Approach 3: Hybrid (Recommended)
**Best for:** Best of both worlds

1. **Download WLASL** (2,000 common signs)
2. **Add custom signs** for domain-specific vocabulary
3. **Train combined model**

---

## ⚡ Quick Start: Adding More Signs

### Easiest Method (Batch Collection):

```bash
# 1. Edit vocabulary in batch_collect_data.py
# Add your desired signs to the VOCABULARY list

# 2. Run batch collection
python batch_collect_data.py

# 3. Select option 1 (or 2 to continue previous session)

# 4. For each sign:
#    - Press SPACE when ready
#    - Perform the sign during recording
#    - Repeat 30 times per sign

# 5. After collection, train the model
python train_model.py

# 6. Test with the app
streamlit run app.py
```

### Tips for Data Collection:
- 🎥 Use good lighting
- 👕 Wear contrasting clothing (solid colors work best)
- 🎭 Vary your signing speed and style slightly between sequences
- 📐 Keep the same background position
- 🔄 Include natural variations (different hand positions, speeds)

---

## 🏋️ Training Tips for Large Vocabularies

### When You Have 100+ Signs:

**1. Increase Model Capacity**
Edit `SignModel.py`:
```python
# Increase LSTM units
model.add(LSTM(128, return_sequences=True))  # was 64
model.add(LSTM(128, return_sequences=True))  # was 64
model.add(LSTM(128))  # was 64
```

**2. Collect More Data Per Sign**
```python
# In batch_collect_data.py or collect_data.py
sequences_per_sign = 50  # Instead of 30
```

**3. Increase Training Epochs**
Edit `train_model.py`:
```python
EPOCHS = 200  # Instead of 100
```

**4. Use Data Augmentation**
Add augmentation to increase effective dataset size:
```python
# Add random noise, temporal shifts, slight scaling
def augment_sequence(sequence):
    # Add small random noise to landmarks
    noise = np.random.normal(0, 0.01, sequence.shape)
    return sequence + noise
```

**5. Monitor for Overfitting**
- Watch validation loss vs training loss
- Use early stopping (already implemented)
- Increase regularization if needed

---

## ⚙️ Performance Optimization

### For Real-Time Inference with 1000+ Signs:

**1. Model Compression**
```python
# After training, quantize the model
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**2. Top-K Predictions**
Already implemented! Only shows top predictions:
```python
predictions = predict_sign(model, sequence, labels, top_k=5)
```

**3. Hierarchical Classification**
For very large vocabularies (1000+ signs):
- Group signs by category (letters, numbers, greetings, etc.)
- First predict category, then sign within category
- Speeds up inference

**4. GPU Acceleration**
Ensure TensorFlow uses GPU:
```bash
pip install tensorflow-macos tensorflow-metal  # For M1/M2 Macs
# Or
pip install tensorflow-gpu  # For NVIDIA GPUs
```

---

## 📊 Data Organization

Your data structure automatically scales:
```
training_data/
├── 0000_hello/
│   ├── sequence_0000.npy
│   ├── sequence_0001.npy
│   └── ...
├── 0001_goodbye/
│   ├── sequence_0000.npy
│   └── ...
├── 0002_thank_you/
│   └── ...
...
├── 0999_[sign_name]/
│   └── ...
└── labels.json  (auto-generated during training)
```

No code changes needed - just add more sign directories!

---

## 🎓 Recommended Sign Progression

Start small and expand:

### Stage 1: Core Vocabulary (10-20 signs)
Letters: A-Z, Numbers: 0-9

### Stage 2: Common Words (50 signs)
- Greetings: hello, goodbye, how, what, when
- Politeness: please, thank you, sorry
- Basics: yes, no, help, want, need
- People: I/me, you, we, family, friend
- Actions: go, stop, eat, drink, sleep

### Stage 3: Conversational (100-200 signs)
Add domain vocabulary based on use case

### Stage 4: Comprehensive (500+ signs)
Use pre-trained datasets + custom additions

---

## 🐛 Troubleshooting

### "Model accuracy drops with more signs"
- ✅ Collect more sequences per sign (50+ instead of 30)
- ✅ Increase model capacity (more LSTM units)
- ✅ Train for more epochs
- ✅ Ensure balanced dataset (same # sequences per sign)

### "Training takes too long"
- ✅ Use GPU acceleration
- ✅ Reduce batch size if memory is limited
- ✅ Use fewer training epochs initially, then fine-tune

### "Real-time predictions are slow"
- ✅ Use model quantization (TFLite)
- ✅ Reduce MediaPipe model complexity
- ✅ Lower camera resolution
- ✅ Process every Nth frame instead of all frames

---

## 🎯 Summary

**Your system is ready to scale!** Choose your approach:

| If you want... | Use this approach | Time needed |
|----------------|-------------------|-------------|
| 10-50 custom signs | Manual collection with `batch_collect_data.py` | 1-5 hours |
| 100-500 signs, fast | Pre-trained dataset (WLASL) + conversion | 2-4 hours |
| 1000+ signs | Pre-trained dataset + custom additions | 4-8 hours |

**Next step:** Run `python batch_collect_data.py` and start building your vocabulary!
