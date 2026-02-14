# ASL Translation System - Models Directory

This directory contains trained models for ASL recognition.

## Model Files

- `asl_model.h5` - Trained LSTM model (created after training)

## Training Your Model

1. Collect training data:
   ```bash
   python collect_data.py
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. The trained model will be saved here as `asl_model.h5`

## Model Architecture

See `SignModel.py` for the complete architecture definition:
- Input: (30, 1662) sequences
- LSTM(64) → LSTM(128) → Dense(64) → Softmax
- Output: (num_classes,) probability distribution

## File Formats

- `.h5` - HDF5 format (recommended)
- SavedModel format also supported
