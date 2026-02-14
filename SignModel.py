import os
import json
import numpy as np
from typing import Optional, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_model(num_classes: int, sequence_length: int = 30, num_features: int = 1662) -> keras.Model:
    
    model = Sequential([
        layers.Input(shape=(sequence_length, num_features)),
        
        layers.LSTM(
            64,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='lstm_1'
        ),
        layers.Dropout(0.2),
        
        layers.LSTM(
            128,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='lstm_2'
        ),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='ASL_LSTM_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model


def load_model(model_path: str) -> keras.Model:
    """
    Load a pre-trained model from disk.
    
    Args:
        model_path: Path to saved model (.h5 or SavedModel format)
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras_load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model


def save_model(model: keras.Model, save_path: str):
    """
    Save model to disk.
    
    Args:
        model: Keras model to save
        save_path: Path where to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.save(save_path)
    print(f"✓ Model saved to {save_path}")


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: str = 'models/asl_model.h5',
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 7
) -> keras.callbacks.History:
    """
    Train the ASL recognition model.
    
    Args:
        model: Keras model to train
        X_train: Training data of shape (num_samples, sequence_length, num_features)
        y_train: Training labels (one-hot encoded)
        X_val: Validation data (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save best model
        early_stopping_patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs to wait before reducing learning rate
        
    Returns:
        Training history object
    """
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    callbacks = [
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy' if validation_data else 'accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n{'='*60}")
    print(f"Training ASL Model")
    print(f"{'='*60}")
    print(f"Training samples: {X_train.shape[0]}")
    if validation_data:
        print(f"Validation samples: {X_val.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Features per frame: {X_train.shape[2]}")
    print(f"Number of classes: {y_train.shape[1]}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def load_labels(labels_path: str = 'labels.json') -> dict:
    """
    Load class labels from JSON file.
    
    Args:
        labels_path: Path to labels.json file
        
    Returns:
        Dictionary mapping class indices to label names
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    labels = {int(k): v for k, v in labels.items()}
    
    print(f"✓ Loaded {len(labels)} class labels from {labels_path}")
    return labels


def predict_sign(
    model: keras.Model,
    sequence: np.ndarray,
    labels: dict,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Predict sign from a sequence of frames.
    
    Args:
        model: Trained Keras model
        sequence: Sequence of landmarks, shape (sequence_length, num_features)
        labels: Dictionary mapping class indices to names
        top_k: Number of top predictions to return
        
    Returns:
        List of (label, confidence) tuples sorted by confidence
    """
    if len(sequence.shape) == 2:
        sequence = np.expand_dims(sequence, axis=0)
    
    predictions = model.predict(sequence, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = [
        (labels.get(idx, f"class_{idx}"), float(predictions[idx]))
        for idx in top_indices
    ]
    
    return top_predictions


class PredictionStabilizer:
    """
    Stabilizes predictions by requiring consistent predictions over multiple frames.
    Prevents "flickering" by only accepting predictions that remain stable.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.90,
        stability_frames: int = 10
    ):
        """
        Initialize prediction stabilizer.
        
        Args:
            min_confidence: Minimum confidence threshold (default: 0.90)
            stability_frames: Number of frames prediction must be stable (default: 10)
        """
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        self.prediction_buffer = []
    
    def add_prediction(self, label: str, confidence: float) -> Optional[str]:
        """
        Add a prediction and check if it should be accepted.
        
        Args:
            label: Predicted label
            confidence: Prediction confidence
            
        Returns:
            The label if it meets stability criteria, None otherwise
        """
        if confidence >= self.min_confidence:
            self.prediction_buffer.append(label)
        else:
            self.prediction_buffer.append(None)
        
        if len(self.prediction_buffer) > self.stability_frames:
            self.prediction_buffer.pop(0)
        
        if len(self.prediction_buffer) == self.stability_frames:
            valid_predictions = [p for p in self.prediction_buffer if p is not None]
            
            if len(valid_predictions) == self.stability_frames:
                if len(set(valid_predictions)) == 1:
                    accepted_label = valid_predictions[0]
                    self.reset()
                    return accepted_label
        
        return None
    
    def reset(self):
        """Reset the prediction buffer."""
        self.prediction_buffer = []


if __name__ == "__main__":
    print("Building ASL Recognition Model...")
    print()
    
    model = build_model(num_classes=100)
    
    model.summary()
    
    print()
    print("Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print()
    print("Expected input shape: (batch_size, 30, 1662)")
    print("Expected output shape: (batch_size, num_classes)")
