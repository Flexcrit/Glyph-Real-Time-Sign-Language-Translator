import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from SignModel import build_model, train_model, save_model
from collect_data import load_collected_data


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training history saved to {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("ASL Model Training")
    print("="*60 + "\n")
    
    DATA_DIR = 'training_data'
    MODEL_SAVE_PATH = 'models/asl_model.h5'
    LABELS_SAVE_PATH = 'labels.json'
    
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    print("Loading collected data...")
    X, y, labels_dict = load_collected_data(DATA_DIR)
    
    print(f"\nDataset Summary:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Number of classes: {len(labels_dict)}")
    print(f"  Sequence length: {X.shape[1]}")
    print(f"  Features per frame: {X.shape[2]}")
    
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nSamples per class:")
    for label_idx, count in zip(unique, counts):
        label_name = labels_dict.get(label_idx, f"class_{label_idx}")
        print(f"  {label_name}: {count}")
    
    print("\nEncoding labels...")
    num_classes = len(labels_dict)
    y_encoded = to_categorical(y, num_classes=num_classes)
    print(f"  Encoded shape: {y_encoded.shape}")
    
    print("\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]} sequences")
    print(f"  Validation set: {X_val.shape[0]} sequences")
    
    print("\nBuilding model...")
    model = build_model(num_classes=num_classes)
    model.summary()
    
    print("\nStarting training...")
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=MODEL_SAVE_PATH,
        early_stopping_patience=15,
        reduce_lr_patience=7
    )
    
    plot_training_history(history)
    
    print(f"\nSaving labels to {LABELS_SAVE_PATH}...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    print("✓ Labels saved")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    val_loss, val_acc, val_top_k = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Top-5 Accuracy: {val_top_k:.4f}")
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Labels saved to: {LABELS_SAVE_PATH}")
    print(f"\nYou can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
