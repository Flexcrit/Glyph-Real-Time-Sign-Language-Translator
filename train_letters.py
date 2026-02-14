"""
train_letters.py

Train a Random Forest classifier for ASL fingerspelling letter detection.
Uses hand landmark data collected by collect_letters.py.

Usage:
    python train_letters.py
"""

import numpy as np
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_letter_data(data_dir: str = "letter_data"):
    """
    Load all collected letter data.
    
    Returns:
        X: feature array (num_samples, 63)
        y: label array (num_samples,)
        label_map: dict mapping index -> letter
    """
    X = []
    y = []
    label_map = {}
    label_idx = 0
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Run collect_letters.py first to collect training data.")
        return None, None, None
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".npy"):
            continue
        
        letter = filename.replace(".npy", "")
        filepath = os.path.join(data_dir, filename)
        
        data = np.load(filepath)
        num_samples = len(data)
        
        if num_samples == 0:
            continue
        
        label_map[label_idx] = letter
        X.append(data)
        y.extend([label_idx] * num_samples)
        label_idx += 1
        
        print(f"  Loaded '{letter}': {num_samples} samples")
    
    if not X:
        print("Error: No data found. Run collect_letters.py first.")
        return None, None, None
    
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y, label_map


def train():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("ASL LETTER MODEL TRAINING")
    print("=" * 60 + "\n")
    
    print("Loading data...")
    X, y, label_map = load_letter_data()
    
    if X is None:
        return
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Number of letters: {len(label_map)}")
    print(f"  Letters: {', '.join(label_map.values())}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print(f"TEST ACCURACY: {accuracy:.1%}")
    print(f"{'=' * 60}\n")
    
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "letter_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    labels_path = "letter_labels.json"
    with open(labels_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"✓ Labels saved to {labels_path}")
    
    print(f"\n{'=' * 60}")
    print("✅ Training complete!")
    print(f"\nNext step: ./run_app.sh")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train()
