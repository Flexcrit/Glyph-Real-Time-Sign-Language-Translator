"""
collect_data.py

Data collection script for ASL signs.
Collects 30-frame sequences with MediaPipe landmarks for model training.
"""

import cv2
import numpy as np
import os
import json
from LandmarkExtractor import LandmarkExtractor
import time


def collect_data_for_sign(
    sign_label: str,
    sign_index: int,
    num_sequences: int = 30,
    sequence_length: int = 30,
    output_dir: str = 'training_data'
):
    """
    Collect training data for a specific sign.
    
    Args:
        sign_label: Human-readable label (e.g., "hello")
        sign_index: Numeric index for the sign
        num_sequences: Number of sequences to collect
        sequence_length: Frames per sequence
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    sign_dir = os.path.join(output_dir, f"{sign_index:04d}_{sign_label}")
    os.makedirs(sign_dir, exist_ok=True)
    
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    print(f"\n{'='*60}")
    print(f"Collecting data for: {sign_label.upper()}")
    print(f"Sign Index: {sign_index}")
    print(f"Sequences to collect: {num_sequences}")
    print(f"Frames per sequence: {sequence_length}")
    print(f"{'='*60}\n")
    
    sequences_collected = 0
    
    while sequences_collected < num_sequences:
        print(f"\nSequence {sequences_collected + 1}/{num_sequences}")
        print("Press SPACE when ready to record...")
        
        ready = False
        while not ready:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            cv2.putText(
                frame,
                f"Sign: {sign_label.upper()}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
            cv2.putText(
                frame,
                f"Sequence: {sequences_collected + 1}/{num_sequences}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "Press SPACE to start recording",
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                ready = True
            elif key == ord('q'):
                print("\nCollection cancelled by user")
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                return
        
        print("Recording...")
        sequence = []
        
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    str(countdown),
                    (frame.shape[1]//2 - 50, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 255),
                    5
                )
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
        
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                continue
            
            frame = cv2.flip(frame, 1)
            
            landmarks, annotated_frame = extractor.process_frame_with_drawing(
                frame, thickness=1
            )
            
            sequence.append(landmarks)
            
            cv2.putText(
                annotated_frame,
                f"Recording: {frame_num + 1}/{sequence_length}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            bar_width = 400
            progress = int((frame_num + 1) / sequence_length * bar_width)
            cv2.rectangle(
                annotated_frame,
                (20, 80),
                (20 + bar_width, 100),
                (100, 100, 100),
                2
            )
            cv2.rectangle(
                annotated_frame,
                (20, 80),
                (20 + progress, 100),
                (0, 255, 0),
                -1
            )
            
            cv2.imshow('Data Collection', annotated_frame)
            cv2.waitKey(1)
        
        if len(sequence) == sequence_length:
            sequence_array = np.array(sequence)
            filename = os.path.join(sign_dir, f"sequence_{sequences_collected:04d}.npy")
            np.save(filename, sequence_array)
            sequences_collected += 1
            print(f"✓ Saved sequence {sequences_collected}/{num_sequences}")
        else:
            print(f"✗ Incomplete sequence, skipping...")
        
        time.sleep(0.5)
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    
    print(f"\n{'='*60}")
    print(f"✓ Data collection complete for '{sign_label}'")
    print(f"Sequences collected: {sequences_collected}")
    print(f"Saved to: {sign_dir}")
    print(f"{'='*60}\n")


def load_collected_data(data_dir: str = 'training_data'):
    """
    Load all collected sequences from directory.
    
    Args:
        data_dir: Directory containing collected data
        
    Returns:
        Tuple of (X, y, labels_dict)
        X: numpy array of shape (num_samples, 30, 1662)
        y: numpy array of labels
        labels_dict: dictionary mapping indices to sign names
    """
    sequences = []
    labels = []
    labels_dict = {}
    
    for sign_dir in sorted(os.listdir(data_dir)):
        sign_path = os.path.join(data_dir, sign_dir)
        
        if not os.path.isdir(sign_path):
            continue
        
        parts = sign_dir.split('_', 1)
        if len(parts) != 2:
            continue
        
        sign_index = int(parts[0])
        sign_label = parts[1]
        labels_dict[sign_index] = sign_label
        
        print(f"Loading data for '{sign_label}' (index {sign_index})...")
        
        for filename in sorted(os.listdir(sign_path)):
            if not filename.endswith('.npy'):
                continue
            
            filepath = os.path.join(sign_path, filename)
            sequence = np.load(filepath)
            
            sequences.append(sequence)
            labels.append(sign_index)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"\n✓ Loaded {len(X)} sequences for {len(labels_dict)} signs")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, labels_dict


def main():
    """Interactive data collection interface."""
    print("\n" + "="*60)
    print("ASL Data Collection Tool")
    print("="*60)
    
    sign_label = input("\nEnter sign label (e.g., 'hello'): ").strip().lower()
    sign_index = int(input("Enter sign index (e.g., 0): "))
    num_sequences = int(input("Number of sequences to collect (default 30): ") or "30")
    
    collect_data_for_sign(
        sign_label=sign_label,
        sign_index=sign_index,
        num_sequences=num_sequences
    )
    
    print("\nDone! You can:")
    print("1. Run this script again to collect more signs")
    print("2. Use load_collected_data() to load all data for training")
    print("3. Save labels_dict to labels.json for use in the app")


if __name__ == "__main__":
    main()
