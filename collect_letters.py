"""
collect_letters.py

Data collection script for static ASL fingerspelling letters (A-Z).
Uses MediaPipe Hands to capture hand landmarks as single-frame snapshots.

Usage:
    python collect_letters.py
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import time


def normalize_hand_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extract and normalize hand landmarks to be position/scale invariant.
    
    Normalization:
    1. Shift all points relative to wrist (landmark 0)
    2. Scale by palm size (wrist to middle finger MCP distance)
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        Normalized array of shape (63,) — 21 landmarks × 3 coords
    """
    coords = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ])
    
    wrist = coords[0].copy()
    coords = coords - wrist
    
    palm_size = np.linalg.norm(coords[9])
    if palm_size > 1e-6:
        coords = coords / palm_size
    
    return coords.flatten().astype(np.float32)


def collect_letters():
    """Interactive ASL letter data collection."""
    
    LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
    
    OUTPUT_DIR = "letter_data"
    SAMPLES_PER_LETTER = 50
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 60)
    print("ASL LETTER DATA COLLECTION")
    print("=" * 60)
    print(f"\nLetters: {', '.join(LETTERS)}")
    print(f"Samples per letter: {SAMPLES_PER_LETTER}")
    print(f"\nControls:")
    print(f"  SPACE  = Capture a sample")
    print(f"  N      = Next letter")
    print(f"  P      = Previous letter")
    print(f"  Q      = Quit and save")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    current_letter_idx = 0
    collected_data = {}
    
    for letter in LETTERS:
        filepath = os.path.join(OUTPUT_DIR, f"{letter}.npy")
        if os.path.exists(filepath):
            existing = np.load(filepath)
            collected_data[letter] = list(existing)
            print(f"  Loaded {len(collected_data[letter])} existing samples for '{letter}'")
        else:
            collected_data[letter] = []
    
    while True:
        letter = LETTERS[current_letter_idx]
        count = len(collected_data[letter])
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, f"Letter: {letter}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        color = (0, 255, 0) if count >= SAMPLES_PER_LETTER else (0, 255, 255)
        cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_LETTER}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        total_done = sum(1 for l in LETTERS if len(collected_data[l]) >= SAMPLES_PER_LETTER)
        cv2.putText(frame, f"Letters done: {total_done}/{len(LETTERS)}", (w - 280, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"({current_letter_idx + 1}/{len(LETTERS)})", (w - 280, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        status = "Hand Detected" if hand_detected else "Show your hand!"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (w // 2 - 100, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(frame, "SPACE=Capture | N=Next | P=Prev | Q=Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("ASL Letter Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and hand_detected and results.multi_hand_landmarks:
            landmarks = normalize_hand_landmarks(results.multi_hand_landmarks[0])
            collected_data[letter].append(landmarks)
            count = len(collected_data[letter])
            print(f"  ✓ Captured sample {count} for '{letter}'")
            
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
            cv2.imshow("ASL Letter Collection", frame)
            cv2.waitKey(100)
            
            if count >= SAMPLES_PER_LETTER:
                print(f"  ✅ Letter '{letter}' complete!")
                if current_letter_idx < len(LETTERS) - 1:
                    current_letter_idx += 1
                    print(f"\n  → Moving to letter '{LETTERS[current_letter_idx]}'")
        
        elif key == ord('n'):
            if current_letter_idx < len(LETTERS) - 1:
                current_letter_idx += 1
                print(f"\n  → Letter: {LETTERS[current_letter_idx]}")
        
        elif key == ord('p'):
            if current_letter_idx > 0:
                current_letter_idx -= 1
                print(f"\n  → Letter: {LETTERS[current_letter_idx]}")
        
        elif key == ord('q'):
            break
    
    print("\n\nSaving data...")
    total_samples = 0
    for letter in LETTERS:
        if collected_data[letter]:
            data_array = np.array(collected_data[letter])
            filepath = os.path.join(OUTPUT_DIR, f"{letter}.npy")
            np.save(filepath, data_array)
            total_samples += len(collected_data[letter])
            print(f"  Saved {len(collected_data[letter]):3d} samples for '{letter}' → {filepath}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Collection complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Saved to: {OUTPUT_DIR}/")
    print(f"\nNext step: python train_letters.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    collect_letters()
