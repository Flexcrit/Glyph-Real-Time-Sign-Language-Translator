import cv2
from LandmarkExtractor import LandmarkExtractor
import numpy as np

def main():
    print("\n" + "="*60)
    print("ASL Camera & Landmark Detection Test")
    print("="*60)
    print("\nThis will test your camera and MediaPipe landmark detection.")
    print("Press 'q' to quit\n")
    
    print("Initializing MediaPipe Holistic...")
    extractor = LandmarkExtractor()
    print("✓ MediaPipe initialized successfully")
    
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access camera")
        print("\nTroubleshooting:")
        print("1. Check if another app is using the camera")
        print("2. Grant camera permissions in System Preferences")
        print("3. Try a different camera index (1 or 2)")
        return
    
    print("✓ Camera opened successfully")
    print(f"✓ Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("\n" + "="*60)
    print("Camera feed started - look for the window")
    print("Press 'q' to quit")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        landmarks, annotated_frame = extractor.process_frame_with_drawing(
            frame, thickness=1
        )
        
        frame_count += 1
        
        cv2.putText(
            annotated_frame,
            "ASL Landmark Detection Test",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        has_landmarks = not np.allclose(landmarks, 0)
        status = "Landmarks: DETECTED" if has_landmarks else "Landmarks: NONE"
        color = (0, 255, 0) if has_landmarks else (0, 0, 255)
        
        cv2.putText(
            annotated_frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Features: {landmarks.shape[0]}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            annotated_frame,
            f"Frames: {frame_count}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            annotated_frame,
            "Press 'q' to quit",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1
        )
        
        cv2.imshow('ASL Camera Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print(f"Total frames processed: {frame_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
