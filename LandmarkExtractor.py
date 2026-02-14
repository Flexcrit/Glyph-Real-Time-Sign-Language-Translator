"""
LandmarkExtractor.py

Handles MediaPipe Holistic landmark extraction and normalization for ASL recognition.
Extracts 1,662 features per frame with distance/position invariance.
"""

import cv2
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    """
    Extracts and normalizes landmarks from video frames using MediaPipe Holistic.
    
    Features extracted:
    - Left hand: 21 landmarks × 3 coords = 63 values
    - Right hand: 21 landmarks × 3 coords = 63 values
    - Pose: 33 landmarks × 3 coords = 99 values
    - Face: 468 landmarks × 3 coords = 1,404 values  
    - Pose visibility: 33 landmarks × 1 = 33 values
    Total: 1,662 features
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        """
        Initialize MediaPipe Holistic model.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
    
    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and normalize landmarks from an image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Normalized landmark array of shape (1662,)
            Returns zeros if no pose is detected
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if not results.pose_landmarks:
            return np.zeros(1662)
        
        pose_landmarks = self._extract_pose(results.pose_landmarks)
        face_landmarks = self._extract_face(results.face_landmarks)
        left_hand_landmarks = self._extract_hand(results.left_hand_landmarks)
        right_hand_landmarks = self._extract_hand(results.right_hand_landmarks)
        
        normalized_data = self._normalize_landmarks(
            pose_landmarks,
            face_landmarks,
            left_hand_landmarks,
            right_hand_landmarks
        )
        
        return normalized_data
    
    def _extract_pose(self, landmarks) -> np.ndarray:
        """Extract pose landmarks including visibility."""
        if landmarks is None:
            return np.zeros((33, 4))
        
        coords = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in landmarks.landmark
        ])
        return coords
    
    def _extract_face(self, landmarks) -> np.ndarray:
        """Extract face landmarks (468 points)."""
        if landmarks is None:
            return np.zeros((468, 3))
        
        coords = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])
        return coords
    
    def _extract_hand(self, landmarks) -> np.ndarray:
        """Extract hand landmarks (21 points)."""
        if landmarks is None:
            return np.zeros((21, 3))
        
        coords = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])
        return coords
    
    def _normalize_landmarks(
        self,
        pose: np.ndarray,
        face: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray
    ) -> np.ndarray:
        """
        Normalize landmarks to be invariant to distance and position.
        
        Normalization strategy:
        1. Shift all coordinates relative to Nose (pose landmark 0)
        2. Scale by shoulder distance (landmarks 11-12) for size invariance
        
        Args:
            pose: (33, 4) array with x, y, z, visibility
            face: (468, 3) array with x, y, z
            left_hand: (21, 3) array with x, y, z
            right_hand: (21, 3) array with x, y, z
            
        Returns:
            Flattened array of shape (1662,)
        """
        nose = pose[0, :3]
        
        left_shoulder = pose[11, :3]
        right_shoulder = pose[12, :3]
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_distance < 1e-6:
            shoulder_distance = 1.0
        
        pose_xyz = pose[:, :3]
        pose_xyz_normalized = (pose_xyz - nose) / shoulder_distance
        pose_visibility = pose[:, 3:4]
        
        face_normalized = (face - nose) / shoulder_distance
        
        left_hand_normalized = (left_hand - nose) / shoulder_distance
        right_hand_normalized = (right_hand - nose) / shoulder_distance
        
        flattened = np.concatenate([
            left_hand_normalized.flatten(),
            right_hand_normalized.flatten(),
            pose_xyz_normalized.flatten(),
            face_normalized.flatten(),
            pose_visibility.flatten()
        ])
        
        assert flattened.shape == (1662,), f"Expected shape (1662,), got {flattened.shape}"
        
        return flattened.astype(np.float32)
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        results,
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw landmarks on image with elegant, thin lines.
        
        Args:
            image: BGR image
            results: MediaPipe Holistic results
            thickness: Line thickness for drawing
            
        Returns:
            Image with drawn landmarks
        """
        
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(200, 0, 0), thickness=thickness, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=thickness
                )
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 200), thickness=thickness, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=thickness
                )
            )
        
        return image
    
    def process_frame_with_drawing(
        self,
        image: np.ndarray,
        thickness: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process frame and return both landmarks and annotated image.
        
        Args:
            image: BGR image
            thickness: Drawing thickness
            
        Returns:
            Tuple of (landmarks, annotated_image)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.holistic.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        annotated_image = self.draw_landmarks(image_bgr.copy(), results, thickness)
        
        if results.pose_landmarks:
            pose_landmarks = self._extract_pose(results.pose_landmarks)
            face_landmarks = self._extract_face(results.face_landmarks)
            left_hand_landmarks = self._extract_hand(results.left_hand_landmarks)
            right_hand_landmarks = self._extract_hand(results.right_hand_landmarks)
            
            landmarks = self._normalize_landmarks(
                pose_landmarks,
                face_landmarks,
                left_hand_landmarks,
                right_hand_landmarks
            )
        else:
            landmarks = np.zeros(1662)
        
        return landmarks, annotated_image
    
    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
