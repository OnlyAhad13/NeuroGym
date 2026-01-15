"""HolisticDetector - MediaPipe Holistic solution wrapper with configurable modules."""

import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LandmarkPoint:
    """Represents a single landmark point with coordinates and visibility."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility
        }
    
    def to_pixel(self, width: int, height: int) -> tuple:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


class HolisticDetector:
    """
    MediaPipe Holistic solution wrapper with configurable module toggles.
    
    Supports selective enabling of pose, hand, and face detection for
    FPS optimization based on use case requirements.
    
    Example:
        detector = HolisticDetector(enable_face=False)  # Disable face for speed
        results = detector.process(frame_rgb)
        landmarks = detector.extract_landmarks(results)
    """
    
    # MediaPipe Face Mesh landmark indices for specific regions
    # These are subsets for fatigue detection (eye tracking, mouth movement)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    
    def __init__(
        self,
        enable_pose: bool = True,
        enable_hands: bool = True,
        enable_face: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        refine_face_landmarks: bool = True
    ):
        """
        Initialize the Holistic detector.
        
        Args:
            enable_pose: Enable pose landmark detection (33 points).
            enable_hands: Enable hand landmark detection (21 points each).
            enable_face: Enable face mesh detection (468 points).
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            model_complexity: Model complexity (0, 1, or 2). Higher = more accurate but slower.
            refine_face_landmarks: If True, enables attention on face for better iris tracking.
        """
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        
        self._mp_holistic = mp.solutions.holistic
        
        # Initialize the holistic model
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._results = None
        
    def process(self, frame_rgb: np.ndarray) -> Any:
        """
        Process a frame and detect landmarks.
        
        Args:
            frame_rgb: Input frame in RGB format.
            
        Returns:
            MediaPipe Holistic results object.
        """
        # Mark image as not writeable for performance
        frame_rgb.flags.writeable = False
        self._results = self._holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        return self._results
    
    def extract_landmarks(self, results: Optional[Any] = None) -> Dict[str, Any]:
        """
        Extract structured landmark data from detection results.
        
        Args:
            results: MediaPipe results object. Uses last processed if None.
            
        Returns:
            Dictionary containing:
                - pose_landmarks: List of 33 pose landmark points
                - left_hand_landmarks: List of 21 left hand landmark points
                - right_hand_landmarks: List of 21 right hand landmark points
                - face_landmarks: Dict with left_eye, right_eye, and mouth regions
        """
        if results is None:
            results = self._results
            
        if results is None:
            return self._empty_landmarks()
        
        extracted = {}
        
        # Extract pose landmarks (33 points)
        if self.enable_pose and results.pose_landmarks:
            extracted["pose_landmarks"] = self._extract_landmark_list(
                results.pose_landmarks.landmark
            )
        else:
            extracted["pose_landmarks"] = None
            
        # Extract left hand landmarks (21 points)
        if self.enable_hands and results.left_hand_landmarks:
            extracted["left_hand_landmarks"] = self._extract_landmark_list(
                results.left_hand_landmarks.landmark
            )
        else:
            extracted["left_hand_landmarks"] = None
            
        # Extract right hand landmarks (21 points)
        if self.enable_hands and results.right_hand_landmarks:
            extracted["right_hand_landmarks"] = self._extract_landmark_list(
                results.right_hand_landmarks.landmark
            )
        else:
            extracted["right_hand_landmarks"] = None
            
        # Extract face landmarks (subset for fatigue detection)
        if self.enable_face and results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            extracted["face_landmarks"] = {
                "left_eye": self._extract_indexed_landmarks(face_lms, self.LEFT_EYE_INDICES),
                "right_eye": self._extract_indexed_landmarks(face_lms, self.RIGHT_EYE_INDICES),
                "mouth": self._extract_indexed_landmarks(face_lms, self.MOUTH_INDICES),
                "all": self._extract_landmark_list(face_lms)  # Full mesh if needed
            }
        else:
            extracted["face_landmarks"] = None
            
        return extracted
    
    def _extract_landmark_list(self, landmarks) -> List[LandmarkPoint]:
        """Convert MediaPipe landmarks to list of LandmarkPoint objects."""
        points = []
        for lm in landmarks:
            visibility = getattr(lm, 'visibility', 1.0)
            points.append(LandmarkPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=visibility
            ))
        return points
    
    def _extract_indexed_landmarks(self, landmarks, indices: List[int]) -> List[LandmarkPoint]:
        """Extract specific landmarks by their indices."""
        points = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                visibility = getattr(lm, 'visibility', 1.0)
                points.append(LandmarkPoint(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=visibility
                ))
        return points
    
    def _empty_landmarks(self) -> Dict[str, Any]:
        """Return empty landmark structure."""
        return {
            "pose_landmarks": None,
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
            "face_landmarks": None
        }
    
    def get_raw_results(self) -> Optional[Any]:
        """Get the raw MediaPipe results from last processing."""
        return self._results
    
    def close(self) -> None:
        """Release resources."""
        self._holistic.close()
        
    def __enter__(self) -> "HolisticDetector":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.close()
