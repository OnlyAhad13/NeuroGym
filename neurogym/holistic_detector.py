"""HolisticDetector - MediaPipe Tasks API wrapper with configurable modules.

This uses the new MediaPipe Tasks API (0.10.31+) instead of the deprecated solutions API.
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Model URLs from MediaPipe model garden
MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


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


def download_model(model_name: str, model_dir: str = "models") -> str:
    """Download a MediaPipe model if not present."""
    model_path = Path(model_dir) / f"{model_name}_landmarker.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Downloading {model_name} model...")
        url = MODEL_URLS[model_name]
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to {model_path}")
    
    return str(model_path)


class HolisticDetector:
    """
    MediaPipe Tasks API wrapper with configurable module toggles.
    
    Uses the new Tasks Vision API (MediaPipe 0.10.31+) with separate
    PoseLandmarker, HandLandmarker, and FaceLandmarker.
    
    Example:
        detector = HolisticDetector(enable_face=False)  # Disable face for speed
        results = detector.process(frame_rgb)
        landmarks = detector.extract_landmarks()
    """
    
    # MediaPipe Face Mesh landmark indices for specific regions
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
        refine_face_landmarks: bool = True,
        model_dir: str = "models"
    ):
        """
        Initialize the Holistic detector with MediaPipe Tasks API.
        
        Args:
            enable_pose: Enable pose landmark detection (33 points).
            enable_hands: Enable hand landmark detection (21 points each).
            enable_face: Enable face mesh detection (478 points).
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            model_complexity: Not used in Tasks API (kept for API compatibility).
            refine_face_landmarks: Not used in Tasks API (kept for API compatibility).
            model_dir: Directory to store downloaded model files.
        """
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.model_dir = model_dir
        
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._face_landmarker = None
        
        # Store latest results
        self._pose_results = None
        self._hand_results = None
        self._face_results = None
        
        # Initialize enabled landmarkers
        if enable_pose:
            self._init_pose_landmarker(min_detection_confidence, min_tracking_confidence)
        if enable_hands:
            self._init_hand_landmarker(min_detection_confidence, min_tracking_confidence)
        if enable_face:
            self._init_face_landmarker(min_detection_confidence, min_tracking_confidence)
    
    def _init_pose_landmarker(self, min_detection: float, min_tracking: float):
        """Initialize the pose landmarker."""
        model_path = download_model("pose", self.model_dir)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            num_poses=1
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    
    def _init_hand_landmarker(self, min_detection: float, min_tracking: float):
        """Initialize the hand landmarker."""
        model_path = download_model("hand", self.model_dir)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            num_hands=2
        )
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
    
    def _init_face_landmarker(self, min_detection: float, min_tracking: float):
        """Initialize the face landmarker."""
        model_path = download_model("face", self.model_dir)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int = 0) -> Dict[str, Any]:
        """
        Process a frame and detect landmarks using all enabled landmarkers.
        
        Args:
            frame_rgb: Input frame in RGB format.
            timestamp_ms: Timestamp in milliseconds for video mode.
            
        Returns:
            Dictionary with pose, hand, and face results.
        """
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process with each enabled landmarker
        if self._pose_landmarker:
            self._pose_results = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if self._hand_landmarker:
            self._hand_results = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if self._face_landmarker:
            self._face_results = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        return {
            "pose": self._pose_results,
            "hands": self._hand_results,
            "face": self._face_results
        }
    
    def extract_landmarks(self) -> Dict[str, Any]:
        """
        Extract structured landmark data from the latest detection results.
        
        Returns:
            Dictionary containing:
                - pose_landmarks: List of 33 pose landmark points
                - left_hand_landmarks: List of 21 left hand landmark points
                - right_hand_landmarks: List of 21 right hand landmark points
                - face_landmarks: Dict with left_eye, right_eye, and mouth regions
        """
        extracted = {}
        
        # Extract pose landmarks (33 points)
        if self._pose_results and self._pose_results.pose_landmarks:
            landmarks = self._pose_results.pose_landmarks[0]  # First pose
            extracted["pose_landmarks"] = self._convert_landmarks(landmarks)
        else:
            extracted["pose_landmarks"] = None
        
        # Extract hand landmarks
        extracted["left_hand_landmarks"] = None
        extracted["right_hand_landmarks"] = None
        
        if self._hand_results and self._hand_results.hand_landmarks:
            for i, hand_landmarks in enumerate(self._hand_results.hand_landmarks):
                handedness = self._hand_results.handedness[i][0].category_name
                landmarks = self._convert_landmarks(hand_landmarks)
                
                # MediaPipe returns handedness from the camera's perspective (mirrored)
                if handedness == "Left":
                    extracted["right_hand_landmarks"] = landmarks
                else:
                    extracted["left_hand_landmarks"] = landmarks
        
        # Extract face landmarks
        if self._face_results and self._face_results.face_landmarks:
            face_lms = self._face_results.face_landmarks[0]
            all_landmarks = self._convert_landmarks(face_lms)
            
            extracted["face_landmarks"] = {
                "left_eye": [all_landmarks[i] for i in self.LEFT_EYE_INDICES if i < len(all_landmarks)],
                "right_eye": [all_landmarks[i] for i in self.RIGHT_EYE_INDICES if i < len(all_landmarks)],
                "mouth": [all_landmarks[i] for i in self.MOUTH_INDICES if i < len(all_landmarks)],
                "all": all_landmarks
            }
        else:
            extracted["face_landmarks"] = None
        
        return extracted
    
    def _convert_landmarks(self, landmarks) -> List[LandmarkPoint]:
        """Convert MediaPipe landmarks to list of LandmarkPoint objects."""
        points = []
        for lm in landmarks:
            visibility = getattr(lm, 'visibility', 1.0) if hasattr(lm, 'visibility') else 1.0
            points.append(LandmarkPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=visibility
            ))
        return points
    
    def get_raw_results(self) -> Dict[str, Any]:
        """Get the raw results from the last processing."""
        return {
            "pose": self._pose_results,
            "hands": self._hand_results,
            "face": self._face_results
        }
    
    # Properties for backward compatibility with old results structure
    @property
    def pose_landmarks(self):
        """Get pose landmarks in a format compatible with drawing utils."""
        if self._pose_results and self._pose_results.pose_landmarks:
            return _LandmarkWrapper(self._pose_results.pose_landmarks[0])
        return None
    
    @property
    def left_hand_landmarks(self):
        """Get left hand landmarks."""
        if self._hand_results and self._hand_results.hand_landmarks:
            for i, hand_landmarks in enumerate(self._hand_results.hand_landmarks):
                handedness = self._hand_results.handedness[i][0].category_name
                if handedness == "Right":  # Mirrored
                    return _LandmarkWrapper(hand_landmarks)
        return None
    
    @property
    def right_hand_landmarks(self):
        """Get right hand landmarks."""
        if self._hand_results and self._hand_results.hand_landmarks:
            for i, hand_landmarks in enumerate(self._hand_results.hand_landmarks):
                handedness = self._hand_results.handedness[i][0].category_name
                if handedness == "Left":  # Mirrored
                    return _LandmarkWrapper(hand_landmarks)
        return None
    
    @property
    def face_landmarks(self):
        """Get face landmarks."""
        if self._face_results and self._face_results.face_landmarks:
            return _LandmarkWrapper(self._face_results.face_landmarks[0])
        return None
    
    def close(self) -> None:
        """Release resources."""
        if self._pose_landmarker:
            self._pose_landmarker.close()
        if self._hand_landmarker:
            self._hand_landmarker.close()
        if self._face_landmarker:
            self._face_landmarker.close()
    
    def __enter__(self) -> "HolisticDetector":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.close()


class _LandmarkWrapper:
    """Wrapper to provide .landmark attribute for backward compatibility."""
    
    def __init__(self, landmarks):
        self.landmark = landmarks
