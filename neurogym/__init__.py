"""NeuroGym - AI-Powered Fitness & Rehab Coach Vision Pipeline."""

from .video_processor import VideoProcessor
from .holistic_detector import HolisticDetector
from .drawing_utils import CyberpunkDrawer
from .geometry_engine import GeometryEngine, calculate_angle, Point3D
from .exercises import Exercise, Squat, BicepCurl, Plank

__all__ = [
    "VideoProcessor",
    "HolisticDetector", 
    "CyberpunkDrawer",
    "GeometryEngine",
    "calculate_angle",
    "Point3D",
    "Exercise",
    "Squat",
    "BicepCurl",
    "Plank",
]
__version__ = "0.1.0"
