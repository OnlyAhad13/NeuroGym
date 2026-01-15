"""NeuroGym - AI-Powered Fitness & Rehab Coach Vision Pipeline."""

from .video_processor import VideoProcessor
from .holistic_detector import HolisticDetector
from .drawing_utils import CyberpunkDrawer

__all__ = ["VideoProcessor", "HolisticDetector", "CyberpunkDrawer"]
__version__ = "0.1.0"
