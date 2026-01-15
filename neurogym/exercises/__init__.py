"""Exercise module - Base class and implementations for exercise tracking."""

from .base import Exercise, ExerciseStage
from .squat import Squat
from .bicep_curl import BicepCurl
from .plank import Plank

__all__ = [
    "Exercise",
    "ExerciseStage",
    "Squat",
    "BicepCurl",
    "Plank",
]
