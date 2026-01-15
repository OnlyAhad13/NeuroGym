"""Base Exercise class - Abstract base class for all exercises with state machine pattern."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time


class ExerciseStage(Enum):
    """Exercise state machine stages."""
    IDLE = "idle"       # Not detected/started
    UP = "up"           # Extended position (e.g., standing, arm down)
    DOWN = "down"       # Contracted position (e.g., squat, arm curled)
    HOLD = "hold"       # Static hold (e.g., plank)
    TRANSITION = "transition"  # Moving between stages


@dataclass
class ExerciseFeedback:
    """Container for exercise feedback."""
    message: str
    severity: str = "info"  # info, warning, error
    joint_name: Optional[str] = None
    current_angle: Optional[float] = None
    target_angle: Optional[float] = None


@dataclass
class JointAngle:
    """Represents a joint angle with metadata."""
    name: str
    angle: Optional[float]
    landmark_indices: Tuple[int, int, int]  # (start, vertex, end)
    target_range: Tuple[float, float] = (0.0, 180.0)
    
    @property
    def is_valid(self) -> bool:
        """Check if angle was successfully calculated."""
        return self.angle is not None
    
    @property
    def in_range(self) -> bool:
        """Check if angle is within target range."""
        if self.angle is None:
            return False
        return self.target_range[0] <= self.angle <= self.target_range[1]


class Exercise(ABC):
    """
    Abstract base class for exercise tracking with state machine.
    
    Implements:
    - State machine for rep counting
    - Form analysis and feedback
    - Joint angle tracking
    
    Subclasses must implement:
    - update(): Process landmarks and update state
    - analyze_form(): Return feedback on current form
    - get_joint_angles(): Return dict of tracked joint angles
    - _get_primary_angle(): Return main angle for state transitions
    """
    
    # Pose landmark indices (MediaPipe standard)
    # Left side
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    
    # Right side
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    
    def __init__(
        self,
        name: str,
        up_threshold: float,
        down_threshold: float,
        side: str = "both"
    ):
        """
        Initialize the exercise.
        
        Args:
            name: Display name of the exercise
            up_threshold: Angle threshold for UP stage (degrees)
            down_threshold: Angle threshold for DOWN stage (degrees)
            side: Which body side to track ("left", "right", or "both")
        """
        self.name = name
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.side = side
        
        # State machine
        self.current_stage = ExerciseStage.IDLE
        self.previous_stage = ExerciseStage.IDLE
        
        # Counters
        self.rep_count = 0
        self.hold_time = 0.0
        self._hold_start: Optional[float] = None
        
        # Current state
        self._joint_angles: Dict[str, JointAngle] = {}
        self._feedback: List[ExerciseFeedback] = []
        self._landmarks = None
        
        # Configuration
        self.min_visibility = 0.5
    
    @property
    def feedback(self) -> List[ExerciseFeedback]:
        """Get current feedback messages."""
        return self._feedback
    
    @property
    def primary_feedback(self) -> str:
        """Get the most important feedback message."""
        if self._feedback:
            return self._feedback[0].message
        return ""
    
    def reset(self) -> None:
        """Reset exercise counters and state."""
        self.current_stage = ExerciseStage.IDLE
        self.previous_stage = ExerciseStage.IDLE
        self.rep_count = 0
        self.hold_time = 0.0
        self._hold_start = None
        self._feedback = []
    
    def update(self, landmarks: List[Any]) -> None:
        """
        Process landmarks and update exercise state.
        
        Args:
            landmarks: List of pose landmarks from MediaPipe
        """
        self._landmarks = landmarks
        self._feedback = []
        
        if landmarks is None or len(landmarks) < 33:
            self.current_stage = ExerciseStage.IDLE
            return
        
        # Calculate joint angles
        self._calculate_angles()
        
        # Get primary angle for state machine
        primary_angle = self._get_primary_angle()
        
        if primary_angle is None:
            self.current_stage = ExerciseStage.IDLE
            return
        
        # Update state machine
        self._update_state_machine(primary_angle)
        
        # Analyze form and generate feedback
        self._feedback = self.analyze_form()
    
    def _update_state_machine(self, angle: float) -> None:
        """
        Update state machine based on primary angle.
        
        Override in subclass for custom state logic.
        """
        self.previous_stage = self.current_stage
        
        if self.current_stage == ExerciseStage.IDLE:
            # Start in UP position
            if angle > self.up_threshold:
                self.current_stage = ExerciseStage.UP
        
        elif self.current_stage == ExerciseStage.UP:
            # Transition to DOWN when angle decreases
            if angle < self.down_threshold:
                self.current_stage = ExerciseStage.DOWN
        
        elif self.current_stage == ExerciseStage.DOWN:
            # Complete rep when returning to UP
            if angle > self.up_threshold:
                self.current_stage = ExerciseStage.UP
                self.rep_count += 1
    
    @abstractmethod
    def _calculate_angles(self) -> None:
        """Calculate all tracked joint angles. Implemented by subclass."""
        pass
    
    @abstractmethod
    def _get_primary_angle(self) -> Optional[float]:
        """Return the primary angle used for state transitions."""
        pass
    
    @abstractmethod
    def analyze_form(self) -> List[ExerciseFeedback]:
        """
        Analyze current form and return feedback.
        
        Returns:
            List of ExerciseFeedback objects with form corrections
        """
        pass
    
    def get_joint_angles(self) -> Dict[str, JointAngle]:
        """Return all tracked joint angles."""
        return self._joint_angles
    
    def get_angle(self, name: str) -> Optional[float]:
        """Get a specific joint angle by name."""
        if name in self._joint_angles:
            return self._joint_angles[name].angle
        return None
    
    def _get_landmark(self, index: int) -> Optional[Any]:
        """Safely get a landmark by index."""
        if self._landmarks and 0 <= index < len(self._landmarks):
            lm = self._landmarks[index]
            if hasattr(lm, 'visibility') and lm.visibility >= self.min_visibility:
                return lm
            elif not hasattr(lm, 'visibility'):
                return lm
        return None
    
    def get_display_info(self) -> Dict[str, Any]:
        """
        Get info dict for UI display.
        
        Returns:
            Dict with rep count, stage, feedback, angles
        """
        return {
            "exercise": self.name,
            "reps": self.rep_count,
            "stage": self.current_stage.value,
            "feedback": self.primary_feedback,
            "hold_time": self.hold_time if self.current_stage == ExerciseStage.HOLD else None,
            "angles": {
                name: angle.angle 
                for name, angle in self._joint_angles.items()
                if angle.is_valid
            }
        }
