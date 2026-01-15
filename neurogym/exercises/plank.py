"""Plank Exercise - Tracks plank form with back alignment analysis."""

import time
from typing import List, Optional, Any
from .base import Exercise, ExerciseStage, ExerciseFeedback, JointAngle
from ..geometry_engine import calculate_angle


class Plank(Exercise):
    """
    Plank exercise tracking with form analysis.
    
    Tracks:
    - Back angle (primary) - Shoulder to Hip to Ankle alignment
    - Hold duration
    
    State machine:
    - IDLE: Not in plank position
    - HOLD: Maintaining plank (back angle ~180°)
    
    Form feedback:
    - "Lift your hips!" - Back angle < 170°
    - "Lower your hips!" - Back angle > 190°
    - "Great form!" - Back angle 175-185°
    """
    
    # Ideal plank alignment is a straight line (~180°)
    IDEAL_BACK_ANGLE = 180.0
    TOLERANCE = 10.0  # ±10° is acceptable
    
    def __init__(self, side: str = "right"):
        """
        Initialize Plank exercise.
        
        Args:
            side: Which side to track ("left", "right")
        """
        super().__init__(
            name="Plank",
            up_threshold=170.0,   # Minimum angle for valid plank
            down_threshold=190.0, # Maximum angle for valid plank
            side=side
        )
        
        # Hold tracking
        self._hold_start_time: Optional[float] = None
        self._is_in_plank = False
        self.total_hold_time = 0.0
    
    def _update_state_machine(self, angle: float) -> None:
        """
        Override state machine for plank (hold-based).
        
        Plank uses HOLD state instead of UP/DOWN transitions.
        """
        self.previous_stage = self.current_stage
        
        # Check if in valid plank position
        in_position = self.up_threshold <= angle <= self.down_threshold
        
        if in_position:
            if self.current_stage != ExerciseStage.HOLD:
                # Entering plank
                self.current_stage = ExerciseStage.HOLD
                self._hold_start_time = time.time()
                self._is_in_plank = True
            else:
                # Continue holding
                if self._hold_start_time:
                    self.hold_time = time.time() - self._hold_start_time
        else:
            if self.current_stage == ExerciseStage.HOLD:
                # Exiting plank
                if self._hold_start_time:
                    self.total_hold_time += time.time() - self._hold_start_time
                    # Count as a rep if held for at least 5 seconds
                    if self.hold_time >= 5.0:
                        self.rep_count += 1
                self._hold_start_time = None
                self.hold_time = 0.0
                self._is_in_plank = False
            
            self.current_stage = ExerciseStage.IDLE
    
    def _calculate_angles(self) -> None:
        """Calculate back alignment angle."""
        self._joint_angles = {}
        
        if self.side == "right":
            shoulder = self._get_landmark(self.RIGHT_SHOULDER)
            hip = self._get_landmark(self.RIGHT_HIP)
            ankle = self._get_landmark(self.RIGHT_ANKLE)
        else:
            shoulder = self._get_landmark(self.LEFT_SHOULDER)
            hip = self._get_landmark(self.LEFT_HIP)
            ankle = self._get_landmark(self.LEFT_ANKLE)
        
        if shoulder and hip and ankle:
            angle = calculate_angle(shoulder, hip, ankle)
            self._joint_angles["back"] = JointAngle(
                name="Back Alignment",
                angle=angle,
                landmark_indices=(
                    self.RIGHT_SHOULDER if self.side == "right" else self.LEFT_SHOULDER,
                    self.RIGHT_HIP if self.side == "right" else self.LEFT_HIP,
                    self.RIGHT_ANKLE if self.side == "right" else self.LEFT_ANKLE
                ),
                target_range=(175.0, 185.0)
            )
    
    def _get_primary_angle(self) -> Optional[float]:
        """Return back angle for state transitions."""
        if "back" in self._joint_angles:
            return self._joint_angles["back"].angle
        return None
    
    def analyze_form(self) -> List[ExerciseFeedback]:
        """Analyze plank form and return feedback."""
        feedback = []
        
        back_angle = self._get_primary_angle()
        
        if back_angle is None:
            return feedback
        
        if self.current_stage == ExerciseStage.HOLD:
            deviation = back_angle - self.IDEAL_BACK_ANGLE
            
            if deviation < -self.TOLERANCE:
                # Hips too low (back sagging)
                feedback.append(ExerciseFeedback(
                    message="Lift your hips!",
                    severity="warning",
                    joint_name="back",
                    current_angle=back_angle,
                    target_angle=self.IDEAL_BACK_ANGLE
                ))
            elif deviation > self.TOLERANCE:
                # Hips too high (pike position)
                feedback.append(ExerciseFeedback(
                    message="Lower your hips!",
                    severity="warning",
                    joint_name="back",
                    current_angle=back_angle,
                    target_angle=self.IDEAL_BACK_ANGLE
                ))
            else:
                # Good form
                feedback.append(ExerciseFeedback(
                    message=f"Great form! Hold: {self.hold_time:.1f}s",
                    severity="info",
                    joint_name="back",
                    current_angle=back_angle
                ))
        elif self.current_stage == ExerciseStage.IDLE:
            if back_angle < self.up_threshold:
                feedback.append(ExerciseFeedback(
                    message="Straighten your back!",
                    severity="warning",
                    joint_name="back",
                    current_angle=back_angle,
                    target_angle=self.IDEAL_BACK_ANGLE
                ))
        
        return feedback
    
    def get_angle_arc_config(self) -> dict:
        """Get configuration for rendering angle arcs."""
        if "back" in self._joint_angles:
            joint = self._joint_angles["back"]
            return {
                "joint": "back",
                "vertex_index": joint.landmark_indices[1],  # Hip
                "current_angle": joint.angle,
                "target_angle": self.IDEAL_BACK_ANGLE,
                "max_angle": 200.0,
                "show_deviation": True
            }
        return {}
    
    def get_display_info(self) -> dict:
        """Override to include hold time in display."""
        info = super().get_display_info()
        info["hold_time"] = self.hold_time
        info["total_hold_time"] = self.total_hold_time
        return info
