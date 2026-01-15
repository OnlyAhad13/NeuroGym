"""BicepCurl Exercise - Tracks bicep curl form with elbow angle analysis."""

from typing import List, Optional, Any
from .base import Exercise, ExerciseStage, ExerciseFeedback, JointAngle
from ..geometry_engine import calculate_angle


class BicepCurl(Exercise):
    """
    Bicep curl exercise tracking with form analysis.
    
    Tracks:
    - Elbow angle (primary) - Shoulder to Elbow to Wrist
    - Shoulder stability - Checks for upper arm movement
    
    State machine:
    - DOWN: Arm extended (elbow angle > 160°)
    - UP: Arm curled (elbow angle < 40°)
    - Rep counted when returning from UP to DOWN
    
    Form feedback:
    - "Squeeze at the top!" - Elbow angle > 30° at top
    - "Full extension!" - Elbow angle < 150° at bottom
    - "Keep upper arm still!" - Excessive shoulder movement
    """
    
    def __init__(self, side: str = "right"):
        """
        Initialize BicepCurl exercise.
        
        Args:
            side: Which arm to track ("left", "right", or "both")
        """
        super().__init__(
            name="Bicep Curl",
            up_threshold=160.0,   # Extended arm
            down_threshold=50.0,  # Curled arm (note: reversed from squat)
            side=side
        )
        
        # Curl-specific thresholds
        self.full_contraction = 35.0   # Target for top of curl
        self.full_extension = 160.0    # Target for bottom
        
        # Track shoulder position for stability check
        self._prev_shoulder_y: Optional[float] = None
        self._shoulder_movement = 0.0
    
    def _update_state_machine(self, angle: float) -> None:
        """
        Override state machine for bicep curl (angle decreases for rep).
        
        Bicep curl is opposite of squat - we start with high angle (extended)
        and decrease for the concentric phase.
        """
        self.previous_stage = self.current_stage
        
        if self.current_stage == ExerciseStage.IDLE:
            # Start in DOWN position (arm extended)
            if angle > self.up_threshold:
                self.current_stage = ExerciseStage.DOWN  # Extended = DOWN for curl
        
        elif self.current_stage == ExerciseStage.DOWN:
            # Transition to UP when curling (angle decreases)
            if angle < self.down_threshold:
                self.current_stage = ExerciseStage.UP
        
        elif self.current_stage == ExerciseStage.UP:
            # Complete rep when extending back
            if angle > self.up_threshold:
                self.current_stage = ExerciseStage.DOWN
                self.rep_count += 1
    
    def _calculate_angles(self) -> None:
        """Calculate elbow angle."""
        self._joint_angles = {}
        
        if self.side in ["right", "both"]:
            shoulder = self._get_landmark(self.RIGHT_SHOULDER)
            elbow = self._get_landmark(self.RIGHT_ELBOW)
            wrist = self._get_landmark(self.RIGHT_WRIST)
            
            if shoulder and elbow and wrist:
                angle = calculate_angle(shoulder, elbow, wrist)
                self._joint_angles["right_elbow"] = JointAngle(
                    name="Right Elbow",
                    angle=angle,
                    landmark_indices=(self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST),
                    target_range=(30.0, 40.0)  # Target at top of curl
                )
                
                # Track shoulder movement
                self._update_shoulder_tracking(shoulder)
        
        if self.side in ["left", "both"]:
            shoulder = self._get_landmark(self.LEFT_SHOULDER)
            elbow = self._get_landmark(self.LEFT_ELBOW)
            wrist = self._get_landmark(self.LEFT_WRIST)
            
            if shoulder and elbow and wrist:
                angle = calculate_angle(shoulder, elbow, wrist)
                self._joint_angles["left_elbow"] = JointAngle(
                    name="Left Elbow",
                    angle=angle,
                    landmark_indices=(self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST),
                    target_range=(30.0, 40.0)
                )
    
    def _update_shoulder_tracking(self, shoulder) -> None:
        """Track shoulder movement for stability feedback."""
        if self._prev_shoulder_y is not None:
            self._shoulder_movement = abs(shoulder.y - self._prev_shoulder_y)
        self._prev_shoulder_y = shoulder.y
    
    def _get_primary_angle(self) -> Optional[float]:
        """Return elbow angle for state transitions."""
        if "right_elbow" in self._joint_angles:
            return self._joint_angles["right_elbow"].angle
        elif "left_elbow" in self._joint_angles:
            return self._joint_angles["left_elbow"].angle
        return None
    
    def analyze_form(self) -> List[ExerciseFeedback]:
        """Analyze bicep curl form and return feedback."""
        feedback = []
        
        elbow_angle = self._get_primary_angle()
        
        if elbow_angle is None:
            return feedback
        
        # Check for full contraction at top
        if self.current_stage == ExerciseStage.UP:
            if elbow_angle > self.full_contraction + 10:
                feedback.append(ExerciseFeedback(
                    message="Squeeze at the top!",
                    severity="warning",
                    joint_name="elbow",
                    current_angle=elbow_angle,
                    target_angle=self.full_contraction
                ))
            else:
                feedback.append(ExerciseFeedback(
                    message="Good squeeze!",
                    severity="info",
                    joint_name="elbow",
                    current_angle=elbow_angle
                ))
        
        # Check for full extension at bottom
        if self.current_stage == ExerciseStage.DOWN:
            if elbow_angle < self.full_extension - 10:
                feedback.append(ExerciseFeedback(
                    message="Full extension!",
                    severity="info",
                    joint_name="elbow",
                    current_angle=elbow_angle,
                    target_angle=self.full_extension
                ))
        
        # Check shoulder stability
        if self._shoulder_movement > 0.02:  # Threshold for excessive movement
            feedback.append(ExerciseFeedback(
                message="Keep upper arm still!",
                severity="warning",
                joint_name="shoulder"
            ))
        
        return feedback
    
    def get_angle_arc_config(self) -> dict:
        """Get configuration for rendering angle arcs."""
        primary_key = "right_elbow" if "right_elbow" in self._joint_angles else "left_elbow"
        
        if primary_key in self._joint_angles:
            joint = self._joint_angles[primary_key]
            return {
                "joint": primary_key,
                "vertex_index": joint.landmark_indices[1],
                "current_angle": joint.angle,
                "target_angle": self.full_contraction,
                "max_angle": 180.0,
                "inverted": True  # Lower angle = better for curl
            }
        return {}
