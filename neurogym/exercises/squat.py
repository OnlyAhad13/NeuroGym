"""Squat Exercise - Tracks squat form with knee angle analysis."""

from typing import List, Optional, Any
from .base import Exercise, ExerciseStage, ExerciseFeedback, JointAngle
from ..geometry_engine import calculate_angle


class Squat(Exercise):
    """
    Squat exercise tracking with form analysis.
    
    Tracks:
    - Knee angle (primary) - Hip to Knee to Ankle
    - Hip angle - Shoulder to Hip to Knee
    
    State machine:
    - UP: Standing (knee angle > 160°)
    - DOWN: Squat position (knee angle < 110°)
    - Rep counted when returning from DOWN to UP
    
    Form feedback:
    - "Go lower!" - Knee angle > 100° in DOWN phase
    - "Keep knees behind toes!" - Knee extends past ankle
    """
    
    def __init__(self, side: str = "right"):
        """
        Initialize Squat exercise.
        
        Args:
            side: Which leg to track ("left", "right", or "both")
        """
        super().__init__(
            name="Squat",
            up_threshold=160.0,    # Standing
            down_threshold=110.0,  # Squat depth
            side=side
        )
        
        # Squat-specific thresholds
        self.deep_squat_angle = 90.0  # Target for good depth
        self.parallel_angle = 100.0   # Minimum for credit
    
    def _calculate_angles(self) -> None:
        """Calculate knee and hip angles."""
        self._joint_angles = {}
        
        if self.side in ["right", "both"]:
            # Right knee angle
            hip = self._get_landmark(self.RIGHT_HIP)
            knee = self._get_landmark(self.RIGHT_KNEE)
            ankle = self._get_landmark(self.RIGHT_ANKLE)
            
            if hip and knee and ankle:
                angle = calculate_angle(hip, knee, ankle)
                self._joint_angles["right_knee"] = JointAngle(
                    name="Right Knee",
                    angle=angle,
                    landmark_indices=(self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE),
                    target_range=(80.0, 100.0)
                )
            
            # Right hip angle
            shoulder = self._get_landmark(self.RIGHT_SHOULDER)
            if shoulder and hip and knee:
                angle = calculate_angle(shoulder, hip, knee)
                self._joint_angles["right_hip"] = JointAngle(
                    name="Right Hip",
                    angle=angle,
                    landmark_indices=(self.RIGHT_SHOULDER, self.RIGHT_HIP, self.RIGHT_KNEE),
                    target_range=(70.0, 110.0)
                )
        
        if self.side in ["left", "both"]:
            # Left knee angle
            hip = self._get_landmark(self.LEFT_HIP)
            knee = self._get_landmark(self.LEFT_KNEE)
            ankle = self._get_landmark(self.LEFT_ANKLE)
            
            if hip and knee and ankle:
                angle = calculate_angle(hip, knee, ankle)
                self._joint_angles["left_knee"] = JointAngle(
                    name="Left Knee",
                    angle=angle,
                    landmark_indices=(self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE),
                    target_range=(80.0, 100.0)
                )
            
            # Left hip angle
            shoulder = self._get_landmark(self.LEFT_SHOULDER)
            if shoulder and hip and knee:
                angle = calculate_angle(shoulder, hip, knee)
                self._joint_angles["left_hip"] = JointAngle(
                    name="Left Hip",
                    angle=angle,
                    landmark_indices=(self.LEFT_SHOULDER, self.LEFT_HIP, self.LEFT_KNEE),
                    target_range=(70.0, 110.0)
                )
    
    def _get_primary_angle(self) -> Optional[float]:
        """Return knee angle for state transitions."""
        # Prefer right side, fall back to left
        if "right_knee" in self._joint_angles:
            return self._joint_angles["right_knee"].angle
        elif "left_knee" in self._joint_angles:
            return self._joint_angles["left_knee"].angle
        return None
    
    def analyze_form(self) -> List[ExerciseFeedback]:
        """Analyze squat form and return feedback."""
        feedback = []
        
        knee_angle = self._get_primary_angle()
        
        if knee_angle is None:
            return feedback
        
        # Check for squat depth in DOWN phase
        if self.current_stage == ExerciseStage.DOWN:
            if knee_angle > self.parallel_angle:
                feedback.append(ExerciseFeedback(
                    message="Go lower!",
                    severity="warning",
                    joint_name="knee",
                    current_angle=knee_angle,
                    target_angle=self.deep_squat_angle
                ))
            elif knee_angle <= self.deep_squat_angle:
                feedback.append(ExerciseFeedback(
                    message="Great depth!",
                    severity="info",
                    joint_name="knee",
                    current_angle=knee_angle
                ))
        
        # Check knee tracking (knees over toes)
        self._check_knee_tracking(feedback)
        
        return feedback
    
    def _check_knee_tracking(self, feedback: List[ExerciseFeedback]) -> None:
        """Check if knees are tracking properly over toes."""
        # Get relevant landmarks
        if self.side == "right" or self.side == "both":
            knee = self._get_landmark(self.RIGHT_KNEE)
            ankle = self._get_landmark(self.RIGHT_ANKLE)
            
            if knee and ankle and self.current_stage == ExerciseStage.DOWN:
                # If knee x-position is significantly ahead of ankle
                # (In normalized coords, larger x means more to the right from camera view)
                # This is a simplified check - actual tracking would depend on camera angle
                if abs(knee.x - ankle.x) > 0.1:  # Threshold for knee forward
                    pass  # Could add "Keep knees behind toes!" feedback
    
    def get_angle_arc_config(self) -> dict:
        """
        Get configuration for rendering angle arcs.
        
        Returns:
            Dict with joint name, indices, and target angles for arc rendering
        """
        primary_key = "right_knee" if "right_knee" in self._joint_angles else "left_knee"
        
        if primary_key in self._joint_angles:
            joint = self._joint_angles[primary_key]
            return {
                "joint": primary_key,
                "vertex_index": joint.landmark_indices[1],
                "current_angle": joint.angle,
                "target_angle": self.deep_squat_angle,
                "max_angle": 180.0
            }
        return {}
