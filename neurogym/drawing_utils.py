"""CyberpunkDrawer - Custom drawing utilities with Sci-Fi/Cyberpunk aesthetic."""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class CyberpunkDrawer:
    """
    Custom drawing utilities that override MediaPipe's default visualization
    with a Cyberpunk/Sci-Fi aesthetic featuring neon colors and thin lines.
    
    Color Palette:
        - Primary (Pose): Cyan (#00FFFF)
        - Secondary (Hands): Magenta (#FF00FF) 
        - Accent (Face): Electric Blue (#00D4FF)
        - Highlight: Neon Green (#00FF00)
        - Gold: (#FFD700)
    """
    
    # Cyberpunk color palette (BGR format for OpenCV)
    CYAN = (255, 255, 0)           # #00FFFF - Primary pose color
    MAGENTA = (255, 0, 255)        # #FF00FF - Hand color
    ELECTRIC_BLUE = (255, 212, 0)  # #00D4FF - Face mesh color
    NEON_GREEN = (0, 255, 0)       # #00FF00 - Accent highlights
    GOLD = (0, 215, 255)           # #FFD700 - Special highlights
    DARK_CYAN = (139, 139, 0)      # Darker cyan for secondary connections
    HOT_PINK = (180, 105, 255)     # #FF69B4 - Alternative accent
    PURPLE = (128, 0, 128)         # #800080 - Deep accent
    
    # Line and point specifications
    POSE_LINE_THICKNESS = 1
    HAND_LINE_THICKNESS = 1
    FACE_LINE_THICKNESS = 1
    LANDMARK_RADIUS = 2
    GLOW_RADIUS = 4
    
    # Pose connections (hardcoded to avoid import issues)
    # Based on MediaPipe POSE_CONNECTIONS
    POSE_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ])
    
    # Hand connections (hardcoded)
    HAND_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),    # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)           # Palm
    ])
    
    # Pose connection groups for color coding
    POSE_LEFT_BODY = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    POSE_RIGHT_BODY = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    
    def __init__(
        self,
        draw_pose: bool = True,
        draw_hands: bool = True,
        draw_face: bool = True,
        glow_effect: bool = True,
        sparse_face: bool = True
    ):
        """
        Initialize the Cyberpunk drawer.
        
        Args:
            draw_pose: Enable pose skeleton drawing.
            draw_hands: Enable hand landmark drawing.
            draw_face: Enable face mesh drawing.
            glow_effect: Add subtle glow/bloom effect to landmarks.
            sparse_face: Draw only key face features (eyes, mouth) instead of full mesh.
        """
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands
        self.draw_face = draw_face
        self.glow_effect = glow_effect
        self.sparse_face = sparse_face
    
    def draw(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw all detected landmarks on the image with cyberpunk styling.
        
        Args:
            image: Input image in BGR format.
            results: MediaPipe Holistic results object.
            
        Returns:
            Image with cyberpunk-styled landmarks drawn.
        """
        # Create overlay for blending effects
        overlay = image.copy()
        
        # Draw face first (background layer)
        if self.draw_face and results.face_landmarks:
            self._draw_face_mesh(overlay, results.face_landmarks)
        
        # Draw pose
        if self.draw_pose and results.pose_landmarks:
            self._draw_pose(overlay, results.pose_landmarks)
        
        # Draw hands (foreground)
        if self.draw_hands:
            if results.left_hand_landmarks:
                self._draw_hand(overlay, results.left_hand_landmarks, is_left=True)
            if results.right_hand_landmarks:
                self._draw_hand(overlay, results.right_hand_landmarks, is_left=False)
        
        return overlay
    
    def _draw_pose(self, image: np.ndarray, landmarks) -> None:
        """Draw pose landmarks with cyberpunk styling."""
        h, w = image.shape[:2]
        
        # Get landmark positions
        points = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y, lm.visibility))
        
        # Draw connections with gradient colors
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx >= len(points) or end_idx >= len(points):
                continue
                
            start = points[start_idx]
            end = points[end_idx]
            
            # Skip if visibility is too low
            if start[2] < 0.5 or end[2] < 0.5:
                continue
            
            # Color based on body side
            if start_idx in self.POSE_LEFT_BODY or end_idx in self.POSE_LEFT_BODY:
                color = self.CYAN
            elif start_idx in self.POSE_RIGHT_BODY or end_idx in self.POSE_RIGHT_BODY:
                color = self.MAGENTA
            else:
                color = self.NEON_GREEN
            
            # Draw connection line
            cv2.line(
                image,
                (start[0], start[1]),
                (end[0], end[1]),
                color,
                self.POSE_LINE_THICKNESS,
                cv2.LINE_AA
            )
        
        # Draw landmarks with glow effect
        for i, (x, y, vis) in enumerate(points):
            if vis < 0.5:
                continue
                
            # Determine color based on position
            if i in self.POSE_LEFT_BODY:
                color = self.CYAN
            elif i in self.POSE_RIGHT_BODY:
                color = self.MAGENTA
            else:
                color = self.NEON_GREEN
            
            if self.glow_effect:
                # Outer glow
                cv2.circle(image, (x, y), self.GLOW_RADIUS, color, 1, cv2.LINE_AA)
            
            # Inner point
            cv2.circle(image, (x, y), self.LANDMARK_RADIUS, color, -1, cv2.LINE_AA)
    
    def _draw_hand(self, image: np.ndarray, landmarks, is_left: bool = True) -> None:
        """Draw hand landmarks with cyberpunk styling."""
        h, w = image.shape[:2]
        
        # Choose color based on hand
        color = self.CYAN if is_left else self.MAGENTA
        accent = self.GOLD
        
        # Get landmark positions
        points = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
        
        # Use hardcoded hand connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            
            start = points[start_idx]
            end = points[end_idx]
            
            # Draw connection
            cv2.line(
                image,
                start,
                end,
                color,
                self.HAND_LINE_THICKNESS,
                cv2.LINE_AA
            )
        
        # Draw fingertips with special highlighting
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        
        for i, (x, y) in enumerate(points):
            if i in fingertips:
                # Fingertips get gold highlight
                if self.glow_effect:
                    cv2.circle(image, (x, y), self.GLOW_RADIUS + 1, accent, 1, cv2.LINE_AA)
                cv2.circle(image, (x, y), self.LANDMARK_RADIUS + 1, accent, -1, cv2.LINE_AA)
            else:
                if self.glow_effect:
                    cv2.circle(image, (x, y), self.GLOW_RADIUS, color, 1, cv2.LINE_AA)
                cv2.circle(image, (x, y), self.LANDMARK_RADIUS, color, -1, cv2.LINE_AA)
    
    def _draw_face_mesh(self, image: np.ndarray, landmarks) -> None:
        """Draw face mesh with cyberpunk styling."""
        h, w = image.shape[:2]
        
        if self.sparse_face:
            # Only draw eyes and mouth contours for cleaner look
            self._draw_face_features(image, landmarks)
        else:
            # Draw all face points as dots (simplified mesh)
            for lm in landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, self.ELECTRIC_BLUE, -1, cv2.LINE_AA)
    
    def _draw_face_features(self, image: np.ndarray, landmarks) -> None:
        """Draw only key face features (eyes, mouth) for clean look."""
        h, w = image.shape[:2]
        
        # Eye contour indices
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Mouth outer contour
        mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        
        # Draw iris indicators (for fatigue tracking)
        left_iris = [468, 469, 470, 471, 472]
        right_iris = [473, 474, 475, 476, 477]
        
        # Get all points
        points = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
        
        # Draw eye contours
        self._draw_contour(image, points, left_eye, self.ELECTRIC_BLUE, closed=True)
        self._draw_contour(image, points, right_eye, self.ELECTRIC_BLUE, closed=True)
        
        # Draw iris if available (refined face landmarks)
        if len(points) > 477:
            self._draw_contour(image, points, left_iris, self.NEON_GREEN, closed=True)
            self._draw_contour(image, points, right_iris, self.NEON_GREEN, closed=True)
        
        # Draw mouth contour
        self._draw_contour(image, points, mouth_outer, self.HOT_PINK, closed=True)
    
    def _draw_contour(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        indices: List[int],
        color: Tuple[int, int, int],
        closed: bool = False
    ) -> None:
        """Draw a contour connecting specific landmark indices."""
        contour_points = []
        
        for idx in indices:
            if idx < len(points):
                contour_points.append(points[idx])
        
        if len(contour_points) < 2:
            return
        
        # Draw lines between consecutive points
        for i in range(len(contour_points) - 1):
            cv2.line(
                image,
                contour_points[i],
                contour_points[i + 1],
                color,
                self.FACE_LINE_THICKNESS,
                cv2.LINE_AA
            )
        
        # Close the contour if requested
        if closed and len(contour_points) > 2:
            cv2.line(
                image,
                contour_points[-1],
                contour_points[0],
                color,
                self.FACE_LINE_THICKNESS,
                cv2.LINE_AA
            )
    
    @staticmethod
    def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> None:
        """
        Draw FPS counter with cyberpunk styling.
        
        Args:
            image: Image to draw on.
            fps: Current FPS value.
            position: Position for the FPS text.
        """
        text = f"FPS: {fps:.1f}"
        
        # Draw shadow/outline
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        
        # Draw main text in neon green
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Neon green
            1,
            cv2.LINE_AA
        )
    
    @staticmethod
    def draw_status(
        image: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 60),
        color: Tuple[int, int, int] = (255, 255, 0)
    ) -> None:
        """Draw status text with cyberpunk styling."""
        # Shadow
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        
        # Main text
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    
    @staticmethod
    def draw_angle_arc(
        image: np.ndarray,
        center: Tuple[int, int],
        angle: float,
        target_angle: float = 90.0,
        max_angle: float = 180.0,
        radius: int = 35,
        start_angle: float = -90.0,
        inverted: bool = False
    ) -> None:
        """
        Draw a curved arc at a joint showing current angle progress.
        
        The arc fills based on how close the current angle is to the target.
        Color gradient: green (good) -> yellow -> pink (needs work)
        
        Args:
            image: Image to draw on
            center: Center point (x, y) of the arc (joint position)
            angle: Current angle in degrees
            target_angle: Target angle for exercise
            max_angle: Maximum possible angle
            radius: Radius of the arc in pixels
            start_angle: Starting angle for the arc (OpenCV convention)
            inverted: If True, lower angle = better (e.g., bicep curl)
        """
        if angle is None:
            return
        
        # Calculate progress (0 to 1)
        if inverted:
            # Lower angle is better (e.g., bicep curl at top)
            progress = 1.0 - (angle / max_angle)
            target_progress = 1.0 - (target_angle / max_angle)
        else:
            # Higher angle is better up to target
            progress = min(angle / target_angle, 1.0) if target_angle > 0 else 0
            target_progress = 1.0
        
        # Calculate how close to target (for color)
        if inverted:
            closeness = 1.0 - abs(angle - target_angle) / max_angle
        else:
            closeness = 1.0 - abs(angle - target_angle) / max_angle
        closeness = max(0.0, min(1.0, closeness))
        
        # Color gradient based on closeness to target
        # Green (good) -> Yellow -> Pink (needs work)
        if closeness > 0.7:
            color = (0, 255, 0)  # Neon green - good
        elif closeness > 0.4:
            color = (0, 255, 255)  # Yellow - getting there
        else:
            color = (180, 105, 255)  # Hot pink - needs work
        
        # Draw background arc (dark)
        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            start_angle,
            start_angle + 270,
            (40, 40, 40),
            2,
            cv2.LINE_AA
        )
        
        # Draw progress arc
        end_angle = start_angle + (progress * 270)
        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            color,
            3,
            cv2.LINE_AA
        )
        
        # Draw angle text
        text = f"{int(angle)}°"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + radius + 15
        
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )
    
    @staticmethod
    def draw_exercise_info(
        image: np.ndarray,
        exercise_name: str,
        rep_count: int,
        stage: str,
        feedback: str = "",
        hold_time: Optional[float] = None
    ) -> None:
        """
        Draw exercise information panel with cyberpunk styling.
        
        Args:
            image: Image to draw on
            exercise_name: Name of the current exercise
            rep_count: Current rep count
            stage: Current exercise stage
            feedback: Form feedback message
            hold_time: Hold time for plank-type exercises
        """
        h, w = image.shape[:2]
        
        # Panel position (bottom-left)
        panel_x = 10
        panel_y = h - 140
        panel_width = 280
        panel_height = 130
        
        # Draw semi-transparent panel background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (20, 20, 20),
            -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw border
        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (255, 255, 0),  # Cyan border
            1,
            cv2.LINE_AA
        )
        
        # Exercise name
        cv2.putText(
            image,
            exercise_name.upper(),
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Cyan
            2,
            cv2.LINE_AA
        )
        
        # Rep counter (large)
        rep_text = f"REPS: {rep_count}"
        cv2.putText(
            image,
            rep_text,
            (panel_x + 10, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),  # Neon green
            2,
            cv2.LINE_AA
        )
        
        # Stage indicator
        stage_color = (0, 255, 0) if stage.lower() in ["up", "hold"] else (0, 255, 255)
        cv2.putText(
            image,
            f"Stage: {stage.upper()}",
            (panel_x + 10, panel_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            stage_color,
            1,
            cv2.LINE_AA
        )
        
        # Hold time for plank
        if hold_time is not None:
            cv2.putText(
                image,
                f"Hold: {hold_time:.1f}s",
                (panel_x + 150, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 215, 255),  # Gold
                1,
                cv2.LINE_AA
            )
        
        # Feedback message
        if feedback:
            # Choose color based on feedback type
            if "great" in feedback.lower() or "good" in feedback.lower():
                fb_color = (0, 255, 0)  # Green
            else:
                fb_color = (180, 105, 255)  # Hot pink
            
            cv2.putText(
                image,
                feedback[:35],  # Truncate long messages
                (panel_x + 10, panel_y + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                fb_color,
                1,
                cv2.LINE_AA
            )
    
    @staticmethod
    def draw_rep_counter_large(
        image: np.ndarray,
        count: int,
        position: Tuple[int, int] = None
    ) -> None:
        """
        Draw a large, prominent rep counter.
        
        Args:
            image: Image to draw on
            count: Rep count to display
            position: Optional position, defaults to top-right
        """
        h, w = image.shape[:2]
        
        if position is None:
            position = (w - 120, 60)
        
        # Background circle
        cv2.circle(image, position, 50, (20, 20, 20), -1)
        cv2.circle(image, position, 50, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Count text
        text = str(count)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = position[0] - text_size[0] // 2
        text_y = position[1] + text_size[1] // 2
        
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )
        
        # "REPS" label below
        cv2.putText(
            image,
            "REPS",
            (position[0] - 25, position[1] + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )
