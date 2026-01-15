"""GeometryEngine - Math utilities for 3D angle and distance calculations.

Provides robust functions for calculating joint angles from body landmarks
with proper handling of edge cases and low visibility.
"""

import math
import numpy as np
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class Point3D:
    """Represents a 3D point with optional visibility."""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
    @classmethod
    def from_landmark(cls, landmark: Any) -> "Point3D":
        """Create Point3D from MediaPipe landmark or LandmarkPoint."""
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, 'visibility', 1.0)
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


def calculate_angle(
    a: Union[Point3D, Any],
    b: Union[Point3D, Any], 
    c: Union[Point3D, Any],
    min_visibility: float = 0.5
) -> Optional[float]:
    """
    Calculate the angle ABC (angle at vertex B) from three 3D points.
    
    Uses the dot product formula:
        angle = arccos((BA · BC) / (|BA| * |BC|))
    
    Args:
        a: First point (one end of the angle)
        b: Vertex point (where the angle is measured)
        c: Third point (other end of the angle)
        min_visibility: Minimum visibility threshold. Returns None if any point
                       has visibility below this threshold.
    
    Returns:
        Angle in degrees (0-180), or None if visibility is insufficient
        or vectors are degenerate.
    
    Example:
        >>> p1 = Point3D(0, 0, 0)
        >>> p2 = Point3D(1, 0, 0)
        >>> p3 = Point3D(1, 1, 0)
        >>> calculate_angle(p1, p2, p3)
        90.0
    """
    # Convert to Point3D if needed
    if not isinstance(a, Point3D):
        a = Point3D.from_landmark(a)
    if not isinstance(b, Point3D):
        b = Point3D.from_landmark(b)
    if not isinstance(c, Point3D):
        c = Point3D.from_landmark(c)
    
    # Check visibility
    if a.visibility < min_visibility or \
       b.visibility < min_visibility or \
       c.visibility < min_visibility:
        return None
    
    # Convert to numpy arrays
    point_a = a.to_array()
    point_b = b.to_array()
    point_c = c.to_array()
    
    # Calculate vectors from vertex B
    ba = point_a - point_b
    bc = point_c - point_b
    
    # Calculate magnitudes
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    # Handle degenerate cases (zero-length vectors)
    if mag_ba < 1e-10 or mag_bc < 1e-10:
        return 0.0
    
    # Calculate dot product
    dot_product = np.dot(ba, bc)
    
    # Calculate cosine (clamp to [-1, 1] to avoid acos domain errors)
    cos_angle = np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def calculate_angle_2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    Calculate the angle ABC from 2D points (ignoring z-coordinate).
    
    Useful for UI calculations where only x,y positions matter.
    
    Args:
        a: First point (x, y)
        b: Vertex point (x, y)
        c: Third point (x, y)
    
    Returns:
        Angle in degrees (0-180)
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    if mag_ba < 1e-10 or mag_bc < 1e-10:
        return 0.0
    
    cos_angle = np.clip(np.dot(ba, bc) / (mag_ba * mag_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def calculate_distance(
    a: Union[Point3D, Any],
    b: Union[Point3D, Any]
) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        a: First point
        b: Second point
    
    Returns:
        Distance (in normalized coordinate space)
    """
    if not isinstance(a, Point3D):
        a = Point3D.from_landmark(a)
    if not isinstance(b, Point3D):
        b = Point3D.from_landmark(b)
    
    return np.linalg.norm(a.to_array() - b.to_array())


def midpoint(
    a: Union[Point3D, Any],
    b: Union[Point3D, Any]
) -> Point3D:
    """
    Calculate the midpoint between two points.
    
    Args:
        a: First point
        b: Second point
    
    Returns:
        Midpoint as Point3D
    """
    if not isinstance(a, Point3D):
        a = Point3D.from_landmark(a)
    if not isinstance(b, Point3D):
        b = Point3D.from_landmark(b)
    
    return Point3D(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
        visibility=min(a.visibility, b.visibility)
    )


def is_visible(landmark: Any, threshold: float = 0.5) -> bool:
    """
    Check if a landmark has sufficient visibility.
    
    Args:
        landmark: MediaPipe landmark or Point3D
        threshold: Minimum visibility value (0-1)
    
    Returns:
        True if visibility >= threshold
    """
    visibility = getattr(landmark, 'visibility', 1.0)
    return visibility >= threshold


def get_landmark_pixel(
    landmark: Any,
    width: int,
    height: int
) -> Tuple[int, int]:
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmark: MediaPipe landmark with x, y attributes
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        (x, y) pixel coordinates
    """
    return (int(landmark.x * width), int(landmark.y * height))


def vector_angle_to_horizontal(
    a: Tuple[float, float],
    b: Tuple[float, float]
) -> float:
    """
    Calculate the angle of vector AB relative to horizontal.
    
    Useful for checking body alignment (e.g., back straightness in plank).
    
    Args:
        a: Start point (x, y)
        b: End point (x, y)
    
    Returns:
        Angle in degrees (-180 to 180), where 0 is horizontal right
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


class GeometryEngine:
    """
    Wrapper class providing geometry calculations for pose analysis.
    
    Encapsulates all math functions and provides a convenient interface
    for analyzing body landmarks.
    """
    
    def __init__(self, min_visibility: float = 0.5):
        """
        Initialize GeometryEngine.
        
        Args:
            min_visibility: Default visibility threshold for calculations
        """
        self.min_visibility = min_visibility
    
    def angle(
        self,
        a: Any,
        b: Any,
        c: Any,
        min_visibility: Optional[float] = None
    ) -> Optional[float]:
        """Calculate angle ABC at vertex B."""
        vis = min_visibility if min_visibility is not None else self.min_visibility
        return calculate_angle(a, b, c, vis)
    
    def distance(self, a: Any, b: Any) -> float:
        """Calculate distance between two points."""
        return calculate_distance(a, b)
    
    def midpoint(self, a: Any, b: Any) -> Point3D:
        """Calculate midpoint between two points."""
        return midpoint(a, b)
    
    def is_visible(self, landmark: Any, threshold: Optional[float] = None) -> bool:
        """Check if landmark is visible."""
        thresh = threshold if threshold is not None else self.min_visibility
        return is_visible(landmark, thresh)
    
    def to_pixel(self, landmark: Any, width: int, height: int) -> Tuple[int, int]:
        """Convert landmark to pixel coordinates."""
        return get_landmark_pixel(landmark, width, height)
