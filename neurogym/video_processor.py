"""VideoProcessor - Handles webcam input using OpenCV."""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator


class VideoProcessor:
    """
    Handles webcam/video input capture and frame processing.
    
    Supports context manager protocol for automatic resource cleanup.
    
    Example:
        with VideoProcessor(source=0, width=1280, height=720) as processor:
            for frame in processor:
                # Process frame
                pass
    """
    
    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        """
        Initialize the video processor.
        
        Args:
            source: Webcam index (0, 1, ...) or video file path.
            width: Desired frame width.
            height: Desired frame height.
            fps: Desired frames per second.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        
    def __enter__(self) -> "VideoProcessor":
        """Context manager entry - initialize capture device."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.release()
        
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate over frames from the video source."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
            
    def start(self) -> bool:
        """
        Start the video capture device.
        
        Returns:
            True if capture device opened successfully.
        """
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Set capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reduce buffer size for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read and return the next frame.
        
        Returns:
            Frame in RGB format, or None if no frame available.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
            
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            return None
        
        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    def read_frame_bgr(self) -> Optional[np.ndarray]:
        """
        Read and return the next frame in BGR format.
        
        Returns:
            Frame in BGR format, or None if no frame available.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
            
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            return None
        
        return frame
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the actual frame dimensions.
        
        Returns:
            Tuple of (width, height).
        """
        if self._cap is None:
            return (self.width, self.height)
            
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (actual_width, actual_height)
    
    def release(self) -> None:
        """Release the video capture device and free resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    @property
    def is_opened(self) -> bool:
        """Check if capture device is opened."""
        return self._cap is not None and self._cap.isOpened()
