"""
Screen Capture Module
High-level screen capture with preprocessing for vision models
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from .adb_interface import ADBInterface


class ScreenCapture:
    """High-level screen capture with preprocessing"""
    
    def __init__(
        self,
        device_id: str = "127.0.0.1:5555",
        target_size: Tuple[int, int] = (640, 360),
        normalize: bool = True
    ):
        """
        Initialize screen capture
        
        Args:
            device_id: BlueStacks ADB device ID
            target_size: Resize captured frames to this size (W, H)
            normalize: Normalize pixel values to [0, 1]
        """
        self.adb = ADBInterface(device_id)
        self.target_size = target_size
        self.normalize = normalize
        self.screen_resolution = None
        
    def connect(self) -> bool:
        """Connect to device and get screen info"""
        if self.adb.connect():
            self.screen_resolution = self.adb.get_screen_resolution()
            print(f"Screen resolution: {self.screen_resolution}")
            return True
        return False
    
    def disconnect(self):
        """Disconnect from device"""
        self.adb.disconnect()
    
    def capture(self, preprocess: bool = True) -> Optional[np.ndarray]:
        """
        Capture and preprocess screen
        
        Args:
            preprocess: Apply preprocessing (resize, normalize)
        
        Returns:
            Preprocessed frame as numpy array or None if failed
        """
        # Capture raw screenshot
        frame = self.adb.capture_screen()
        
        if frame is None:
            return None
        
        if not preprocess:
            return frame
        
        # Preprocess
        return self._preprocess(frame)
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for vision model
        
        Args:
            frame: Raw frame (H, W, 3)
        
        Returns:
            Preprocessed frame
        """
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        preprocess: bool = True
    ) -> Optional[np.ndarray]:
        """
        Capture specific screen region
        
        Args:
            x, y: Top-left corner (in original resolution)
            width, height: Region size
            preprocess: Apply preprocessing
        
        Returns:
            Cropped and preprocessed region
        """
        frame = self.adb.capture_screen()
        
        if frame is None:
            return None
        
        # Crop region
        region = frame[y:y+height, x:x+width]
        
        if preprocess:
            region = self._preprocess(region)
        
        return region
    
    def get_coordinates_mapping(self) -> dict:
        """
        Get mapping between normalized coordinates and screen pixels
        
        Returns:
            Dictionary with conversion functions
        """
        if self.screen_resolution is None:
            return None
        
        width, height = self.screen_resolution
        
        return {
            'screen_width': width,
            'screen_height': height,
            'to_pixels': lambda x_norm, y_norm: (
                int(x_norm * width),
                int(y_norm * height)
            ),
            'to_normalized': lambda x_px, y_px: (
                x_px / width,
                y_px / height
            )
        }


if __name__ == "__main__":
    # Test screen capture
    capture = ScreenCapture()
    
    if capture.connect():
        # Capture frame
        frame = capture.capture()
        
        if frame is not None:
            print(f"Captured frame: {frame.shape}, dtype: {frame.dtype}")
            print(f"Value range: [{frame.min():.3f}, {frame.max():.3f}]")
            
            # Save test image
            import cv2
            test_img = (frame * 255).astype(np.uint8)
            cv2.imwrite("test_capture.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
            print("Saved test_capture.png")
        
        capture.disconnect()
