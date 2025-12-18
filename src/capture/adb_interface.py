"""
ADB Interface for BlueStacks
Handles low-level communication with BlueStacks emulator
"""

import subprocess
import time
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import io


class ADBInterface:
    """Low-level ADB wrapper for BlueStacks"""
    
    def __init__(self, device_id: str = "127.0.0.1:5555"):
        """
        Initialize ADB interface
        
        Args:
            device_id: BlueStacks ADB device ID (default: 127.0.0.1:5555)
        """
        self.device_id = device_id
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to BlueStacks via ADB"""
        try:
            # Connect to BlueStacks
            result = subprocess.run(
                ["adb", "connect", self.device_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "connected" in result.stdout.lower():
                self.connected = True
                print(f"✓ Connected to BlueStacks at {self.device_id}")
                return True
            else:
                print(f"✗ Failed to connect: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"✗ ADB connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from BlueStacks"""
        if self.connected:
            subprocess.run(["adb", "disconnect", self.device_id])
            self.connected = False
            print("Disconnected from BlueStacks")
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """
        Capture screenshot from BlueStacks
        
        Returns:
            Screenshot as numpy array (H, W, 3) in RGB format, or None if failed
        """
        if not self.connected:
            print("✗ Not connected to ADB")
            return None
        
        try:
            # Capture screenshot using screencap
            result = subprocess.run(
                ["adb", "-s", self.device_id, "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Convert bytes to image
                image = Image.open(io.BytesIO(result.stdout))
                return np.array(image)
            else:
                print(f"✗ Screenshot failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"✗ Screenshot error: {e}")
            return None
    
    def tap(self, x: int, y: int, duration: int = 100):
        """
        Simulate tap at screen coordinates
        
        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            duration: Tap duration in milliseconds
        """
        if not self.connected:
            print("✗ Not connected to ADB")
            return
        
        try:
            subprocess.run(
                ["adb", "-s", self.device_id, "shell", "input", "tap", str(x), str(y)],
                timeout=2
            )
        except Exception as e:
            print(f"✗ Tap error: {e}")
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """
        Simulate swipe/drag gesture
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            duration: Swipe duration in milliseconds
        """
        if not self.connected:
            print("✗ Not connected to ADB")
            return
        
        try:
            subprocess.run(
                ["adb", "-s", self.device_id, "shell", "input", "swipe",
                 str(x1), str(y1), str(x2), str(y2), str(duration)],
                timeout=2
            )
        except Exception as e:
            print(f"✗ Swipe error: {e}")
    
    def get_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """
        Get screen resolution
        
        Returns:
            (width, height) tuple or None if failed
        """
        if not self.connected:
            return None
        
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "wm", "size"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Parse output: "Physical size: 1080x1920"
            if "Physical size:" in result.stdout:
                size_str = result.stdout.split("Physical size:")[1].strip()
                width, height = map(int, size_str.split("x"))
                return (width, height)
            
            return None
            
        except Exception as e:
            print(f"✗ Resolution error: {e}")
            return None
    
    def is_screen_on(self) -> bool:
        """Check if screen is on"""
        if not self.connected:
            return False
        
        try:
            result = subprocess.run(
                ["adb", "-s", self.device_id, "shell", "dumpsys", "power"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            return "mHoldingDisplaySuspendBlocker=true" in result.stdout
            
        except Exception as e:
            print(f"✗ Screen check error: {e}")
            return False


if __name__ == "__main__":
    # Test ADB connection
    adb = ADBInterface()
    
    if adb.connect():
        print(f"Screen resolution: {adb.get_screen_resolution()}")
        print(f"Screen on: {adb.is_screen_on()}")
        
        # Test screenshot
        screen = adb.capture_screen()
        if screen is not None:
            print(f"Screenshot captured: {screen.shape}")
        
        adb.disconnect()
