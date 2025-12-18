"""
Vision Detector
Lightweight object detection for Clash Royale game state extraction
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class CardDetector(nn.Module):
    """
    Lightweight card detector using MobileNetV3
    Detects cards in hand from screen capture
    """
    
    def __init__(self, num_cards: int = 12, pretrained: bool = True):
        """
        Initialize card detector
        
        Args:
            num_cards: Number of card classes (12 for Arena 1-2)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Remove classifier
        self.features = mobilenet.features
        
        # Card classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_cards)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Card class logits (B, num_cards)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


class ElixirDetector:
    """
    OCR-based elixir counter detector
    Uses template matching and digit recognition
    """
    
    def __init__(self):
        """Initialize elixir detector"""
        self.elixir_region = None  # Will be calibrated
    
    def detect(self, frame: np.ndarray) -> int:
        """
        Detect elixir count from frame
        
        Args:
            frame: Screen capture (H, W, 3)
        
        Returns:
            Elixir count (0-10)
        """
        # TODO: Implement OCR-based detection
        # For now, return placeholder
        return 5
    
    def calibrate(self, frame: np.ndarray):
        """
        Calibrate elixir region from a reference frame
        
        Args:
            frame: Reference frame with visible elixir counter
        """
        # Elixir counter is typically at bottom-left
        h, w = frame.shape[:2]
        
        # Approximate region (will need fine-tuning)
        self.elixir_region = {
            'x': int(w * 0.05),
            'y': int(h * 0.85),
            'width': int(w * 0.15),
            'height': int(h * 0.10)
        }


class TowerDetector:
    """
    Detect towers and estimate health from health bars
    """
    
    def __init__(self):
        """Initialize tower detector"""
        # Tower positions (normalized coordinates)
        self.tower_positions = {
            'friendly_king': (0.5, 0.1),
            'friendly_left': (0.3, 0.2),
            'friendly_right': (0.7, 0.2),
            'enemy_king': (0.5, 0.9),
            'enemy_left': (0.3, 0.8),
            'enemy_right': (0.7, 0.8)
        }
    
    def detect_health(
        self,
        frame: np.ndarray,
        tower_name: str
    ) -> float:
        """
        Detect tower health from health bar
        
        Args:
            frame: Screen capture
            tower_name: Tower identifier
        
        Returns:
            Health percentage (0.0 to 1.0)
        """
        if tower_name not in self.tower_positions:
            return 1.0
        
        # Get tower region
        h, w = frame.shape[:2]
        x_norm, y_norm = self.tower_positions[tower_name]
        
        # Extract health bar region (approximate)
        x = int(x_norm * w)
        y = int(y_norm * h)
        
        # Health bar is typically above tower
        bar_region = frame[
            max(0, y - 50):y,
            max(0, x - 40):min(w, x + 40)
        ]
        
        if bar_region.size == 0:
            return 1.0
        
        # Detect green pixels (health bar color)
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_RGB2HSV)
        
        # Green color range
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate health percentage
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = np.sum(mask > 0)
        
        if total_pixels == 0:
            return 1.0
        
        health = green_pixels / total_pixels
        return min(1.0, max(0.0, health))


class VisionDetector:
    """
    Main vision detector that combines all detection modules
    """
    
    def __init__(
        self,
        card_model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize vision detector
        
        Args:
            card_model_path: Path to trained card detector model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Initialize detectors
        self.card_detector = CardDetector(num_cards=12)
        
        if card_model_path and Path(card_model_path).exists():
            self.card_detector.load_state_dict(torch.load(card_model_path, map_location=device))
        
        self.card_detector.to(device)
        self.card_detector.eval()
        
        self.elixir_detector = ElixirDetector()
        self.tower_detector = TowerDetector()
        
        # Card regions in hand (normalized coordinates)
        self.card_regions = [
            {'x': 0.15, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 1
            {'x': 0.35, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 2
            {'x': 0.55, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 3
            {'x': 0.75, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 4
        ]
        
        # Card name mapping
        self.card_names = [
            'knight', 'archers', 'bomber', 'fireball', 'arrows',
            'giant', 'mini_pekka', 'musketeer', 'goblin_barrel',
            'skeleton_army', 'tombstone', 'baby_dragon'
        ]
    
    def detect_cards_in_hand(self, frame: np.ndarray) -> List[str]:
        """
        Detect which cards are in hand
        
        Args:
            frame: Screen capture (H, W, 3)
        
        Returns:
            List of card names
        """
        h, w = frame.shape[:2]
        cards = []
        
        for region in self.card_regions:
            # Extract card region
            x = int(region['x'] * w)
            y = int(region['y'] * h)
            card_w = int(region['w'] * w)
            card_h = int(region['h'] * h)
            
            card_img = frame[y:y+card_h, x:x+card_w]
            
            if card_img.size == 0:
                cards.append('unknown')
                continue
            
            # Resize and normalize
            card_img = cv2.resize(card_img, (64, 64))
            card_tensor = torch.from_numpy(card_img).permute(2, 0, 1).float() / 255.0
            card_tensor = card_tensor.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.card_detector(card_tensor)
                pred = torch.argmax(logits, dim=1).item()
            
            cards.append(self.card_names[pred])
        
        return cards
    
    def detect_elixir(self, frame: np.ndarray) -> int:
        """Detect current elixir count"""
        return self.elixir_detector.detect(frame)
    
    def detect_tower_health(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect all tower healths
        
        Returns:
            Dictionary of tower_name -> health_percentage
        """
        healths = {}
        
        for tower_name in self.tower_detector.tower_positions:
            health = self.tower_detector.detect_health(frame, tower_name)
            healths[tower_name] = health
        
        return healths
    
    def calibrate(self, frame: np.ndarray):
        """
        Calibrate detectors using a reference frame
        
        Args:
            frame: Reference frame from gameplay
        """
        self.elixir_detector.calibrate(frame)
        print("✓ Vision detectors calibrated")


if __name__ == "__main__":
    # Test vision detector
    print("=== Vision Detector Test ===")
    
    detector = VisionDetector()
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    # Test card detection
    print("\n=== Card Detection ===")
    cards = detector.detect_cards_in_hand(dummy_frame)
    print(f"Detected cards: {cards}")
    
    # Test elixir detection
    print("\n=== Elixir Detection ===")
    elixir = detector.detect_elixir(dummy_frame)
    print(f"Elixir: {elixir}")
    
    # Test tower detection
    print("\n=== Tower Health Detection ===")
    healths = detector.detect_tower_health(dummy_frame)
    for tower, health in healths.items():
        print(f"{tower:20s}: {health:.2%}")
    
    print("\n✓ Vision detector initialized successfully")
