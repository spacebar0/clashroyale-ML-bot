"""
Action Space Definition
Defines the action space for the RL agent
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class Action:
    """
    Agent action
    
    Combines discrete card selection with continuous placement
    """
    card_idx: int  # Card index (0-3 for cards, 4 for no-op)
    placement: Tuple[float, float]  # Normalized (x, y) in [-1, 1]
    
    def is_noop(self) -> bool:
        """Check if this is a no-op action"""
        return self.card_idx == 4
    
    def __repr__(self) -> str:
        if self.is_noop():
            return "Action(NO-OP)"
        return f"Action(card={self.card_idx}, pos=({self.placement[0]:.2f}, {self.placement[1]:.2f}))"


class ActionSpace:
    """Action space manager"""
    
    def __init__(self, n_cards: int = 4):
        """
        Initialize action space
        
        Args:
            n_cards: Number of cards in hand (default: 4)
        """
        self.n_cards = n_cards
        self.n_actions = n_cards + 1  # Cards + no-op
        
        # Placement grid (normalized to [-1, 1])
        # Clash Royale uses 18x32 tile grid
        self.grid_width = 18
        self.grid_height = 32
    
    def sample_random(self) -> Action:
        """Sample random action"""
        card_idx = np.random.randint(0, self.n_actions)
        
        # Random placement in valid area (avoid edges)
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-0.8, 0.8)
        
        return Action(card_idx, (x, y))
    
    def create_action(
        self,
        card_idx: int,
        x: float,
        y: float
    ) -> Action:
        """
        Create action from components
        
        Args:
            card_idx: Card index (0-3) or 4 for no-op
            x, y: Placement coordinates in [-1, 1]
        
        Returns:
            Action object
        """
        # Clip to valid range
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)
        
        return Action(card_idx, (x, y))
    
    def noop(self) -> Action:
        """Create no-op action"""
        return Action(4, (0.0, 0.0))
    
    def normalize_screen_coords(
        self,
        x_pixels: int,
        y_pixels: int,
        screen_width: int,
        screen_height: int
    ) -> Tuple[float, float]:
        """
        Convert screen pixel coordinates to normalized [-1, 1]
        
        Args:
            x_pixels, y_pixels: Screen coordinates in pixels
            screen_width, screen_height: Screen resolution
        
        Returns:
            Normalized (x, y) in [-1, 1]
        """
        # Normalize to [0, 1]
        x_norm = x_pixels / screen_width
        y_norm = y_pixels / screen_height
        
        # Convert to [-1, 1]
        x = x_norm * 2.0 - 1.0
        y = y_norm * 2.0 - 1.0
        
        return (x, y)
    
    def denormalize_to_screen(
        self,
        x: float,
        y: float,
        screen_width: int,
        screen_height: int
    ) -> Tuple[int, int]:
        """
        Convert normalized coordinates to screen pixels
        
        Args:
            x, y: Normalized coordinates in [-1, 1]
            screen_width, screen_height: Screen resolution
        
        Returns:
            Screen coordinates in pixels
        """
        # Convert from [-1, 1] to [0, 1]
        x_norm = (x + 1.0) / 2.0
        y_norm = (y + 1.0) / 2.0
        
        # Convert to pixels
        x_pixels = int(x_norm * screen_width)
        y_pixels = int(y_norm * screen_height)
        
        return (x_pixels, y_pixels)
    
    def snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """
        Snap coordinates to nearest grid cell
        
        Args:
            x, y: Normalized coordinates in [-1, 1]
        
        Returns:
            Snapped coordinates
        """
        # Convert to grid indices
        x_idx = int((x + 1.0) / 2.0 * self.grid_width)
        y_idx = int((y + 1.0) / 2.0 * self.grid_height)
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, self.grid_width - 1)
        y_idx = np.clip(y_idx, 0, self.grid_height - 1)
        
        # Convert back to normalized
        x_snapped = (x_idx / self.grid_width) * 2.0 - 1.0
        y_snapped = (y_idx / self.grid_height) * 2.0 - 1.0
        
        return (x_snapped, y_snapped)
    
    def is_valid_placement(
        self,
        x: float,
        y: float,
        card_name: str = None
    ) -> bool:
        """
        Check if placement is valid
        
        Args:
            x, y: Normalized coordinates
            card_name: Optional card name for card-specific rules
        
        Returns:
            True if valid placement
        """
        # Basic bounds check
        if abs(x) > 1.0 or abs(y) > 1.0:
            return False
        
        # For now, allow all placements
        # In reality, some areas are restricted (e.g., can't place in enemy territory early game)
        return True
    
    def get_placement_zones(self) -> dict:
        """
        Get predefined placement zones
        
        Returns:
            Dictionary of zone names to (x, y) coordinates
        """
        return {
            'bridge_left': (-0.4, 0.0),
            'bridge_right': (0.4, 0.0),
            'center': (0.0, 0.0),
            'back_left': (-0.4, -0.5),
            'back_right': (0.4, -0.5),
            'back_center': (0.0, -0.5),
            'defensive_left': (-0.4, -0.3),
            'defensive_right': (0.4, -0.3),
        }


if __name__ == "__main__":
    # Test action space
    action_space = ActionSpace()
    
    # Sample random actions
    print("=== Random Actions ===")
    for _ in range(5):
        action = action_space.sample_random()
        print(action)
    
    # Test no-op
    print("\n=== No-op ===")
    print(action_space.noop())
    
    # Test coordinate conversion
    print("\n=== Coordinate Conversion ===")
    screen_w, screen_h = 1080, 1920
    
    # Normalized to pixels
    x_norm, y_norm = 0.5, -0.3
    x_px, y_px = action_space.denormalize_to_screen(x_norm, y_norm, screen_w, screen_h)
    print(f"Normalized ({x_norm}, {y_norm}) -> Pixels ({x_px}, {y_px})")
    
    # Pixels to normalized
    x_norm2, y_norm2 = action_space.normalize_screen_coords(x_px, y_px, screen_w, screen_h)
    print(f"Pixels ({x_px}, {y_px}) -> Normalized ({x_norm2:.2f}, {y_norm2:.2f})")
    
    # Test grid snapping
    print("\n=== Grid Snapping ===")
    x, y = 0.37, -0.62
    x_snap, y_snap = action_space.snap_to_grid(x, y)
    print(f"({x:.2f}, {y:.2f}) -> ({x_snap:.2f}, {y_snap:.2f})")
    
    # Show placement zones
    print("\n=== Placement Zones ===")
    for zone_name, coords in action_space.get_placement_zones().items():
        print(f"{zone_name:20s}: {coords}")
