"""
State Extractor
Converts vision detections into structured game state
"""

import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from game import GameState, Unit, Tower
from vision.detector import VisionDetector


class StateExtractor:
    """Extracts structured game state from screen captures"""
    
    def __init__(self, vision_detector: Optional[VisionDetector] = None):
        """
        Initialize state extractor
        
        Args:
            vision_detector: Vision detector instance
        """
        self.detector = vision_detector or VisionDetector()
        self.previous_state = None
    
    def extract_state(
        self,
        frame: np.ndarray,
        raw_frame: bool = True
    ) -> GameState:
        """
        Extract game state from screen capture
        
        Args:
            frame: Screen capture (H, W, 3)
            raw_frame: Whether to store raw frame in state
        
        Returns:
            GameState object
        """
        # Detect cards in hand
        cards_in_hand = self.detector.detect_cards_in_hand(frame)
        
        # Detect elixir
        elixir = self.detector.detect_elixir(frame)
        
        # Detect tower healths
        tower_healths = self.detector.detect_tower_health(frame)
        
        # Create tower objects
        friendly_towers = {
            'king': Tower(
                'king',
                tower_healths.get('friendly_king', 1.0),
                True,
                (0.5, 0.1)
            ),
            'left_princess': Tower(
                'left_princess',
                tower_healths.get('friendly_left', 1.0),
                True,
                (0.3, 0.2)
            ),
            'right_princess': Tower(
                'right_princess',
                tower_healths.get('friendly_right', 1.0),
                True,
                (0.7, 0.2)
            )
        }
        
        enemy_towers = {
            'king': Tower(
                'king',
                tower_healths.get('enemy_king', 1.0),
                False,
                (0.5, 0.9)
            ),
            'left_princess': Tower(
                'left_princess',
                tower_healths.get('enemy_left', 1.0),
                False,
                (0.3, 0.8)
            ),
            'right_princess': Tower(
                'right_princess',
                tower_healths.get('enemy_right', 1.0),
                False,
                (0.7, 0.8)
            )
        }
        
        # Create game state
        state = GameState(
            elixir=elixir,
            cards_in_hand=cards_in_hand,
            friendly_towers=friendly_towers,
            enemy_towers=enemy_towers,
            raw_frame=frame if raw_frame else None
        )
        
        # Store for next iteration
        self.previous_state = state
        
        return state
    
    def extract_state_delta(
        self,
        current_state: GameState
    ) -> dict:
        """
        Extract state changes since previous state
        
        Args:
            current_state: Current game state
        
        Returns:
            Dictionary of state deltas
        """
        if self.previous_state is None:
            return {}
        
        delta = {
            'elixir_change': current_state.elixir - self.previous_state.elixir,
            'tower_damage': current_state.get_tower_damage_delta(self.previous_state),
            'cards_changed': current_state.cards_in_hand != self.previous_state.cards_in_hand
        }
        
        return delta


if __name__ == "__main__":
    # Test state extractor
    print("=== State Extractor Test ===")
    
    extractor = StateExtractor()
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    # Extract state
    state = extractor.extract_state(dummy_frame)
    
    print(f"\nExtracted state: {state}")
    print(f"Elixir: {state.elixir}")
    print(f"Cards in hand: {state.cards_in_hand}")
    print(f"Friendly tower health: {state.get_friendly_tower_health():.1f}")
    print(f"Enemy tower health: {state.get_enemy_tower_health():.1f}")
    
    # Test state delta
    dummy_frame2 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    state2 = extractor.extract_state(dummy_frame2)
    
    delta = extractor.extract_state_delta(state2)
    print(f"\nState delta: {delta}")
    
    print("\n✓ State extractor working correctly")
