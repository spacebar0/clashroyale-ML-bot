"""
Action Executor
Converts agent actions to ADB touch commands
"""

import time
import numpy as np
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from capture import ADBInterface
from game import Action, ActionSpace, GameState, CardDatabase


class ActionExecutor:
    """Executes agent actions on BlueStacks"""
    
    def __init__(
        self,
        adb: ADBInterface,
        action_space: ActionSpace,
        card_db: CardDatabase,
        screen_width: int = 1080,
        screen_height: int = 1920
    ):
        """
        Initialize action executor
        
        Args:
            adb: ADB interface instance
            action_space: Action space instance
            card_db: Card database
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.adb = adb
        self.action_space = action_space
        self.card_db = card_db
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Card positions in hand (normalized coordinates)
        # These are approximate and may need calibration
        self.card_positions = [
            (0.2, 0.9),   # Card 1
            (0.4, 0.9),   # Card 2
            (0.6, 0.9),   # Card 3
            (0.8, 0.9),   # Card 4
        ]
    
    def execute_action(
        self,
        action: Action,
        state: GameState,
        validate: bool = True
    ) -> bool:
        """
        Execute action on device
        
        Args:
            action: Action to execute
            state: Current game state
            validate: Whether to validate action before executing
        
        Returns:
            True if action executed successfully
        """
        # No-op
        if action.is_noop():
            time.sleep(0.1)  # Small delay
            return True
        
        # Validate action
        if validate and not self._validate_action(action, state):
            print(f"✗ Invalid action: {action}")
            return False
        
        # Get card
        card_idx = action.card_idx
        if card_idx >= len(state.cards_in_hand):
            print(f"✗ Card index out of range: {card_idx}")
            return False
        
        card_name = state.cards_in_hand[card_idx]
        card = self.card_db.get_card(card_name)
        
        if card is None:
            print(f"✗ Unknown card: {card_name}")
            return False
        
        # Check elixir
        if not state.can_afford(card.cost):
            print(f"✗ Cannot afford {card_name} (cost: {card.cost}, have: {state.elixir})")
            return False
        
        # Execute drag gesture
        success = self._drag_card_to_position(card_idx, action.placement)
        
        if success:
            print(f"✓ Played {card_name} at ({action.placement[0]:.2f}, {action.placement[1]:.2f})")
        
        return success
    
    def _validate_action(self, action: Action, state: GameState) -> bool:
        """Validate action is feasible"""
        # Check card index
        if action.card_idx >= len(state.cards_in_hand):
            return False
        
        # Check placement bounds
        x, y = action.placement
        if abs(x) > 1.0 or abs(y) > 1.0:
            return False
        
        # Check elixir
        card_name = state.cards_in_hand[action.card_idx]
        card = self.card_db.get_card(card_name)
        
        if card and not state.can_afford(card.cost):
            return False
        
        return True
    
    def _drag_card_to_position(
        self,
        card_idx: int,
        placement: Tuple[float, float]
    ) -> bool:
        """
        Drag card from hand to placement position
        
        Args:
            card_idx: Index of card in hand (0-3)
            placement: Normalized placement coordinates (-1 to 1)
        
        Returns:
            True if successful
        """
        # Get card position in hand
        if card_idx >= len(self.card_positions):
            return False
        
        card_x_norm, card_y_norm = self.card_positions[card_idx]
        
        # Convert to pixels
        start_x = int(card_x_norm * self.screen_width)
        start_y = int(card_y_norm * self.screen_height)
        
        # Convert placement to pixels
        end_x, end_y = self.action_space.denormalize_to_screen(
            placement[0], placement[1],
            self.screen_width, self.screen_height
        )
        
        # Ensure placement is in valid area (arena, not UI)
        # Arena is roughly in the middle 60% of screen vertically
        arena_y_min = int(self.screen_height * 0.2)
        arena_y_max = int(self.screen_height * 0.8)
        
        end_y = np.clip(end_y, arena_y_min, arena_y_max)
        
        # Execute swipe
        try:
            self.adb.swipe(start_x, start_y, end_x, end_y, duration=300)
            time.sleep(0.2)  # Wait for animation
            return True
        except Exception as e:
            print(f"✗ Swipe failed: {e}")
            return False
    
    def tap_position(self, x: float, y: float) -> bool:
        """
        Tap at normalized position
        
        Args:
            x, y: Normalized coordinates (-1 to 1)
        
        Returns:
            True if successful
        """
        # Convert to pixels
        px, py = self.action_space.denormalize_to_screen(
            x, y, self.screen_width, self.screen_height
        )
        
        try:
            self.adb.tap(px, py)
            return True
        except Exception as e:
            print(f"✗ Tap failed: {e}")
            return False
    
    def calibrate_card_positions(self) -> bool:
        """
        Calibrate card positions by detecting them in a screenshot
        
        Returns:
            True if calibration successful
        """
        # TODO: Implement card position detection using vision
        print("⚠ Card position calibration not yet implemented")
        print("Using default positions")
        return False


if __name__ == "__main__":
    # Test action executor
    from capture import ADBInterface
    from game import GameState, Action
    
    # Connect to ADB
    adb = ADBInterface()
    
    if not adb.connect():
        print("Failed to connect to ADB")
        exit(1)
    
    # Get screen resolution
    resolution = adb.get_screen_resolution()
    if resolution is None:
        print("Failed to get screen resolution")
        exit(1)
    
    screen_w, screen_h = resolution
    
    # Create executor
    action_space = ActionSpace()
    card_db = CardDatabase()
    executor = ActionExecutor(adb, action_space, card_db, screen_w, screen_h)
    
    # Create test state
    state = GameState(
        elixir=10,
        cards_in_hand=['knight', 'archers', 'giant', 'fireball']
    )
    
    # Create test action
    action = Action(card_idx=0, placement=(0.0, 0.2))
    
    print(f"Test action: {action}")
    print(f"State: {state}")
    
    # Validate action
    if executor._validate_action(action, state):
        print("✓ Action is valid")
        
        # Ask user before executing
        response = input("Execute action on device? (y/n): ")
        
        if response.lower() == 'y':
            success = executor.execute_action(action, state)
            print(f"Execution {'successful' if success else 'failed'}")
    else:
        print("✗ Action is invalid")
    
    adb.disconnect()
