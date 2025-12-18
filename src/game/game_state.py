"""
Game State Representation
Structured game state extracted from vision
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Unit:
    """Unit on battlefield"""
    card_name: str
    position: Tuple[float, float]  # Normalized (x, y) in [0, 1]
    is_friendly: bool
    health: float  # Estimated health percentage
    
    def __repr__(self) -> str:
        team = "Friendly" if self.is_friendly else "Enemy"
        return f"{team} {self.card_name} at ({self.position[0]:.2f}, {self.position[1]:.2f})"


@dataclass
class Tower:
    """Tower state"""
    name: str  # "king", "left_princess", "right_princess"
    health: float  # Health percentage [0, 1]
    is_friendly: bool
    position: Tuple[float, float]  # Normalized position
    
    def is_destroyed(self) -> bool:
        return self.health <= 0.0


@dataclass
class GameState:
    """Complete game state"""
    # Resources
    elixir: int  # Current elixir (0-10)
    
    # Cards
    cards_in_hand: List[str]  # Card names in hand
    next_card: Optional[str] = None  # Next card in cycle
    
    # Battlefield
    friendly_units: List[Unit] = field(default_factory=list)
    enemy_units: List[Unit] = field(default_factory=list)
    
    # Towers
    friendly_towers: Dict[str, Tower] = field(default_factory=dict)
    enemy_towers: Dict[str, Tower] = field(default_factory=dict)
    
    # Game info
    time_remaining: float = 180.0  # Seconds remaining
    is_overtime: bool = False
    
    # Frame
    raw_frame: Optional[np.ndarray] = None  # Raw screen capture
    
    def __post_init__(self):
        """Initialize default towers if not provided"""
        if not self.friendly_towers:
            self.friendly_towers = {
                'king': Tower('king', 1.0, True, (0.5, 0.1)),
                'left_princess': Tower('left_princess', 1.0, True, (0.3, 0.2)),
                'right_princess': Tower('right_princess', 1.0, True, (0.7, 0.2))
            }
        
        if not self.enemy_towers:
            self.enemy_towers = {
                'king': Tower('king', 1.0, False, (0.5, 0.9)),
                'left_princess': Tower('left_princess', 1.0, False, (0.3, 0.8)),
                'right_princess': Tower('right_princess', 1.0, False, (0.7, 0.8))
            }
    
    def get_friendly_tower_health(self) -> float:
        """Get total friendly tower health"""
        return sum(tower.health for tower in self.friendly_towers.values())
    
    def get_enemy_tower_health(self) -> float:
        """Get total enemy tower health"""
        return sum(tower.health for tower in self.enemy_towers.values())
    
    def get_tower_damage_delta(self, previous_state: 'GameState') -> float:
        """
        Get tower damage delta since previous state
        
        Returns:
            Positive if we dealt more damage, negative if we took more
        """
        our_damage = previous_state.get_enemy_tower_health() - self.get_enemy_tower_health()
        their_damage = previous_state.get_friendly_tower_health() - self.get_friendly_tower_health()
        
        return our_damage - their_damage
    
    def get_elixir_advantage(self) -> int:
        """
        Estimate elixir advantage
        
        Note: This is a placeholder. In reality, we can't see opponent's elixir.
        We could estimate based on their recent plays.
        """
        # For now, just return our elixir
        return self.elixir
    
    def has_card(self, card_name: str) -> bool:
        """Check if card is in hand"""
        return card_name in self.cards_in_hand
    
    def can_afford(self, card_cost: int) -> bool:
        """Check if we can afford a card"""
        return self.elixir >= card_cost
    
    def get_enemy_units_near(
        self,
        position: Tuple[float, float],
        radius: float = 0.2
    ) -> List[Unit]:
        """Get enemy units near a position"""
        nearby = []
        
        for unit in self.enemy_units:
            dist = np.sqrt(
                (unit.position[0] - position[0])**2 +
                (unit.position[1] - position[1])**2
            )
            if dist <= radius:
                nearby.append(unit)
        
        return nearby
    
    def is_tower_threatened(self, tower_name: str = 'king') -> bool:
        """Check if a tower is under threat"""
        if tower_name not in self.friendly_towers:
            return False
        
        tower = self.friendly_towers[tower_name]
        
        # Check if enemy units are near tower
        nearby_enemies = self.get_enemy_units_near(tower.position, radius=0.15)
        
        return len(nearby_enemies) > 0
    
    def to_vector(self) -> np.ndarray:
        """
        Convert state to feature vector for neural network
        
        Returns:
            Feature vector
        """
        features = []
        
        # Elixir (normalized)
        features.append(self.elixir / 10.0)
        
        # Time remaining (normalized)
        features.append(self.time_remaining / 180.0)
        
        # Overtime flag
        features.append(float(self.is_overtime))
        
        # Tower healths
        for tower in self.friendly_towers.values():
            features.append(tower.health)
        
        for tower in self.enemy_towers.values():
            features.append(tower.health)
        
        # Unit counts
        features.append(len(self.friendly_units) / 10.0)  # Normalize
        features.append(len(self.enemy_units) / 10.0)
        
        # Cards in hand (one-hot encoding would be better, but this is simpler)
        features.append(len(self.cards_in_hand) / 4.0)
        
        return np.array(features, dtype=np.float32)
    
    def __repr__(self) -> str:
        return (
            f"GameState(elixir={self.elixir}, "
            f"cards={len(self.cards_in_hand)}, "
            f"friendly_units={len(self.friendly_units)}, "
            f"enemy_units={len(self.enemy_units)}, "
            f"time={self.time_remaining:.0f}s)"
        )


if __name__ == "__main__":
    # Test game state
    state = GameState(
        elixir=7,
        cards_in_hand=['knight', 'archers', 'giant', 'fireball']
    )
    
    # Add some units
    state.friendly_units.append(
        Unit('knight', (0.5, 0.4), True, 0.8)
    )
    state.enemy_units.append(
        Unit('giant', (0.5, 0.6), False, 1.0)
    )
    
    print(state)
    print(f"\nFriendly tower health: {state.get_friendly_tower_health():.1f}")
    print(f"Enemy tower health: {state.get_enemy_tower_health():.1f}")
    print(f"King tower threatened: {state.is_tower_threatened('king')}")
    
    # Test feature vector
    features = state.to_vector()
    print(f"\nFeature vector shape: {features.shape}")
    print(f"Features: {features}")
