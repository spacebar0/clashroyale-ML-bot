"""
Rule-Guided Action Prior
Soft rules that bias action probabilities based on game state
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from game import GameState, Action, ActionSpace, CardDatabase


class RulePrior:
    """Rule-guided action prior system"""
    
    def __init__(
        self,
        rules_config: str = "config/game_rules.yaml",
        card_db: Optional[CardDatabase] = None
    ):
        """
        Initialize rule prior
        
        Args:
            rules_config: Path to game rules YAML
            card_db: Card database instance
        """
        self.rules_config = Path(rules_config)
        self.card_db = card_db or CardDatabase()
        
        # Load rules
        with open(self.rules_config, 'r') as f:
            self.rules = yaml.safe_load(f)
        
        self.placement_rules = self.rules['placement_rules']
        self.elixir_rules = self.rules['elixir_rules']
        self.card_selection_rules = self.rules['card_selection_rules']
        self.timing_rules = self.rules['timing_rules']
    
    def get_action_prior(
        self,
        state: GameState,
        rule_weight: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action prior distribution based on rules
        
        Args:
            state: Current game state
            rule_weight: Weight of rules (0=no rules, 1=full rules)
        
        Returns:
            (card_probs, placement_bias) tuple
            - card_probs: Probability distribution over cards [n_cards]
            - placement_bias: Bias for placement (x, y) coordinates
        """
        # Get card selection prior
        card_probs = self._get_card_selection_prior(state)
        
        # Get placement prior
        placement_bias = self._get_placement_prior(state)
        
        # Apply rule weight
        if rule_weight < 1.0:
            # Interpolate between uniform and rule-based
            n_cards = len(state.cards_in_hand) + 1  # +1 for no-op
            uniform_card = np.ones(n_cards) / n_cards
            card_probs = rule_weight * card_probs + (1 - rule_weight) * uniform_card
            
            # Placement bias decays to zero
            placement_bias = placement_bias * rule_weight
        
        # Normalize
        card_probs = card_probs / card_probs.sum()
        
        return card_probs, placement_bias
    
    def _get_card_selection_prior(self, state: GameState) -> np.ndarray:
        """Get card selection probability distribution"""
        n_cards = len(state.cards_in_hand) + 1  # +1 for no-op
        probs = np.ones(n_cards) * 0.1  # Base probability
        
        # Check elixir constraints
        for i, card_name in enumerate(state.cards_in_hand):
            card = self.card_db.get_card(card_name)
            
            if not state.can_afford(card.cost):
                probs[i] = 0.01  # Very low probability if can't afford
                continue
            
            # Counter-play bonus
            counter_bonus = self._get_counter_play_bonus(state, card_name)
            probs[i] += counter_bonus
            
            # Synergy bonus
            synergy_bonus = self._get_synergy_bonus(state, card_name)
            probs[i] += synergy_bonus
            
            # Pressure play bonus
            pressure_bonus = self._get_pressure_bonus(state, card_name)
            probs[i] += pressure_bonus
        
        # No-op probability (last index)
        if state.elixir < self.elixir_rules['min_reserve']:
            probs[-1] = 0.5  # Higher no-op if low elixir
        else:
            probs[-1] = 0.1
        
        return probs
    
    def _get_counter_play_bonus(self, state: GameState, card_name: str) -> float:
        """Get bonus for counter-play"""
        bonus = 0.0
        
        # Check if enemy units can be countered by this card
        for rule in self.card_selection_rules['counter_play']:
            # Check if any enemy units match the counter rule
            for enemy_unit in state.enemy_units:
                if enemy_unit.card_name == rule['enemy_card']:
                    if card_name in rule['preferred_counters']:
                        bonus += rule['weight']
        
        return bonus
    
    def _get_synergy_bonus(self, state: GameState, card_name: str) -> float:
        """Get bonus for synergy with recently played cards"""
        bonus = 0.0
        
        # Check synergies with friendly units on field
        for rule in self.card_selection_rules['synergy_play']:
            # Check if we have the synergy card on field
            for friendly_unit in state.friendly_units:
                if friendly_unit.card_name == rule['played_card']:
                    if card_name in rule['follow_up']:
                        bonus += rule['weight']
        
        return bonus
    
    def _get_pressure_bonus(self, state: GameState, card_name: str) -> float:
        """Get bonus for pressure plays"""
        bonus = 0.0
        
        for rule in self.card_selection_rules['pressure_play']:
            condition = rule['condition']
            
            # Check condition
            if condition == 'elixir_full' and state.elixir >= 9:
                if card_name in rule['preferred_cards']:
                    bonus += rule['weight']
            
            elif condition == 'tower_low_hp':
                # Check if any enemy tower is low
                for tower in state.enemy_towers.values():
                    if tower.health < 0.3:  # Below 30%
                        if card_name in rule['preferred_cards']:
                            bonus += rule['weight']
        
        return bonus
    
    def _get_placement_prior(self, state: GameState) -> Tuple[float, float]:
        """
        Get placement bias based on rules
        
        Returns:
            (x_bias, y_bias) in normalized coordinates
        """
        # Default to center
        x_bias, y_bias = 0.0, 0.0
        
        # If tower is threatened, bias towards defensive positions
        if state.is_tower_threatened():
            y_bias = -0.3  # Bias towards our side
        
        # If we have elixir advantage, bias towards offensive
        elif state.elixir >= 8:
            y_bias = 0.2  # Bias towards enemy side
        
        return (x_bias, y_bias)
    
    def get_placement_zone(
        self,
        card_name: str,
        state: GameState
    ) -> Tuple[float, float]:
        """
        Get recommended placement zone for a card
        
        Returns:
            (x, y) normalized coordinates
        """
        card = self.card_db.get_card(card_name)
        
        if card is None:
            return (0.0, 0.0)
        
        # Get role-based placement
        role = card.role.value
        
        if role not in self.placement_rules:
            return (0.0, 0.0)
        
        zones = self.placement_rules[role]['preferred_zones']
        
        # Pick zone based on weights
        weights = [zone['weight'] for zone in zones]
        chosen_zone = np.random.choice(zones, p=np.array(weights) / sum(weights))
        
        # Sample from zone
        x = np.random.uniform(chosen_zone['x_range'][0], chosen_zone['x_range'][1])
        y = np.random.uniform(chosen_zone['y_range'][0], chosen_zone['y_range'][1])
        
        return (x, y)


if __name__ == "__main__":
    # Test rule prior
    from game import GameState, Unit
    
    rule_prior = RulePrior()
    
    # Create test state
    state = GameState(
        elixir=7,
        cards_in_hand=['knight', 'archers', 'giant', 'fireball']
    )
    
    # Add enemy giant
    state.enemy_units.append(Unit('giant', (0.5, 0.6), False, 1.0))
    
    # Get action prior
    card_probs, placement_bias = rule_prior.get_action_prior(state, rule_weight=1.0)
    
    print("=== Card Selection Prior ===")
    for i, card_name in enumerate(state.cards_in_hand):
        print(f"{card_name:15s}: {card_probs[i]:.3f}")
    print(f"{'NO-OP':15s}: {card_probs[-1]:.3f}")
    
    print(f"\n=== Placement Bias ===")
    print(f"Bias: ({placement_bias[0]:.2f}, {placement_bias[1]:.2f})")
    
    # Test placement zones
    print("\n=== Placement Zones ===")
    for card_name in state.cards_in_hand:
        x, y = rule_prior.get_placement_zone(card_name, state)
        print(f"{card_name:15s}: ({x:.2f}, {y:.2f})")
