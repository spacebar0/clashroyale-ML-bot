"""
Reward Shaping
Compute rewards from game state transitions
"""

import yaml
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from game import GameState


class RewardShaper:
    """Reward shaping for RL training"""
    
    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Initialize reward shaper
        
        Args:
            config_path: Path to training config with reward weights
        """
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.rewards = config['rewards']
        else:
            # Default rewards
            self.rewards = {
                'win': 10.0,
                'loss': -10.0,
                'draw': 0.0,
                'tower_damage_dealt': 0.01,
                'tower_damage_taken': -0.01,
                'elixir_efficiency': 0.005,
                'time_penalty': -0.001,
                'invalid_action': -0.1
            }
    
    def compute_reward(
        self,
        prev_state: GameState,
        curr_state: GameState,
        action_valid: bool = True,
        game_over: bool = False,
        game_result: Optional[str] = None
    ) -> float:
        """
        Compute reward for state transition
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            action_valid: Whether action was valid
            game_over: Whether game ended
            game_result: 'win', 'loss', or 'draw' if game_over
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Terminal reward
        if game_over and game_result is not None:
            reward += self.rewards.get(game_result, 0.0)
            return reward
        
        # Invalid action penalty
        if not action_valid:
            reward += self.rewards['invalid_action']
            return reward
        
        # Tower damage reward
        tower_damage_delta = curr_state.get_tower_damage_delta(prev_state)
        reward += tower_damage_delta * self.rewards['tower_damage_dealt']
        
        # Elixir efficiency
        # Reward spending elixir (encourages action)
        elixir_spent = prev_state.elixir - curr_state.elixir
        if elixir_spent > 0:
            reward += elixir_spent * self.rewards['elixir_efficiency']
        
        # Time penalty (encourage faster play)
        reward += self.rewards['time_penalty']
        
        return reward
    
    def compute_episode_reward(
        self,
        game_result: str,
        total_damage_dealt: float,
        total_damage_taken: float,
        episode_length: int
    ) -> float:
        """
        Compute total episode reward
        
        Args:
            game_result: 'win', 'loss', or 'draw'
            total_damage_dealt: Total tower damage dealt
            total_damage_taken: Total tower damage taken
            episode_length: Number of steps in episode
        
        Returns:
            Total episode reward
        """
        reward = 0.0
        
        # Terminal reward
        reward += self.rewards.get(game_result, 0.0)
        
        # Damage rewards
        reward += total_damage_dealt * self.rewards['tower_damage_dealt']
        reward += total_damage_taken * self.rewards['tower_damage_taken']
        
        # Time penalty
        reward += episode_length * self.rewards['time_penalty']
        
        return reward


class AdvantageEstimator:
    """Generalized Advantage Estimation (GAE)"""
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize GAE
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_advantages(
        self,
        rewards: list,
        values: list,
        dones: list
    ) -> tuple:
        """
        Compute advantages using GAE
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
        
        Returns:
            (advantages, returns) tuple
        """
        advantages = []
        returns = []
        
        gae = 0.0
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns


if __name__ == "__main__":
    # Test reward shaper
    from game import GameState, Unit
    
    reward_shaper = RewardShaper()
    
    # Create states
    prev_state = GameState(elixir=7, cards_in_hand=['knight', 'archers', 'giant', 'fireball'])
    curr_state = GameState(elixir=4, cards_in_hand=['knight', 'archers', 'giant', 'fireball'])
    
    # Damage enemy tower
    curr_state.enemy_towers['left_princess'].health = 0.8
    
    # Compute reward
    reward = reward_shaper.compute_reward(prev_state, curr_state, action_valid=True)
    
    print("=== Reward Shaping ===")
    print(f"Previous elixir: {prev_state.elixir}")
    print(f"Current elixir: {curr_state.elixir}")
    print(f"Enemy tower health: {curr_state.enemy_towers['left_princess'].health}")
    print(f"Reward: {reward:.4f}")
    
    # Test terminal reward
    terminal_reward = reward_shaper.compute_reward(
        prev_state, curr_state,
        game_over=True,
        game_result='win'
    )
    print(f"\nTerminal reward (win): {terminal_reward:.4f}")
    
    # Test GAE
    print("\n=== Advantage Estimation ===")
    gae = AdvantageEstimator(gamma=0.99, gae_lambda=0.95)
    
    rewards = [0.1, 0.2, -0.1, 0.5, 10.0]
    values = [1.0, 1.5, 1.2, 2.0, 0.0]
    dones = [0, 0, 0, 0, 1]
    
    advantages, returns = gae.compute_advantages(rewards, values, dones)
    
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")
    print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")
    print(f"Returns: {[f'{r:.3f}' for r in returns]}")
