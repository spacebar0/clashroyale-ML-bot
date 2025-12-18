"""
Curriculum Learning
Gradually reduce rule influence and increase difficulty
"""

import yaml
from pathlib import Path
from typing import Dict


class CurriculumScheduler:
    """Manages curriculum learning schedule"""
    
    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Initialize curriculum scheduler
        
        Args:
            config_path: Path to training config
        """
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.curriculum = config.get('curriculum', {})
        else:
            # Default curriculum
            self.curriculum = {
                'enable': True,
                'rule_weight_schedule': {
                    'type': 'exponential',
                    'initial': 1.0,
                    'final': 0.1,
                    'decay_rate': 0.9999
                }
            }
        
        self.current_episode = 0
        self.rule_weight = self.curriculum['rule_weight_schedule']['initial']
    
    def step(self, episode: int = None):
        """
        Update curriculum for current episode
        
        Args:
            episode: Episode number (if None, increment current)
        """
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1
        
        # Update rule weight
        self._update_rule_weight()
    
    def _update_rule_weight(self):
        """Update rule weight based on schedule"""
        if not self.curriculum.get('enable', True):
            self.rule_weight = 0.0
            return
        
        schedule = self.curriculum['rule_weight_schedule']
        schedule_type = schedule.get('type', 'exponential')
        
        if schedule_type == 'exponential':
            # Exponential decay
            initial = schedule['initial']
            final = schedule['final']
            decay_rate = schedule['decay_rate']
            
            self.rule_weight = final + (initial - final) * (decay_rate ** self.current_episode)
        
        elif schedule_type == 'linear':
            # Linear decay
            initial = schedule['initial']
            final = schedule['final']
            total_episodes = schedule.get('total_episodes', 10000)
            
            progress = min(self.current_episode / total_episodes, 1.0)
            self.rule_weight = initial + (final - initial) * progress
        
        elif schedule_type == 'step':
            # Step decay
            initial = schedule['initial']
            final = schedule['final']
            steps = schedule.get('steps', [1000, 3000, 5000])
            
            self.rule_weight = initial
            for step in steps:
                if self.current_episode >= step:
                    self.rule_weight *= 0.5
            
            self.rule_weight = max(self.rule_weight, final)
    
    def get_rule_weight(self) -> float:
        """Get current rule weight"""
        return self.rule_weight
    
    def get_stage(self) -> str:
        """Get current curriculum stage"""
        if self.rule_weight > 0.7:
            return "beginner"
        elif self.rule_weight > 0.4:
            return "intermediate"
        elif self.rule_weight > 0.2:
            return "advanced"
        else:
            return "expert"
    
    def get_info(self) -> Dict:
        """Get curriculum info"""
        return {
            'episode': self.current_episode,
            'rule_weight': self.rule_weight,
            'stage': self.get_stage()
        }
    
    def should_save_checkpoint(self, save_freq: int = 100) -> bool:
        """Check if should save checkpoint"""
        return self.current_episode % save_freq == 0


if __name__ == "__main__":
    # Test curriculum scheduler
    scheduler = CurriculumScheduler()
    
    print("=== Curriculum Learning Schedule ===")
    print(f"Initial rule weight: {scheduler.get_rule_weight():.4f}")
    print(f"Initial stage: {scheduler.get_stage()}\n")
    
    # Simulate training
    test_episodes = [0, 100, 500, 1000, 3000, 5000, 10000]
    
    for ep in test_episodes:
        scheduler.step(ep)
        info = scheduler.get_info()
        
        print(f"Episode {ep:5d} | "
              f"Rule weight: {info['rule_weight']:.4f} | "
              f"Stage: {info['stage']}")
