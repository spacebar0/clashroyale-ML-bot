"""
Training Metrics
Track and compute training statistics
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MetricsTracker:
    """Track training metrics with moving averages"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.episode_count = 0
    
    def add(self, metric_name: str, value: float):
        """
        Add metric value
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        self.metrics[metric_name].append(value)
    
    def get_mean(self, metric_name: str) -> Optional[float]:
        """Get mean of metric over window"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        return np.mean(self.metrics[metric_name])
    
    def get_std(self, metric_name: str) -> Optional[float]:
        """Get standard deviation of metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        return np.std(self.metrics[metric_name])
    
    def get_last(self, metric_name: str) -> Optional[float]:
        """Get last value of metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        return self.metrics[metric_name][-1]
    
    def get_all(self, metric_name: str) -> List[float]:
        """Get all values of metric in window"""
        if metric_name not in self.metrics:
            return []
        
        return list(self.metrics[metric_name])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics
        
        Returns:
            Dictionary of {metric_name: {mean, std, last}}
        """
        summary = {}
        
        for metric_name in self.metrics:
            summary[metric_name] = {
                'mean': self.get_mean(metric_name),
                'std': self.get_std(metric_name),
                'last': self.get_last(metric_name),
                'count': len(self.metrics[metric_name])
            }
        
        return summary
    
    def reset(self, metric_name: Optional[str] = None):
        """
        Reset metrics
        
        Args:
            metric_name: Specific metric to reset (None = reset all)
        """
        if metric_name is None:
            self.metrics = {}
        elif metric_name in self.metrics:
            self.metrics[metric_name].clear()
    
    def increment_episode(self):
        """Increment episode counter"""
        self.episode_count += 1
    
    def format_summary(self) -> str:
        """Format summary as string"""
        summary = self.get_summary()
        
        lines = [f"Episode {self.episode_count} | Metrics Summary:"]
        
        for metric_name, stats in summary.items():
            if stats['mean'] is not None:
                lines.append(
                    f"  {metric_name:20s}: "
                    f"mean={stats['mean']:7.3f}, "
                    f"std={stats['std']:6.3f}, "
                    f"last={stats['last']:7.3f}"
                )
        
        return "\n".join(lines)


class WinRateTracker:
    """Track win/loss/draw statistics"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize win rate tracker
        
        Args:
            window_size: Window for computing win rate
        """
        self.window_size = window_size
        self.results = deque(maxlen=window_size)
        self.total_wins = 0
        self.total_losses = 0
        self.total_draws = 0
    
    def add_result(self, result: str):
        """
        Add game result
        
        Args:
            result: 'win', 'loss', or 'draw'
        """
        self.results.append(result)
        
        if result == 'win':
            self.total_wins += 1
        elif result == 'loss':
            self.total_losses += 1
        elif result == 'draw':
            self.total_draws += 1
    
    def get_win_rate(self) -> float:
        """Get win rate over window"""
        if len(self.results) == 0:
            return 0.0
        
        wins = sum(1 for r in self.results if r == 'win')
        return wins / len(self.results)
    
    def get_stats(self) -> Dict[str, float]:
        """Get detailed statistics"""
        total = len(self.results)
        
        if total == 0:
            return {
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'draw_rate': 0.0,
                'total_games': 0
            }
        
        wins = sum(1 for r in self.results if r == 'win')
        losses = sum(1 for r in self.results if r == 'loss')
        draws = sum(1 for r in self.results if r == 'draw')
        
        return {
            'win_rate': wins / total,
            'loss_rate': losses / total,
            'draw_rate': draws / total,
            'total_games': total,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'total_draws': self.total_draws
        }


if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker(window_size=10)
    
    for i in range(20):
        tracker.add('reward', np.random.randn())
        tracker.add('loss', np.random.rand())
        tracker.increment_episode()
    
    print(tracker.format_summary())
    
    # Test win rate tracker
    win_tracker = WinRateTracker(window_size=10)
    
    for result in ['win', 'loss', 'win', 'win', 'draw', 'loss']:
        win_tracker.add_result(result)
    
    print(f"\nWin rate: {win_tracker.get_win_rate():.2%}")
    print(f"Stats: {win_tracker.get_stats()}")
