"""
Policy Network
Neural network that outputs action distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Policy network with vision encoder and action heads
    
    Architecture:
    - Vision encoder: MobileNetV3 (pretrained)
    - State encoder: MLP for scalar features
    - Card selection head: Categorical distribution
    - Placement head: Beta distribution for continuous placement
    """
    
    def __init__(
        self,
        n_cards: int = 5,  # 4 cards + no-op
        state_dim: int = 13,  # From GameState.to_vector()
        vision_dim: int = 256,
        hidden_dim: int = 128
    ):
        """
        Initialize policy network
        
        Args:
            n_cards: Number of card actions (including no-op)
            state_dim: Dimension of state vector
            vision_dim: Dimension of vision features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.n_cards = n_cards
        
        # Vision encoder (MobileNetV3-Small for CPU efficiency)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # Remove classifier, keep feature extractor
        self.vision_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Freeze early layers
        for i, layer in enumerate(self.vision_encoder):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Vision feature projection
        self.vision_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, vision_dim),  # MobileNetV3-Small output: 576
            nn.ReLU()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_dim = vision_dim + hidden_dim // 2
        
        # Card selection head (categorical)
        self.card_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cards)
        )
        
        # Placement head (beta distribution parameters)
        # Beta distribution is good for bounded continuous actions
        self.placement_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # alpha_x, beta_x, alpha_y, beta_y
        )
    
    def forward(
        self,
        vision_input: torch.Tensor,
        state_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            vision_input: Screen image (B, 3, H, W)
            state_input: State vector (B, state_dim)
        
        Returns:
            (card_logits, placement_alpha, placement_beta)
            - card_logits: (B, n_cards) logits for card selection
            - placement_alpha: (B, 2) alpha parameters for Beta distribution
            - placement_beta: (B, 2) beta parameters for Beta distribution
        """
        # Vision features
        vision_features = self.vision_encoder(vision_input)
        vision_features = self.vision_proj(vision_features)
        
        # State features
        state_features = self.state_encoder(state_input)
        
        # Combine features
        combined = torch.cat([vision_features, state_features], dim=1)
        
        # Card selection logits
        card_logits = self.card_head(combined)
        
        # Placement parameters
        placement_params = self.placement_head(combined)
        
        # Split into alpha and beta (ensure positive)
        placement_alpha = F.softplus(placement_params[:, :2]) + 1.0  # (B, 2)
        placement_beta = F.softplus(placement_params[:, 2:]) + 1.0   # (B, 2)
        
        return card_logits, placement_alpha, placement_beta
    
    def sample_action(
        self,
        vision_input: torch.Tensor,
        state_input: torch.Tensor,
        rule_prior: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            vision_input: Screen image
            state_input: State vector
            rule_prior: Optional rule prior for card selection (B, n_cards)
        
        Returns:
            (card_idx, placement, card_log_prob, placement_log_prob)
        """
        card_logits, placement_alpha, placement_beta = self.forward(vision_input, state_input)
        
        # Card selection
        if rule_prior is not None:
            # Blend policy with rule prior
            card_probs = F.softmax(card_logits, dim=-1)
            card_probs = 0.5 * card_probs + 0.5 * rule_prior
            card_probs = card_probs / card_probs.sum(dim=-1, keepdim=True)
            card_dist = torch.distributions.Categorical(probs=card_probs)
        else:
            card_dist = torch.distributions.Categorical(logits=card_logits)
        
        card_idx = card_dist.sample()
        card_log_prob = card_dist.log_prob(card_idx)
        
        # Placement (Beta distribution)
        placement_dist = torch.distributions.Beta(placement_alpha, placement_beta)
        placement_sample = placement_dist.sample()  # (B, 2) in [0, 1]
        
        # Convert from [0, 1] to [-1, 1]
        placement = placement_sample * 2.0 - 1.0
        
        # Log probability
        placement_log_prob = placement_dist.log_prob(placement_sample).sum(dim=-1)
        
        return card_idx, placement, card_log_prob, placement_log_prob
    
    def evaluate_actions(
        self,
        vision_input: torch.Tensor,
        state_input: torch.Tensor,
        card_idx: torch.Tensor,
        placement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions
        
        Args:
            vision_input: Screen image
            state_input: State vector
            card_idx: Card indices (B,)
            placement: Placement coordinates (B, 2) in [-1, 1]
        
        Returns:
            (card_log_prob, placement_log_prob, entropy)
        """
        card_logits, placement_alpha, placement_beta = self.forward(vision_input, state_input)
        
        # Card log prob
        card_dist = torch.distributions.Categorical(logits=card_logits)
        card_log_prob = card_dist.log_prob(card_idx)
        
        # Placement log prob
        # Convert placement from [-1, 1] to [0, 1]
        placement_01 = (placement + 1.0) / 2.0
        placement_dist = torch.distributions.Beta(placement_alpha, placement_beta)
        placement_log_prob = placement_dist.log_prob(placement_01).sum(dim=-1)
        
        # Entropy
        card_entropy = card_dist.entropy()
        placement_entropy = placement_dist.entropy().sum(dim=-1)
        entropy = card_entropy + placement_entropy
        
        return card_log_prob, placement_log_prob, entropy


if __name__ == "__main__":
    # Test policy network
    batch_size = 4
    
    policy = PolicyNetwork(n_cards=5, state_dim=13)
    
    # Dummy inputs
    vision = torch.randn(batch_size, 3, 360, 640)
    state = torch.randn(batch_size, 13)
    
    # Forward pass
    card_logits, alpha, beta = policy(vision, state)
    
    print("=== Policy Network ===")
    print(f"Card logits shape: {card_logits.shape}")
    print(f"Placement alpha shape: {alpha.shape}")
    print(f"Placement beta shape: {beta.shape}")
    
    # Sample action
    card_idx, placement, card_lp, place_lp = policy.sample_action(vision, state)
    
    print(f"\n=== Sampled Action ===")
    print(f"Card indices: {card_idx}")
    print(f"Placement: {placement}")
    print(f"Card log prob: {card_lp}")
    print(f"Placement log prob: {place_lp}")
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    print(f"\n=== Parameters ===")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
