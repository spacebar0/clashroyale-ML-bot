"""
Value Network
Estimates state value for advantage calculation
"""

import torch
import torch.nn as nn
from torchvision import models


class ValueNetwork(nn.Module):
    """
    Value network for state value estimation
    
    Can share encoder with policy network for efficiency
    """
    
    def __init__(
        self,
        state_dim: int = 13,
        vision_dim: int = 256,
        hidden_dim: int = 128,
        shared_encoder: bool = True
    ):
        """
        Initialize value network
        
        Args:
            state_dim: Dimension of state vector
            vision_dim: Dimension of vision features
            hidden_dim: Hidden layer dimension
            shared_encoder: Whether to share encoder with policy
        """
        super().__init__()
        
        self.shared_encoder = shared_encoder
        
        if not shared_encoder:
            # Independent vision encoder
            mobilenet = models.mobilenet_v3_small(pretrained=True)
            self.vision_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
            
            # Freeze early layers
            for i, layer in enumerate(self.vision_encoder):
                if i < 5:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            self.vision_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(576, vision_dim),
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
            
            combined_dim = vision_dim + hidden_dim // 2
        else:
            # Will use shared features from policy network
            combined_dim = vision_dim + hidden_dim // 2
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        vision_input: torch.Tensor = None,
        state_input: torch.Tensor = None,
        combined_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            vision_input: Screen image (B, 3, H, W) - if not using shared encoder
            state_input: State vector (B, state_dim) - if not using shared encoder
            combined_features: Combined features from policy network - if using shared encoder
        
        Returns:
            State values (B, 1)
        """
        if combined_features is not None:
            # Using shared encoder
            features = combined_features
        else:
            # Independent encoder
            if not self.shared_encoder:
                vision_features = self.vision_encoder(vision_input)
                vision_features = self.vision_proj(vision_features)
                state_features = self.state_encoder(state_input)
                features = torch.cat([vision_features, state_features], dim=1)
            else:
                raise ValueError("Must provide combined_features when using shared encoder")
        
        # Value prediction
        value = self.value_head(features)
        
        return value


if __name__ == "__main__":
    # Test value network
    batch_size = 4
    
    # Test independent encoder
    print("=== Independent Encoder ===")
    value_net = ValueNetwork(state_dim=13, shared_encoder=False)
    
    vision = torch.randn(batch_size, 3, 360, 640)
    state = torch.randn(batch_size, 13)
    
    values = value_net(vision_input=vision, state_input=state)
    print(f"Values shape: {values.shape}")
    print(f"Values: {values.squeeze()}")
    
    # Test shared encoder
    print("\n=== Shared Encoder ===")
    value_net_shared = ValueNetwork(state_dim=13, shared_encoder=True)
    
    # Simulate combined features from policy network
    combined_features = torch.randn(batch_size, 256 + 64)  # vision_dim + hidden_dim//2
    
    values_shared = value_net_shared(combined_features=combined_features)
    print(f"Values shape: {values_shared.shape}")
    print(f"Values: {values_shared.squeeze()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in value_net.parameters())
    trainable_params = sum(p.numel() for p in value_net.parameters() if p.requires_grad)
    
    print(f"\n=== Parameters (Independent) ===")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    total_params_shared = sum(p.numel() for p in value_net_shared.parameters())
    trainable_params_shared = sum(p.numel() for p in value_net_shared.parameters() if p.requires_grad)
    
    print(f"\n=== Parameters (Shared) ===")
    print(f"Total: {total_params_shared:,}")
    print(f"Trainable: {trainable_params_shared:,}")
