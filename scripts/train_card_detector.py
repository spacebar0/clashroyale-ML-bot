"""
Card Detector Training Script
Trains the MobileNetV3-based card classifier on Clash Royale card images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vision.detector import CardDetector
from utils import get_logger


class CardDataset(Dataset):
    """Dataset for card images"""
    
    def __init__(
        self,
        data_dir: str,
        card_names: List[str],
        transform=None,
        augment: bool = True
    ):
        """
        Initialize card dataset
        
        Args:
            data_dir: Directory containing card images
            card_names: List of card names (classes)
            transform: Image transformations
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.card_names = card_names
        self.transform = transform
        self.augment = augment
        
        # Build dataset
        self.samples = []
        self._build_dataset()
        
        # Default transforms
        if self.transform is None:
            if augment:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(0.3),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
    
    def _build_dataset(self):
        """Build dataset from directory structure"""
        for idx, card_name in enumerate(self.card_names):
            card_dir = self.data_dir / card_name
            
            if not card_dir.exists():
                print(f"Warning: {card_dir} not found, skipping {card_name}")
                continue
            
            # Find all images
            for img_path in card_dir.glob("*.png"):
                self.samples.append((str(img_path), idx))
            
            for img_path in card_dir.glob("*.jpg"):
                self.samples.append((str(img_path), idx))
        
        print(f"Loaded {len(self.samples)} card images across {len(self.card_names)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CardDetectorTrainer:
    """Trainer for card detector"""
    
    def __init__(
        self,
        model: CardDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        checkpoint_dir: str = 'data/checkpoints'
    ):
        """
        Initialize trainer
        
        Args:
            model: CardDetector model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(log_dir='logs')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = 50):
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log results
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('card_detector_best.pt', epoch, val_acc)
                self.logger.info(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'card_detector_epoch_{epoch+1}.pt', epoch, val_acc)
        
        self.logger.info(f"\n✓ Training complete! Best Val Acc: {self.best_val_acc:.2f}%")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, checkpoint_path)
    
    def save_history(self):
        """Save training history as JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")


def create_synthetic_dataset(output_dir: str = 'data/card_images'):
    """
    Create synthetic card dataset for testing
    (In production, use real Clash Royale screenshots)
    """
    output_path = Path(output_dir)
    
    card_names = [
        'knight', 'archers', 'bomber', 'fireball', 'arrows',
        'giant', 'mini_pekka', 'musketeer', 'goblin_barrel',
        'skeleton_army', 'tombstone', 'baby_dragon'
    ]
    
    print("Creating synthetic card dataset...")
    
    for card_name in card_names:
        card_dir = output_path / card_name
        card_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 50 synthetic images per card
        for i in range(50):
            # Random colored image (placeholder for real cards)
            img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
            
            # Add some variation
            color = np.random.randint(0, 255, 3)
            img[:, :] = color
            
            # Add noise
            noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save
            cv2.imwrite(str(card_dir / f'{card_name}_{i:03d}.png'), img)
    
    print(f"✓ Created synthetic dataset at {output_path}")
    print("⚠ Replace with real Clash Royale card screenshots for actual training!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train card detector')
    parser.add_argument('--data-dir', type=str, default='data/card_images',
                       help='Directory containing card images')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to train on (cpu or cuda)')
    parser.add_argument('--create-synthetic', action='store_true',
                       help='Create synthetic dataset for testing')
    
    args = parser.parse_args()
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir)
    
    # Card names
    card_names = [
        'knight', 'archers', 'bomber', 'fireball', 'arrows',
        'giant', 'mini_pekka', 'musketeer', 'goblin_barrel',
        'skeleton_army', 'tombstone', 'baby_dragon'
    ]
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CardDataset(
        args.data_dir,
        card_names,
        augment=True
    )
    
    val_dataset = CardDataset(
        args.data_dir,
        card_names,
        augment=False
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    _, val_dataset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = CardDetector(num_cards=len(card_names), pretrained=True)
    
    # Create trainer
    trainer = CardDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n✓ Training complete!")
    print(f"Best model saved at: data/checkpoints/card_detector_best.pt")
