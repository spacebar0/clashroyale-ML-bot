"""
Card Image Collector
Extracts card images from Clash Royale screenshots for training
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent))

from capture import ScreenCapture


class CardImageCollector:
    """Collects card images from screenshots"""
    
    def __init__(self, output_dir: str = 'data/card_images'):
        """
        Initialize collector
        
        Args:
            output_dir: Directory to save card images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Card regions (normalized coordinates)
        self.card_regions = [
            {'x': 0.15, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 1
            {'x': 0.35, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 2
            {'x': 0.55, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 3
            {'x': 0.75, 'y': 0.85, 'w': 0.15, 'h': 0.12},  # Card 4
        ]
        
        # Card names
        self.card_names = [
            'knight', 'archers', 'bomber', 'fireball', 'arrows',
            'giant', 'mini_pekka', 'musketeer', 'goblin_barrel',
            'skeleton_army', 'tombstone', 'baby_dragon'
        ]
        
        # Create directories for each card
        for card_name in self.card_names:
            (self.output_dir / card_name).mkdir(exist_ok=True)
        
        # Counter for saved images
        self.counters = {name: 0 for name in self.card_names}
    
    def extract_cards_from_screenshot(
        self,
        screenshot_path: str,
        card_labels: List[str]
    ):
        """
        Extract card images from a screenshot
        
        Args:
            screenshot_path: Path to screenshot
            card_labels: List of 4 card names in the screenshot (left to right)
        """
        # Load screenshot
        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"✗ Failed to load {screenshot_path}")
            return
        
        h, w = img.shape[:2]
        
        # Extract each card
        for i, (region, card_name) in enumerate(zip(self.card_regions, card_labels)):
            if card_name not in self.card_names:
                print(f"⚠ Unknown card: {card_name}, skipping")
                continue
            
            # Calculate pixel coordinates
            x = int(region['x'] * w)
            y = int(region['y'] * h)
            card_w = int(region['w'] * w)
            card_h = int(region['h'] * h)
            
            # Extract card region
            card_img = img[y:y+card_h, x:x+card_w]
            
            if card_img.size == 0:
                print(f"✗ Failed to extract card {i+1}")
                continue
            
            # Save card image
            counter = self.counters[card_name]
            save_path = self.output_dir / card_name / f'{card_name}_{counter:04d}.png'
            
            cv2.imwrite(str(save_path), card_img)
            self.counters[card_name] += 1
            
            print(f"✓ Saved {card_name} #{counter} from position {i+1}")
    
    def collect_from_video(
        self,
        video_path: str,
        sample_rate: int = 30
    ):
        """
        Extract card images from gameplay video
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every N frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return
        
        frame_count = 0
        extracted_count = 0
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Save frame for manual labeling
                frame_path = self.output_dir / f'frame_{frame_count:06d}.png'
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
                
                print(f"Extracted frame {frame_count} ({extracted_count} total)")
        
        cap.release()
        
        print(f"\n✓ Extracted {extracted_count} frames from video")
        print(f"⚠ Please manually label cards in each frame and run:")
        print(f"   python scripts/collect_card_images.py --label")
    
    def print_stats(self):
        """Print collection statistics"""
        print("\n=== Card Collection Statistics ===")
        
        total = 0
        for card_name in self.card_names:
            count = len(list((self.output_dir / card_name).glob('*.png')))
            print(f"{card_name:20s}: {count:4d} images")
            total += count
        
        print(f"\nTotal: {total} card images")
        
        # Check balance
        counts = [len(list((self.output_dir / card_name).glob('*.png'))) 
                 for card_name in self.card_names]
        
        if max(counts) > 0:
            balance = min(counts) / max(counts)
            print(f"Dataset balance: {balance:.2%}")
            
            if balance < 0.5:
                print("⚠ Dataset is imbalanced! Collect more images for underrepresented cards.")


def interactive_labeling(collector: CardImageCollector):
    """Interactive tool for labeling screenshots"""
    print("\n=== Interactive Card Labeling ===")
    print("For each screenshot, enter the 4 cards from left to right")
    print("Available cards:")
    
    for i, card in enumerate(collector.card_names, 1):
        print(f"  {i:2d}. {card}")
    
    print("\nCommands:")
    print("  Enter 4 card names separated by spaces")
    print("  Type 'quit' to exit")
    print("  Type 'stats' to see statistics\n")
    
    # Find unlabeled screenshots
    screenshots = list(collector.output_dir.glob('frame_*.png'))
    
    if not screenshots:
        print("No screenshots found. Use --video to extract frames first.")
        return
    
    for screenshot in screenshots:
        print(f"\nScreenshot: {screenshot.name}")
        
        # Show image
        img = cv2.imread(str(screenshot))
        if img is not None:
            cv2.imshow('Screenshot', img)
            cv2.waitKey(100)
        
        # Get labels
        while True:
            labels_input = input("Enter 4 cards (or 'skip'/'quit'): ").strip().lower()
            
            if labels_input == 'quit':
                cv2.destroyAllWindows()
                return
            
            if labels_input == 'skip':
                break
            
            if labels_input == 'stats':
                collector.print_stats()
                continue
            
            labels = labels_input.split()
            
            if len(labels) != 4:
                print("✗ Please enter exactly 4 card names")
                continue
            
            if all(label in collector.card_names for label in labels):
                collector.extract_cards_from_screenshot(str(screenshot), labels)
                screenshot.unlink()  # Delete labeled screenshot
                break
            else:
                invalid = [l for l in labels if l not in collector.card_names]
                print(f"✗ Invalid cards: {invalid}")
    
    cv2.destroyAllWindows()
    print("\n✓ Labeling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect card images for training')
    
    parser.add_argument('--output', type=str, default='data/card_images',
                       help='Output directory for card images')
    parser.add_argument('--video', type=str,
                       help='Extract frames from video file')
    parser.add_argument('--screenshot', type=str,
                       help='Extract cards from single screenshot')
    parser.add_argument('--cards', type=str, nargs=4,
                       help='Card names for screenshot (left to right)')
    parser.add_argument('--label', action='store_true',
                       help='Interactive labeling mode')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='Sample every N frames from video')
    
    args = parser.parse_args()
    
    # Create collector
    collector = CardImageCollector(args.output)
    
    if args.stats:
        collector.print_stats()
    
    elif args.video:
        collector.collect_from_video(args.video, args.sample_rate)
    
    elif args.screenshot and args.cards:
        collector.extract_cards_from_screenshot(args.screenshot, args.cards)
        print("✓ Cards extracted successfully")
    
    elif args.label:
        interactive_labeling(collector)
    
    else:
        print("Card Image Collector")
        print("\nUsage:")
        print("  1. Extract frames from video:")
        print("     python scripts/collect_card_images.py --video path/to/video.mp4")
        print("\n  2. Label extracted frames:")
        print("     python scripts/collect_card_images.py --label")
        print("\n  3. Extract from single screenshot:")
        print("     python scripts/collect_card_images.py --screenshot image.png --cards knight archers giant fireball")
        print("\n  4. Show statistics:")
        print("     python scripts/collect_card_images.py --stats")
        
        collector.print_stats()
