"""
Simple Card Labeler
Label extracted frames without GUI - just shows file path
"""

import cv2
from pathlib import Path
import sys

def label_frames():
    """Simple text-based labeling"""
    
    # Card names (full names)
    card_names = [
        'arrows', 'fireball', 'giant', 'mini_pekka',
        'archers', 'knight', 'spear_goblins', 'minions', 'musketeer'
    ]
    
    # Shorthand codes for faster typing
    shorthand = {
        'a': 'arrows',
        'as': 'arrows',
        'f': 'fireball',
        'g': 'giant',
        'm': 'mini_pekka',
        'mp': 'mini_pekka',
        'ar': 'archers',
        'k': 'knight',
        'sg': 'spear_goblins',
        's': 'spear_goblins',
        'mi': 'minions',
        'mn': 'minions',
        'mu': 'musketeer',
        'ms': 'musketeer'
    }
    
    data_dir = Path('data/card_images')
    
    # Create card directories
    for card in card_names:
        (data_dir / card).mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    frames = sorted(data_dir.glob('frame_*.png'))
    
    if not frames:
        print("No frames found in data/card_images/")
        print("Make sure you ran the video extraction first!")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(frames)} frames to label")
    print(f"{'='*60}\n")
    
    print("Available cards (with shorthand codes):")
    print("  1. arrows       (a, as)")
    print("  2. fireball     (f)")
    print("  3. giant        (g)")
    print("  4. mini_pekka   (m, mp)")
    print("  5. archers      (ar)")
    print("  6. knight       (k)")
    print("  7. spear_goblins (sg, s)")
    print("  8. minions      (mi, mn)")
    print("  9. musketeer    (mu, ms)")
    
    print("\nInstructions:")
    print("  1. Open the frame image file to see the cards")
    print("  2. Enter 4 cards using full names OR shorthand (e.g., 'k ar g f')")
    print("  3. Type 'skip' to skip, 'quit' to exit, 'stats' for statistics\n")
    
    labeled_count = 0
    card_counts = {card: 0 for card in card_names}
    
    for i, frame_path in enumerate(frames, 1):
        print(f"\n[{i}/{len(frames)}] Frame: {frame_path.name}")
        print(f"Location: {frame_path.absolute()}")
        print("-" * 60)
        
        while True:
            labels_input = input("Enter 4 cards: ").strip().lower()
            
            if labels_input == 'quit':
                print(f"\nLabeled {labeled_count} frames. Progress saved!")
                return
            
            if labels_input == 'skip':
                print("Skipped.")
                break
            
            if labels_input == 'stats':
                print("\n=== Statistics ===")
                for card, count in card_counts.items():
                    print(f"  {card:20s}: {count:3d} images")
                print(f"  Total: {sum(card_counts.values())} images")
                continue
            
            labels = labels_input.split()
            
            if len(labels) != 4:
                print(f"Error: Need exactly 4 cards, got {len(labels)}")
                continue
            
            # Convert shorthand to full names
            converted_labels = []
            for label in labels:
                if label in shorthand:
                    converted_labels.append(shorthand[label])
                elif label in card_names:
                    converted_labels.append(label)
                else:
                    converted_labels.append(label)  # Keep as-is for error message
            
            labels = converted_labels
            
            if not all(label in card_names for label in labels):
                invalid = [l for l in labels if l not in card_names]
                print(f"Error: Invalid cards: {invalid}")
                print(f"Use full names or shorthand codes")
                continue
            
            # Extract and save cards
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"Error: Could not load image")
                break
            
            h, w = img.shape[:2]
            
            # Card regions (normalized)
            regions = [
                {'x': 0.15, 'y': 0.85, 'w': 0.15, 'h': 0.12},
                {'x': 0.35, 'y': 0.85, 'w': 0.15, 'h': 0.12},
                {'x': 0.55, 'y': 0.85, 'w': 0.15, 'h': 0.12},
                {'x': 0.75, 'y': 0.85, 'w': 0.15, 'h': 0.12},
            ]
            
            for card_name, region in zip(labels, regions):
                x = int(region['x'] * w)
                y = int(region['y'] * h)
                card_w = int(region['w'] * w)
                card_h = int(region['h'] * h)
                
                card_img = img[y:y+card_h, x:x+card_w]
                
                if card_img.size > 0:
                    save_path = data_dir / card_name / f'{card_name}_{card_counts[card_name]:04d}.png'
                    cv2.imwrite(str(save_path), card_img)
                    card_counts[card_name] += 1
            
            print(f"Saved: {', '.join(labels)}")
            labeled_count += 1
            
            # Delete labeled frame
            frame_path.unlink()
            break
    
    print(f"\n{'='*60}")
    print(f"Labeling complete! Labeled {labeled_count} frames")
    print(f"{'='*60}\n")
    
    print("=== Final Statistics ===")
    for card, count in card_counts.items():
        print(f"  {card:20s}: {count:3d} images")
    print(f"\nTotal: {sum(card_counts.values())} card images")
    
    balance = min(card_counts.values()) / max(card_counts.values()) if max(card_counts.values()) > 0 else 0
    print(f"Dataset balance: {balance:.1%}")

if __name__ == "__main__":
    try:
        label_frames()
    except KeyboardInterrupt:
        print("\n\nLabeling interrupted. Progress saved!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
