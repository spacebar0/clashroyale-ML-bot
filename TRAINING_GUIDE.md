# Card Detector Training Guide

## Quick Start

### Option 1: Test with Synthetic Data (Immediate)

```bash
cd "c:\Users\naman\Desktop\CLR BOT"

# Create synthetic dataset for testing
python scripts/train_card_detector.py --create-synthetic --epochs 10

# This will:
# 1. Generate 50 synthetic images per card (600 total)
# 2. Train for 10 epochs
# 3. Save best model to data/checkpoints/card_detector_best.pt
```

### Option 2: Train with Real Data (Recommended)

#### Step 1: Collect Card Images

**From Screenshots:**
```bash
# Extract cards from a single screenshot
python scripts/collect_card_images.py --screenshot gameplay.png --cards knight archers giant fireball
```

**From Videos:**
```bash
# 1. Extract frames from gameplay video
python scripts/collect_card_images.py --video gameplay.mp4 --sample-rate 30

# 2. Label extracted frames interactively
python scripts/collect_card_images.py --label

# 3. Check collection statistics
python scripts/collect_card_images.py --stats
```

#### Step 2: Train the Model

```bash
# Train on collected images
python scripts/train_card_detector.py --data-dir data/card_images --epochs 50 --batch-size 32

# With GPU (if available)
python scripts/train_card_detector.py --data-dir data/card_images --epochs 50 --device cuda
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `data/card_images` | Directory with card images |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--device` | `cpu` | Device (cpu or cuda) |

## Expected Results

### Synthetic Data (Testing Only)
- **Training Accuracy**: ~95-100% (overfitting on synthetic data)
- **Validation Accuracy**: ~90-95%
- **Training Time**: ~5-10 minutes on CPU

⚠️ **Note**: Synthetic data is for testing the pipeline only. The model won't work on real Clash Royale screenshots!

### Real Data (Production)
- **Recommended Dataset Size**: 100+ images per card (1,200+ total)
- **Expected Accuracy**: 85-95% (depends on data quality)
- **Training Time**: 20-30 minutes on CPU, 5-10 minutes on GPU

## Dataset Requirements

### Image Quality
- **Resolution**: At least 64x64 pixels per card
- **Clarity**: Cards should be clearly visible
- **Variety**: Different backgrounds, lighting conditions
- **Balance**: Similar number of images per card

### Recommended Sources
1. **Gameplay Videos**: Download from YouTube using `scripts/download_videos.py`
2. **Live Gameplay**: Capture screenshots while playing
3. **Replays**: Record and extract frames from replays

## Data Collection Workflow

```
1. Download Videos
   ↓
2. Extract Frames (every 30 frames)
   ↓
3. Label Cards Interactively
   ↓
4. Check Dataset Balance
   ↓
5. Train Model
   ↓
6. Evaluate on Validation Set
```

## Monitoring Training

Training progress is logged to:
- **Console**: Real-time loss and accuracy
- **TensorBoard**: Detailed metrics (coming soon)
- **Checkpoints**: Saved every 10 epochs
- **History**: JSON file with all metrics

## Using the Trained Model

```python
from vision.detector import CardDetector
import torch

# Load trained model
model = CardDetector(num_cards=12)
model.load_state_dict(torch.load('data/checkpoints/card_detector_best.pt')['model_state_dict'])
model.eval()

# Use in VisionDetector
from vision import VisionDetector

detector = VisionDetector(card_model_path='data/checkpoints/card_detector_best.pt')
```

## Troubleshooting

### Low Accuracy
- **Collect more data**: Aim for 100+ images per card
- **Balance dataset**: Ensure similar counts for all cards
- **Increase epochs**: Try 100+ epochs
- **Check data quality**: Remove blurry or mislabeled images

### Overfitting
- **Reduce epochs**: Stop when validation accuracy plateaus
- **Add more augmentation**: Modify transforms in `train_card_detector.py`
- **Collect more diverse data**: Different game modes, arenas

### Slow Training
- **Reduce batch size**: Try 16 instead of 32
- **Use GPU**: Add `--device cuda` if available
- **Reduce image size**: Modify dataset to use smaller images

## Next Steps

After training:
1. ✅ Test model on new screenshots
2. ✅ Integrate with full vision pipeline
3. ✅ Calibrate detection regions
4. ✅ Test in live gameplay
5. ✅ Collect more data for underperforming cards

## Advanced: Fine-Tuning

```bash
# Continue training from checkpoint
python scripts/train_card_detector.py \
    --data-dir data/card_images \
    --epochs 20 \
    --checkpoint data/checkpoints/card_detector_best.pt
```

(Note: Checkpoint loading not yet implemented - coming soon!)
