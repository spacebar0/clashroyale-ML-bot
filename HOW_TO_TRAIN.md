# How to Train the Card Detector - Step by Step

## 📁 Step 1: Organize Your Gameplay Footage

### Where to Put Your Files

Create these folders and upload your files:

```
c:\Users\naman\Desktop\CLR BOT\data\
├── raw_videos\          ← PUT YOUR GAMEPLAY VIDEOS HERE (.mp4, .avi, .webm)
└── raw_screenshots\     ← PUT YOUR SCREENSHOTS HERE (.png, .jpg)
```

**Create the folders:**
```bash
cd "c:\Users\naman\Desktop\CLR BOT"
mkdir data\raw_videos
mkdir data\raw_screenshots
```

### What Kind of Footage to Upload

**Best Quality:**
- ✅ Clear, high-resolution gameplay (720p or higher)
- ✅ Cards clearly visible at bottom of screen
- ✅ Different game modes (1v1, 2v2, challenges)
- ✅ Various card combinations
- ✅ Good lighting/visibility

**Avoid:**
- ❌ Blurry or low-resolution footage
- ❌ Menu screens or non-gameplay
- ❌ Heavily compressed videos

---

## 🎬 Step 2: Extract Card Images from Videos

### Option A: Automatic Extraction (Recommended)

```bash
cd "c:\Users\naman\Desktop\CLR BOT"

# Extract frames from all videos in raw_videos folder
for %f in (data\raw_videos\*.mp4) do python scripts\collect_card_images.py --video "%f" --sample-rate 30
```

This will:
- Extract 1 frame every 30 frames (~1 per second)
- Save frames to `data/card_images/` for labeling
- Process all videos automatically

### Option B: Manual Screenshots

If you have screenshots instead:
```bash
# Copy your screenshots to raw_screenshots folder
# Then we'll label them in Step 3
```

---

## 🏷️ Step 3: Label the Cards

This is the **most important step**! You need to tell the system which cards are in each image.

### Interactive Labeling Tool

```bash
cd "c:\Users\naman\Desktop\CLR BOT"
python scripts\collect_card_images.py --label
```

**What happens:**
1. A window shows each screenshot
2. You type the 4 card names from **left to right**
3. Cards are automatically extracted and saved

**Example:**
```
Screenshot: frame_000030.png
[Image shows cards: Knight, Archers, Giant, Fireball]

Enter 4 cards: knight archers giant fireball
✓ Saved knight #0 from position 1
✓ Saved archers #0 from position 2
✓ Saved giant #0 from position 3
✓ Saved fireball #0 from position 4
```

### Available Card Names

Type exactly these names (lowercase):
```
knight          archers         bomber
fireball        arrows          giant
mini_pekka      musketeer       goblin_barrel
skeleton_army   tombstone       baby_dragon
```

### Labeling Tips

- **Be accurate**: Wrong labels = bad model
- **Skip unclear images**: Type `skip` if cards aren't visible
- **Take breaks**: Label 50-100 at a time
- **Check progress**: Type `stats` to see how many you've labeled

---

## 📊 Step 4: Check Dataset Balance

Before training, make sure you have enough images:

```bash
python scripts\collect_card_images.py --stats
```

**Good Dataset:**
```
=== Card Collection Statistics ===
knight              :   95 images
archers             :  102 images
bomber              :   88 images
...
Total: 1,150 card images
Dataset balance: 85%
```

**Minimum Requirements:**
- ✅ At least **50 images per card** (600 total)
- ✅ Recommended: **100+ images per card** (1,200 total)
- ✅ Balance above **70%** (similar counts for all cards)

**If imbalanced:**
- Collect more footage featuring underrepresented cards
- Or remove some images from overrepresented cards

---

## 🚀 Step 5: Train the Model

### Quick Test (5 minutes)

First, test with synthetic data to make sure everything works:

```bash
python scripts\train_card_detector.py --create-synthetic --epochs 5
```

If this works, proceed to real training ↓

### Real Training

```bash
cd "c:\Users\naman\Desktop\CLR BOT"

# Train on your labeled data
python scripts\train_card_detector.py --data-dir data\card_images --epochs 50 --batch-size 32
```

**What you'll see:**
```
Loading datasets...
Loaded 960 card images across 12 classes

Initializing model...

==================================================
Starting training...
==================================================

Epoch 1/50
Training: 100%|████████| 24/24 [00:45<00:00]
  loss: 2.1234, acc: 45.23%
Validation: 100%|████████| 6/6 [00:08<00:00]
  Val Loss: 1.8765, Val Acc: 52.34%

Epoch 2/50
...
✓ New best model saved! Val Acc: 67.89%
```

**Training will take:**
- CPU: 20-40 minutes
- GPU (if available): 5-10 minutes

---

## 📈 Step 6: Monitor Training Progress

### What to Watch

**Good Signs:**
- ✅ Validation accuracy increasing
- ✅ Training loss decreasing
- ✅ Accuracy above 80% by epoch 30-40

**Warning Signs:**
- ⚠️ Validation accuracy stuck below 70%
  → Need more/better data
- ⚠️ Training accuracy 100%, validation accuracy 50%
  → Overfitting, need more diverse data
- ⚠️ Both accuracies very low
  → Check if labels are correct

### Training Output Files

After training completes:
```
data\checkpoints\
├── card_detector_best.pt          ← BEST MODEL (use this!)
├── card_detector_epoch_10.pt      ← Checkpoint at epoch 10
├── card_detector_epoch_20.pt      ← Checkpoint at epoch 20
└── training_history.json          ← Training metrics
```

---

## ✅ Step 7: Test Your Trained Model

### Quick Test

```python
# Test the trained model
cd "c:\Users\naman\Desktop\CLR BOT"
python

>>> from vision.detector import VisionDetector
>>> import cv2
>>> 
>>> # Load detector with your trained model
>>> detector = VisionDetector(card_model_path='data/checkpoints/card_detector_best.pt')
>>> 
>>> # Test on a screenshot
>>> img = cv2.imread('data/raw_screenshots/test.png')
>>> cards = detector.detect_cards_in_hand(img)
>>> print(f"Detected cards: {cards}")
```

### Integration Test

The model is now ready to use in the full RL agent!

---

## 🔄 Step 8: Improve the Model (Optional)

If accuracy is low, try:

### Collect More Data
```bash
# Add more videos
# Copy to data\raw_videos\
# Re-run extraction and labeling
```

### Train Longer
```bash
python scripts\train_card_detector.py --data-dir data\card_images --epochs 100
```

### Fine-Tune
After collecting more data, continue training:
```bash
# Load previous best model and train more
# (Feature coming soon)
```

---

## 📋 Complete Workflow Summary

```
1. Upload videos → data\raw_videos\
2. Extract frames → python scripts\collect_card_images.py --video ...
3. Label cards → python scripts\collect_card_images.py --label
4. Check stats → python scripts\collect_card_images.py --stats
5. Train model → python scripts\train_card_detector.py --epochs 50
6. Use model → VisionDetector(card_model_path='data/checkpoints/card_detector_best.pt')
```

---

## ❓ Troubleshooting

### "No screenshots found"
- Make sure you ran the video extraction step first
- Check that frames are in `data/card_images/`

### "Invalid card name"
- Use exact names from the list (lowercase)
- Check for typos

### "Dataset is imbalanced"
- Collect more footage with underrepresented cards
- Or label more existing frames

### Training is very slow
- Reduce batch size: `--batch-size 16`
- Reduce epochs for testing: `--epochs 20`
- Consider using a GPU if available

### Low accuracy (< 70%)
- Collect more data (aim for 100+ per card)
- Check label quality (re-label if needed)
- Train for more epochs (try 100)

---

## 🎯 Expected Results

With **100+ images per card**:
- Training Accuracy: 90-95%
- Validation Accuracy: 85-92%
- Real-world Accuracy: 80-90%

The model should correctly identify cards in most gameplay situations!

---

## 📞 Need Help?

If you get stuck:
1. Check the error message
2. Verify file paths are correct
3. Make sure Python and dependencies are installed
4. Check that images are labeled correctly

Ready to start? Begin with **Step 1** and upload your gameplay footage! 🚀
