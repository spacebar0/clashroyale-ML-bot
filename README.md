# Clash Royale Vision-Based RL Agent

A research-grade reinforcement learning agent that learns to play Clash Royale from raw screen pixels using PyTorch and PPO.

## 🎯 Overview

This project implements a vision-based RL agent that:
- **Observes** gameplay from raw screen pixels via BlueStacks
- **Perceives** game state using computer vision (MobileNet-based detector)
- **Decides** strategically using PPO with rule-guided priors
- **Acts** by executing touch commands via ADB
- **Learns** progressively through imitation learning + self-play

## 🏗️ Architecture

```
Screen Capture (ADB) → Vision (MobileNet) → Game State → PPO Agent → Action Execution
                                                              ↑
                                                      Rule-Guided Prior
                                                      (Curriculum Learning)
```

### Key Components

- **Vision Pipeline**: MobileNet-SSD for CPU-efficient object detection
- **Game Knowledge**: Arena 1-2 card database with roles, synergies, counters
- **PPO Agent**: Policy + value networks with soft rule guidance
- **Action Space**: Discrete card selection + continuous placement grid
- **Learning**: Imitation learning (YouTube videos) → PPO self-play

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- BlueStacks Android emulator
- ADB (Android Debug Bridge)
- (Optional) GPU for faster training

### Installation

```bash
# Clone repository
cd "CLR BOT"

# Install dependencies
pip install -r requirements.txt

# Verify ADB connection to BlueStacks
adb connect 127.0.0.1:5555
```

### Usage

```bash
# Download gameplay videos from YouTube
python scripts/download_videos.py --query "clash royale arena 1 gameplay" --count 50

# Process videos for imitation learning
python scripts/process_videos.py --input data/videos --output data/processed

# Pre-train with imitation learning
python main.py train --mode imitation --data data/processed --epochs 50

# Fine-tune with PPO self-play
python main.py train --mode ppo --checkpoint data/checkpoints/imitation_best.pt --episodes 10000

# Evaluate agent
python main.py eval --checkpoint data/checkpoints/ppo_best.pt --episodes 100

# Watch agent play live
python main.py play --checkpoint data/checkpoints/ppo_best.pt
```

## 📁 Project Structure

```
clash_royale_rl/
├── config/              # Configuration files
│   ├── cards.yaml       # Card database (Arena 1-2)
│   ├── game_rules.yaml  # Soft gameplay rules
│   └── training.yaml    # Hyperparameters
├── src/
│   ├── capture/         # Screen capture & ADB
│   ├── vision/          # Object detection & state extraction
│   ├── game/            # Card DB, game state, action space
│   ├── agent/           # PPO agent & rule priors
│   ├── learning/        # Training loops
│   ├── execution/       # Action execution
│   └── utils/           # Logging & metrics
├── data/
│   ├── videos/          # Downloaded gameplay videos
│   ├── processed/       # Extracted training data
│   └── checkpoints/     # Model checkpoints
└── tests/               # Unit tests
```

## 🎮 Arena 1-2 Cards

The agent is initially trained on 12 cards:

**Troops**: Knight, Archers, Bomber, Giant, Mini P.E.K.K.A, Musketeer, Baby Dragon, Skeleton Army  
**Spells**: Arrows, Fireball, Goblin Barrel  
**Buildings**: Tombstone

## 🧠 Learning Approach

### 1. Imitation Learning (Pre-training)
- Download gameplay videos from YouTube
- Extract state-action pairs using vision
- Train policy via behavioral cloning

### 2. PPO Self-Play (Fine-tuning)
- Agent plays against itself
- Reward shaping: tower damage, elixir efficiency, win/loss
- Curriculum learning: gradually reduce rule guidance

### 3. Rule-Guided Priors
Soft rules bias action probabilities (not hard-coded):
- **Placement**: Tanks at bridge, ranged behind tanks, buildings in center
- **Elixir**: Maintain 2+ elixir reserve, spend more when ahead
- **Counter-play**: Anti-air vs flying units, spells vs swarms
- **Synergies**: Giant + Musketeer, Knight + Archers

Curriculum schedule reduces rule weight: 1.0 → 0.1 over 10K episodes

## 📊 Training Configuration

**PPO Hyperparameters**:
- Learning rate: 3e-4
- Batch size: 64 (with gradient accumulation)
- Clip epsilon: 0.2
- GAE lambda: 0.95
- Entropy coefficient: 0.01

**Network Architecture**:
- Vision encoder: MobileNetV3-Small (CPU-optimized)
- Policy head: Categorical (card) + Beta (placement)
- Value network: Shared encoder

**Reward Shaping**:
- Win: +10, Loss: -10
- Tower damage: ±0.01 per HP
- Elixir efficiency: +0.005
- Time penalty: -0.001 (encourage action)

## 🔧 CPU Optimization

Designed for limited GPU resources:
- Lightweight MobileNet backbone (~30 FPS on CPU)
- Gradient accumulation for larger effective batch size
- Mixed precision training (FP16) when GPU available
- Checkpoint-based training (pause/resume friendly)

## 📈 Monitoring

Training metrics logged to TensorBoard:
```bash
tensorboard --logdir logs/
```

Metrics tracked:
- Win rate
- Average reward
- Policy entropy
- Value/policy loss
- KL divergence
- Elixir efficiency

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_vision.py
pytest tests/test_agent.py
```

## 🛠️ Development

### Adding New Cards
1. Update `config/cards.yaml` with card stats
2. Add synergies and counters
3. Update rule priors in `config/game_rules.yaml`

### Extending to Higher Arenas
1. Expand card database
2. Retrain vision model for new cards
3. Adjust rule priors for new mechanics

## 📝 TODO

- [x] Implement vision model (card, elixir, tower detection)
- [ ] Train card detector on Clash Royale screenshots
- [ ] Add video download script
- [ ] Complete PPO agent implementation
- [ ] Add action executor
- [ ] Create imitation learning pipeline
- [ ] Implement self-play environment
- [ ] Add comprehensive tests
- [ ] Optimize for real-time inference

## 🤝 Contributing

This is a research project. Contributions welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- OpenAI for PPO algorithm
- BlueStacks for Android emulation
- Clash Royale community for gameplay insights

---

**Note**: This agent is for educational and research purposes only. Use responsibly and in accordance with Clash Royale's terms of service.
