# Drone Racing RL

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/SB3-TQC-orange.svg)

**Train autonomous drones to race through gates using deep reinforcement learning**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Results](#results) • [Documentation](#documentation)

</div>

---

## 🎯 Overview

This project implements a reinforcement learning agent that learns to pilot a racing drone through a circular gate course. The drone must navigate through 5 gates and return to the starting position to complete a lap, all while maintaining stable flight.

<div align="center">
<img src="docs/demo.gif" alt="Drone Racing Demo" width="600">
</div>

### Key Achievements

- 🏆 **7.3 second** lap times (optimized from 30+ seconds)
- 🚀 Stable flight with smooth gate transitions
- 📈 Consistent performance across episodes

## ✨ Features

- **Custom Gymnasium Environment**: 2D physics simulation with realistic drone dynamics
- **TQC Algorithm**: State-of-the-art off-policy RL with distributional critics
- **GPU Acceleration**: CUDA support for fast training on NVIDIA GPUs
- **Parallel Training**: 16 simultaneous environments for efficient learning
- **Experiment Tracking**: Full Weights & Biases integration
- **Manual Control**: Test the environment with keyboard controls

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-racing-rl.git
cd drone-racing-rl

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Quick Start

### Run Pre-trained Model

```bash
python scripts/demo.py
```

### Train New Model

```bash
python scripts/train.py
```

Training takes approximately 30 minutes on an RTX 3070 (500k timesteps).

## 📊 Results

### Training Performance

| Metric | Value |
|--------|-------|
| Best Lap Time | 7.3 seconds |
| Average Reward | ~10,000 |
| Training Steps | 500,000 |
| Training Time | ~5 minutes |

### Learning Curve

The agent learns to complete laps within the first 200k steps, then optimizes for speed:

![Training Curve](docs/training_curve.png)

## 🏗️ Project Structure

```
drone-racing-rl/
├── src/
│   ├── __init__.py
│   └── environment/
│       ├── __init__.py
│       └── drone_racing_env.py    # Gymnasium environment
├── scripts/
│   ├── train.py                   # Training script
│   ├── demo.py                    # Visual demonstration
│   └── manual_control.py          # Keyboard control
├── models/
│   └── best_model.zip             # Pre-trained TQC model
├── configs/
│   └── hyperparameters.yaml       # Training configuration
├── docs/
│   └── ...                        # Documentation
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Environment Details

### Observation Space (8D)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | angle | Drone angle (radians) |
| 1 | angular_velocity | Rotation speed (rad/s) |
| 2 | vx | Horizontal velocity (m/s) |
| 3 | vy | Vertical velocity (m/s) |
| 4 | dx | X distance to target gate |
| 5 | dy | Y distance to target gate |
| 6 | sin_rel_angle | Sine of angle to target |
| 7 | cos_rel_angle | Cosine of angle to target |

### Action Space (2D)

| Index | Range | Description |
|-------|-------|-------------|
| 0 | [-1, 1] | Left motor thrust |
| 1 | [-1, 1] | Right motor thrust |

Note: Action value of 0 corresponds to hover thrust.

### Reward Structure

| Event | Reward |
|-------|--------|
| Progress toward gate | (prev_dist - curr_dist) × 100 |
| Velocity bonus | speed × 0.5 |
| Gate crossing | 300 + speed_bonus |
| Lap completion | 2000 + time_bonus |
| Time penalty | -0.1 per step |
| Crash | -20 |
| Timeout | -500 |

## ⚙️ Hyperparameters

Optimized via Weights & Biases sweep:

| Parameter | Value |
|-----------|-------|
| Algorithm | TQC |
| Gamma (γ) | 0.999 |
| Learning Rate | 0.001 |
| Buffer Size | 50,000 |
| Tau (τ) | 0.005 |
| Batch Size | 32 |
| Parallel Envs | 16 |

## 📈 Weights & Biases

Training metrics are automatically logged to W&B:

- Episode rewards
- Lap completion times
- Learning curves
- Model checkpoints

View runs at: [wandb.ai/My-project](https://wandb.ai/moty-ruppin-academic-center/drone-racing-final)
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the RL framework
- [Gymnasium](https://gymnasium.farama.org/) for the environment API
- [Weights & Biases](https://wandb.ai) for experiment tracking

---

<div align="center">
Made with ❤️ by fakesociety 

⭐ Star this repo if you find it useful!
</div>
