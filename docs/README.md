# Documentation

This folder contains documentation and media for the Drone Racing RL project.

## Contents

- `demo.gif` - Animated demonstration of the trained agent (add this after recording)
- `training_curve.png` - Training performance graph from W&B (export from your dashboard)
- `architecture.md` - Technical architecture details (optional)

## Creating Documentation Assets

### Recording a Demo GIF

1. Run the demo: `python scripts/demo.py`
2. Use screen recording software (OBS, ShareX, etc.)
3. Convert to GIF using ffmpeg or online tools
4. Save as `demo.gif`

### Exporting Training Curves

1. Go to your W&B dashboard
2. Select the training run
3. Export the reward/episode graphs
4. Save as `training_curve.png`
