"""
Visual Demonstration Script for Trained Drone Racing Agent.

Loads a trained TQC model and runs visual demonstrations of the
drone racing through the gate course.

Features:
- Real-time visualization with pygame
- Episode metrics logging to Weights & Biases
- Lap time display and completion detection

Usage:
    python scripts/demo.py
    
Controls:
    - Close window to exit

Requirements:
    - Trained model at 'models/best_model.zip'
    - pygame installed
"""

import os
import sys
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import wandb

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from sb3_contrib import TQC
from environment import DroneRacingEnv


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'models', 
    'best_model.zip'
)

NUM_EPISODES = 5
WANDB_PROJECT = "drone-racing"


# =============================================================================
# DEMONSTRATION FUNCTION
# =============================================================================

def run_demonstration():
    """Load the trained model and run visual demonstrations."""
    
    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    
    print("="*60)
    print("  DRONE RACING - VISUAL DEMONSTRATION")
    print("="*60)
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at: {MODEL_PATH}")
        print("\nPlease ensure you have a trained model. You can:")
        print("  1. Run training: python scripts/train.py")
        print("  2. Copy your model to: models/best_model.zip")
        return

    # Initialize wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        name="visual_demo",
        config={
            "model_path": MODEL_PATH,
            "num_episodes": NUM_EPISODES,
        },
        settings=wandb.Settings(symlink=False)
    )
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 Loading model on device: {device}")
    
    try:
        model = TQC.load(MODEL_PATH, device=device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        wandb.finish()
        return

    # Create visualization environment
    env = DroneRacingEnv(render_mode="human")
    
    if not PYGAME_AVAILABLE:
        print("⚠️ Warning: pygame not installed. No visualization.")

    # -------------------------------------------------------------------------
    # RUN EPISODES
    # -------------------------------------------------------------------------
    
    print(f"\n📊 Running {NUM_EPISODES} demonstration episodes...\n")
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        episode_reward = 0
        step_count = 0
        start_time = time.time()
        
        print(f"--- Episode {episode + 1}/{NUM_EPISODES} ---")
        
        while not terminated and not truncated:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Log step metrics
            run.log({
                "step_reward": reward,
                "episode": episode + 1,
                "step": step_count,
                "gate_idx": env.current_gate_idx
            })
            
            # Handle window close
            if PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        wandb.finish()
                        print("\n👋 Demonstration ended by user.")
                        return
        
        # Episode summary
        episode_time = time.time() - start_time
        
        run.log({
            "episode_reward": episode_reward,
            "episode": episode + 1,
            "episode_steps": step_count,
            "episode_time": episode_time
        })
        
        status = "🏁 FINISHED!" if env._race_completed else "❌ Failed"
        print(f"  {status} Reward: {episode_reward:.1f}, Time: {episode_time:.2f}s, Steps: {step_count}")
        
        time.sleep(1)  # Pause between episodes

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    
    env.close()
    wandb.finish()
    
    print(f"\n{'='*60}")
    print("  DEMONSTRATION COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    run_demonstration()
