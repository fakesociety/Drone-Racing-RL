def wandb_example():
    import random
    import wandb

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="moty-ruppin-academic-center",  # החלף ל-entity שלך
        # Set the wandb project where this run will be logged.
        project="RacingDrone",   # החלף ל-project שלך
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
        settings=wandb.Settings(symlink=False)
    )

    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # Log metrics to wandb.
        run.log({"acc": acc, "loss": loss})

    # Finish the run and upload any remaining data.
    run.finish()
import time
import random
import wandb
import torch
import os
import sys
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
from sb3_contrib import TQC
from src.environment import DroneRacingEnv 

# --- Settings ---

MODEL_PATH = "top_model_7.3s\/\/best_model.zip"

def run_demonstration():
    """Loads the trained model and runs a visual demonstration."""
    
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="moty-ruppin-academic-center",  # TODO: החלף ל-entity שלך
        # Set the wandb project where this run will be logged.
        project="RacingDrone",  # TODO: החלף ל-project שלך
        # Track hyperparameters and run metadata.
        config={
            "model": "moty",
            "visual_demo": True,
        },
        settings=wandb.Settings(symlink=False)
    )
    # 1. Check model existence
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print("Please run the training script (train.py) first.")
        return

    # 2. Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Loading model on device: {device}")
    
    try:
        model = TQC.load(MODEL_PATH, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Create environment for visualization
    eval_env = DroneRacingEnv(render_mode="human")
    
    if not PYGAME_AVAILABLE:
        print("⚠️ Warning: Pygame not installed. Visualization will be skipped, but loop will run.")

    num_episodes_to_run = 5
    for episode in range(num_episodes_to_run):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False

        print(f"\n--- Starting Episode {episode + 1} ---")

        episode_reward = 0
        step_count = 0
        start_time = time.time()
        last_gate_idx = None
        while not terminated and not truncated:
            # AI chooses action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            step_count += 1

            # Print gate index if changed
            gate_idx = getattr(eval_env, 'current_gate_idx', None)
            if gate_idx is not None and gate_idx != last_gate_idx:
                print(f"Gate index: {gate_idx}")
                last_gate_idx = gate_idx

            # Log metrics to wandb for each step (optional, can be per episode)
            run.log({"step_reward": reward, "episode": episode + 1, "step": step_count, "gate_idx": gate_idx})

            # 4. Safe exit
            if PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        eval_env.close()
                        run.finish()
                        return
            # Optional: Limit FPS for human viewing if not handled by env
            # eval_env.render() # Env handles this in step() if render_mode is human

        episode_time = time.time() - start_time
        # Log episode summary
        run.log({"episode_reward": episode_reward, "episode": episode + 1, "steps": step_count, "episode_time": episode_time})
        print(f"Episode finished. Reward: {episode_reward}, Time: {episode_time:.2f} seconds")
        time.sleep(1)

    eval_env.close()

    # Finish the run and upload any remaining data.
    run.finish()

if __name__ == '__main__':
    run_demonstration()