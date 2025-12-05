import os
import sys
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.environment import DroneRacingEnv


class WandbNoSymlink(WandbCallback):
    """Custom Wandb callback that avoids symlinks (Windows compatible)."""
    def save_model(self):
        if self.model_save_path is None:
            return
        dst = os.path.join(self.model_save_path, "model.zip")
        self.model.save(dst)


def print_gpu_info():
    """Print GPU diagnostics."""
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")


def make_env(rank, log_dir):
    def _init():
        env = DroneRacingEnv()
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
    return _init


def make_eval_env():
    def _init():
        return DroneRacingEnv()
    return _init


if __name__ == "__main__":
    # ----------------------
    #  CONFIGURATION
    # ----------------------
    MODEL_NAME = "drone_racing_final"
    TOTAL_TIMESTEPS = 500_000
    NUM_ENVS = 16
    
    # Best hyperparameters from sweep
    GAMMA = 0.999
    LEARNING_RATE = 0.001
    BUFFER_SIZE = 50000
    TAU = 0.005
    BATCH_SIZE = 32
    
    # ----------------------
    #  SETUP
    # ----------------------
    print_gpu_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"Training Final Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Environments: {NUM_ENVS}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"{'='*50}\n")
    
    # Directories
    log_dir = os.path.join("final_training", "logs")
    checkpoint_dir = os.path.join("final_training", "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ----------------------
    #  WANDB INIT
    # ----------------------
    run = wandb.init(
        project="drone-racing-final",
        name=MODEL_NAME,
        config={
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "tau": TAU,
            "batch_size": BATCH_SIZE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
        },
        sync_tensorboard=True,
        settings=wandb.Settings(symlink=False),
    )
    
    # ----------------------
    #  ENVIRONMENTS
    # ----------------------
    env = SubprocVecEnv([make_env(k, log_dir) for k in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_eval_env() for _ in range(4)])
    
    # ----------------------
    #  MODEL
    # ----------------------
    print(f"\nCreating new TQC model...")
    model = TQC(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        tau=TAU,
        batch_size=BATCH_SIZE,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
    )
    
    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} steps")
    print(f"Hyperparameters:")
    print(f"  gamma={GAMMA}, lr={LEARNING_RATE}, buffer={BUFFER_SIZE}, tau={TAU}, batch={BATCH_SIZE}")
    
    # ----------------------
    #  CALLBACKS
    # ----------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // NUM_ENVS,  # Save every ~100k steps
        save_path=checkpoint_dir,
        name_prefix=MODEL_NAME,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=50_000 // NUM_ENVS,  # Evaluate every ~50k steps
        n_eval_episodes=10,
        deterministic=True,
    )
    
    wandb_callback = WandbNoSymlink(
        model_save_path=os.path.join("final_training", "wandb_models", run.id),
        model_save_freq=100_000 // NUM_ENVS,
        verbose=1,
    )
    
    # ----------------------
    #  TRAINING
    # ----------------------
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback, wandb_callback],
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}_final.zip")
        model.save(final_path)
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Final model saved to: {final_path}")
        print(f"Best model saved to: {checkpoint_dir}/best_model.zip")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(os.path.join(checkpoint_dir, f"{MODEL_NAME}_interrupted.zip"))
        
    finally:
        env.close()
        eval_env.close()
        wandb.finish()
