import os
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# TQC is part of stable-baselines3-contrib
from sb3_contrib import TQC

from racing_env import DroneRacingEnv


def print_gpu_info():
    """Print basic PyTorch / CUDA diagnostics to show which device will be used."""
    try:
        print(f"torch.__version__ = {torch.__version__}")
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                print(f"CUDA device[0] name: {torch.cuda.get_device_name(0)}")
            except Exception:
                print("CUDA device name: <unavailable>")
    except Exception as e:
        print(f"Failed to query torch/CUDA info: {e}")


# ----------------------
#  CUSTOM CALLBACK (NO SYMLINKS)
# ----------------------
class WandbNoSymlink(WandbCallback):
    def save_model(self):
        if self.model_save_path is None:
            return

        print(f"Saving model to {self.model_save_path}...")
        dst = os.path.join(self.model_save_path, "model.zip")
        # SB3 models expose .save
        self.model.save(dst)


# ----------------------
#  PARAMETER SEARCH CONFIG
# ----------------------
params = ["tau", "batch_size"]

tau_range = [0.1]
batch_size_range = [32, 64, 128]

ranges = [
    tau_range,
    batch_size_range,
]

defaults = [0.99, 0.0003, 50000, 0.005, 64]


# ----------------------
#  VEC ENV CREATOR
# ----------------------
def make_env(rank, log_dir, render_mode=None):
    def _init():
        env = DroneRacingEnv(render_mode=render_mode)
        monitor_file = os.path.join(log_dir, f"monitor_{rank}.csv")
        return Monitor(env, filename=monitor_file)
    return _init


if __name__ == "__main__":
    # Number of subprocess environments
    num_envs = 16

    # Print GPU/Torch diagnostics and detect device
    print_gpu_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train_best_parameters starting — device={device}, num_envs={num_envs}")

    # Total timesteps per run (adjust as needed)
    total_timesteps = 500_000

    for i in range(len(params)):
        for j in range(len(ranges[i])):

            # Unpack defaults
            gamma, learning_rate, buffer_size, tau, batch_size = defaults

            # Override based on current parameter
            if params[i] == "gamma":
                gamma = ranges[i][j]
            elif params[i] == "learning_rate":
                learning_rate = ranges[i][j]
            elif params[i] == "buffer_size":
                buffer_size = ranges[i][j]
            elif params[i] == "tau":
                tau = ranges[i][j]
            elif params[i] == "batch_size":
                batch_size = ranges[i][j]

            # ----------------------
            #  INIT WANDB
            # ----------------------
            run = wandb.init(
                project="quadai-params-tqc",
                sync_tensorboard=True,
                monitor_gym=True,
                name=f"{params[i]}_{ranges[i][j]}",
                config={
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "buffer_size": buffer_size,
                    "tau": tau,
                    "batch_size": batch_size,
                },
                settings=wandb.Settings(symlink=False),
            )

            # ----------------------
            #  LOG DIR SETUP
            # ----------------------
            log_dir = os.path.join("Drone_Path", "tmp")
            os.makedirs(log_dir, exist_ok=True)

            # Create vectorized environment
            env = SubprocVecEnv([make_env(k, log_dir) for k in range(num_envs)])

            # ----------------------
            #  CREATE TQC MODEL
            # ----------------------
            model = TQC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                gamma=gamma,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                tau=tau,
                batch_size=batch_size,
                device=device,
            )

            # ----------------------
            #  SAVE EVERY N STEPS (per env)
            # ----------------------
            save_every = 100_000 // num_envs

            # ----------------------
            # CALLBACKS
            # ----------------------
            callbacks = [
                CheckpointCallback(
                    save_freq=save_every,
                    save_path=log_dir,
                    name_prefix="rl_model_tqc",
                ),
                WandbNoSymlink(
                    model_save_path=os.path.join("Drone_Path", "models", run.id),
                    model_save_freq=save_every,
                    verbose=2,
                ),
            ]

            # ----------------------
            #  TRAIN
            # ----------------------
            try:
                model.learn(total_timesteps=total_timesteps, callback=callbacks)
            except Exception as e:
                print(f"Training failed: {e}")
            finally:
                env.close()
                run.finish()
