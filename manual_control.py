"""Manual control for DroneRacingEnv using keyboard (pygame).

Keys:
  W / Up: increase both motors
  S / Down: decrease both motors
  A / Left: increase left motor, decrease right motor (turn left)
  D / Right: decrease left motor, increase right motor (turn right)
  Space: kill motors (set both to 0)
  R: reset environment
  Esc or window close: quit

Run:
  python manual_control.py

Note: requires `pygame`.
"""

import sys
import os
import time
import numpy as np

try:
    import pygame
except Exception:
    print("pygame is required for manual control. Install with: pip install pygame")
    sys.exit(1)

from src.environment import DroneRacingEnv


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def main():
    env = DroneRacingEnv(render_mode="human")
    obs, _ = env.reset()

    # Control variables: throttle around 0 = hover, yaw is differential command
    throttle = 0.0  # -1 .. 1, 0 = hover
    yaw = 0.0       # -yaw_limit .. yaw_limit
    yaw_limit = 0.35
    delta = 0.05
    # yaw_decay: multiplicative decay applied each frame so yaw returns toward 0
    # when keys are not pressed. Value in (0,1], lower = faster decay.

    print("Manual control started. Keys: W/S Up/Down, A/D Left/Right, Space kill, R reset, Esc quit")

    clock = pygame.time.Clock()
    running = True

    # use the env's framerate if available
    fps = env.metadata.get("render_fps", 60)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    throttle = 0.0
                    yaw = 0.0
                elif event.key == pygame.K_SPACE:
                    throttle = 0.0
                    yaw = 0.0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Poll key state for continuous control (holding keys changes values each frame)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle = clamp(throttle + delta, -1.0, 1.0)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            throttle = clamp(throttle - delta, -1.0, 1.0)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            yaw = float(np.clip(yaw - delta, -yaw_limit, yaw_limit))
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            yaw = float(np.clip(yaw + delta, -yaw_limit, yaw_limit))
        # Note: R and Space handled above in KEYDOWN for immediate reset/kill

        # Build action from throttle and yaw. Left/right action in [-1,1], where 0 is hover.
        # Prevent throttle from saturating fully when yaw is commanded so yaw still has authority.
        throttle_max_when_yaw = 0.7
        if abs(yaw) > 1e-3:
            eff_throttle = min(throttle, throttle_max_when_yaw)
        else:
            eff_throttle = throttle

        left = float(np.clip(eff_throttle - yaw, -1.0, 1.0))
        right = float(np.clip(eff_throttle + yaw, -1.0, 1.0))
        action = np.array([left, right], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Print a short status line
        gate_info = f"Gate {env.current_gate_idx}/{len(env.gates)}"
        print(f"thr={throttle:.2f} yaw={yaw:.2f} act={[round(float(x),3) for x in action]} rew={reward:.3f} {gate_info}", end="\r")

        if terminated or truncated:
            print()  # newline after the status line
            print(f"Episode ended (terminated={terminated}, truncated={truncated}). Resetting...")
            obs, _ = env.reset()
            throttle = 0.0
            yaw = 0.0


        # Let the env render and keep framerate
        clock.tick(fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
