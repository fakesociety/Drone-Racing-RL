"""Manual control for DroneRacingEnv using keyboard (pygame).

Controls:
    W / Up Arrow    : Increase throttle
    S / Down Arrow  : Decrease throttle  
    A / Left Arrow  : Yaw left (turn counter-clockwise)
    D / Right Arrow : Yaw right (turn clockwise)
    Space           : Kill motors (emergency stop)
    R               : Reset environment
    Escape          : Quit

Usage:
    python scripts/manual_control.py

Requirements:
    - pygame installed
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

try:
    import pygame
except ImportError:
    print("❌ pygame is required for manual control.")
    print("   Install with: pip install pygame")
    sys.exit(1)

from src.environment import DroneRacingEnv


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp a value to the specified range."""
    return max(min_val, min(max_val, value))


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def main():
    """Run manual control mode."""
    
    print("="*60)
    print("  DRONE RACING - MANUAL CONTROL")
    print("="*60)
    print("\nControls:")
    print("  W/S or Up/Down  : Throttle")
    print("  A/D or Left/Right : Yaw")
    print("  Space : Kill motors")
    print("  R : Reset")
    print("  Escape : Quit")
    print("="*60 + "\n")
    
    # Create environment
    env = DroneRacingEnv(render_mode="human")
    obs, _ = env.reset()

    # Control parameters
    throttle = 0.0      # -1 to 1, where 0 = hover
    yaw = 0.0           # Differential thrust for turning
    yaw_limit = 0.35    # Maximum yaw command
    delta = 0.05        # Control sensitivity
    
    # Throttle limiting during yaw (preserves yaw authority)
    throttle_max_when_yaw = 0.7

    clock = pygame.time.Clock()
    fps = env.metadata.get("render_fps", 60)
    running = True

    while running:
        # ---------------------------------------------------------------------
        # EVENT HANDLING
        # ---------------------------------------------------------------------
        
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
                    print("\n🔄 Environment reset!")
                    
                elif event.key == pygame.K_SPACE:
                    throttle = 0.0
                    yaw = 0.0
                    print("\n⚠️ Motors killed!")

        # ---------------------------------------------------------------------
        # CONTINUOUS KEY POLLING
        # ---------------------------------------------------------------------
        
        keys = pygame.key.get_pressed()
        
        # Throttle control
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle = clamp(throttle + delta)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            throttle = clamp(throttle - delta)
            
        # Yaw control
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            yaw = float(np.clip(yaw - delta, -yaw_limit, yaw_limit))
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            yaw = float(np.clip(yaw + delta, -yaw_limit, yaw_limit))

        # ---------------------------------------------------------------------
        # BUILD ACTION
        # ---------------------------------------------------------------------
        
        # Limit throttle during yaw for better turn authority
        if abs(yaw) > 1e-3:
            effective_throttle = min(throttle, throttle_max_when_yaw)
        else:
            effective_throttle = throttle

        # Convert to motor commands
        left_motor = float(np.clip(effective_throttle - yaw, -1.0, 1.0))
        right_motor = float(np.clip(effective_throttle + yaw, -1.0, 1.0))
        action = np.array([left_motor, right_motor], dtype=np.float32)
        
        # ---------------------------------------------------------------------
        # STEP ENVIRONMENT
        # ---------------------------------------------------------------------
        
        obs, reward, terminated, truncated, _ = env.step(action)

        # Print status
        gate_info = f"Gate {env.current_gate_idx}/{len(env.gates)}"
        print(f"Throttle={throttle:+.2f}  Yaw={yaw:+.2f}  "
              f"Action=[{left_motor:+.2f}, {right_motor:+.2f}]  "
              f"Reward={reward:+.1f}  {gate_info}    ", end="\r")

        # Handle episode end
        if terminated or truncated:
            print("\n")
            if env._race_completed:
                print(f"🏁 LAP COMPLETE! Time: {env._current_time:.2f}s")
            else:
                print(f"💥 Episode ended (terminated={terminated}, truncated={truncated})")
            print("Press R to reset or Escape to quit.\n")
            
            # Wait for reset
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key == pygame.K_r:
                            obs, _ = env.reset()
                            throttle = 0.0
                            yaw = 0.0
                            waiting = False
                            print("🔄 Environment reset!\n")
                clock.tick(30)

        clock.tick(fps)

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    
    env.close()
    pygame.quit()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
