import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 30.0  # Pixels per meter

# Physics
GRAVITY = -9.81
MASS = 1.0
TIME_STEP = 0.02

# Drone properties
DRONE_WIDTH = 1.0
DRONE_HEIGHT = 0.25
INERTIA = (1/12) * MASS * (DRONE_WIDTH ** 2) * 1.5  # Lower for better turn authority
MAX_THRUST = 12.0  # Reduced from 20 to slow down acceleration
TORQUE_SCALE = 1.5  # Amplify torque for easier turning
LATERAL_DRAG = 0.3  # Reduced to not stick so much
THRUST_HOVER = (-GRAVITY * MASS) / 2.0

class DroneRacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(DroneRacingEnv, self).__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode == "human" and not PYGAME_AVAILABLE:
            print("Warning: Render mode 'human' requires pygame, which is not installed.")
            self.render_mode = None

        # 1. Action Space: 2 motors [Left, Right] with range [0.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 2. Observation Space: 8 values
        # [angle, angular_vel, vx, vy, dx, dy, sin(rel_angle), cos(rel_angle)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Race management
        self.gates = []          
        self.current_gate_idx = 0 
        self.state = None # [x, y, angle, vx, vy, angular_vel]
        self.prev_dist_to_gate = 0
        self.steps_survived = 0
        self.start_time = 0
        self.current_time = 0
        self.race_completed = False
        self.steps_near_gate = 0  # Counter for loitering detection
        self.total_steps = 0  # Total steps in episode
        self.passed_all_gates = False  # True when all gates passed, waiting for return to start

    def _generate_track(self):
        """Generates a semi-circular track of gates."""
        self.gates = []
        
        # Center of the semi-circle
        center_x = 13.0
        center_y = 10.0
        radius = 8.0
        
        # Generate 5 gates along a semi-circle (from -90 to +90 degrees roughly, or 180 to 0)
        # Let's do a bottom-to-top arc or left-to-right arc.
        # Start at angle PI (left) go to 0 (right)
        
        num_gates = 5
        start_angle = math.pi # 180 degrees (Left)
        end_angle = 0.0       # 0 degrees (Right)
        
        for i in range(num_gates):
            # Interpolate angle
            t = i / (num_gates - 1)
            angle = start_angle + t * (end_angle - start_angle)
            
            gx = center_x + radius * math.cos(angle)
            gy = center_y + radius * math.sin(angle)
            
            self.gates.append(np.array([gx, gy]))

    def _get_obs(self):
        if self.current_gate_idx >= len(self.gates):
            return np.zeros(8, dtype=np.float32) 
            
        target = self.gates[self.current_gate_idx]
        
        # Relative position to target
        dx = target[0] - self.state[0]
        dy = target[1] - self.state[1]
        
        # Relative angle to target
        angle_to_target = math.atan2(dy, dx)
        rel_angle = angle_to_target - self.state[2]
        
        return np.array([
            self.state[2],      # Angle
            self.state[5],      # Angular Velocity
            self.state[3],      # Vx
            self.state[4],      # Vy
            dx,                 # Distance X to gate
            dy,                 # Distance Y to gate
            math.sin(rel_angle), # Sine of relative angle
            math.cos(rel_angle)  # Cosine of relative angle
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate track - gates[0] is start/finish
        self._generate_track()
        
        # Start position is AT the first gate
        start_gate = self.gates[0]
        start_x = start_gate[0]
        start_y = start_gate[1]
        
        self.state = np.array([start_x, start_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Start at gate 1 (skip gate 0), will return to gate 0 at the end
        self.current_gate_idx = 1
        self.steps_survived = 0
        self.start_time = 0
        self.current_time = 0
        self.race_completed = False
        self.steps_near_gate = 0
        self.total_steps = 0
        self.passed_all_gates = False
        
        self.prev_dist_to_gate = np.linalg.norm(self.state[0:2] - self.gates[1])

        if self.render_mode == "human" and self.screen is None:
            self._init_render()
            
        return self._get_obs(), {}

    def step(self, action):
        # --- Physics ---
        
        # Compute thrust from actions. Add a hover trim so `action==0` corresponds
        # to a balanced hover (helps learning because the agent doesn't have to
        # immediately counter gravity at episode start).
        left_thrust = (action[0] * MAX_THRUST) + THRUST_HOVER
        right_thrust = (action[1] * MAX_THRUST) + THRUST_HOVER

        # Clip thrust to valid range [0, MAX_THRUST]
        left_thrust = float(np.clip(left_thrust, 0.0, MAX_THRUST))
        right_thrust = float(np.clip(right_thrust, 0.0, MAX_THRUST))

        angle = self.state[2]
        total_thrust = left_thrust + right_thrust

        # Forces include gravity so the drone will fall when motors are zero
        fx = total_thrust * math.sin(angle)
        fy = total_thrust * math.cos(angle) + (GRAVITY * MASS)
        torque = (right_thrust - left_thrust) * (DRONE_WIDTH / 2.0) * TORQUE_SCALE

        # Accelerations
        ax = fx / MASS
        ay = fy / MASS
        a_ang = torque / INERTIA

        # Integrate velocities (accumulate) so inertia and gravity behave naturally
        self.state[3] += ax * TIME_STEP
        self.state[4] += ay * TIME_STEP
        self.state[5] += a_ang * TIME_STEP

        # Air resistance / Damping (stability) - reduced for better momentum
        self.state[3] *= 0.98  # Less damping for better speed
        self.state[4] *= 0.98
        self.state[5] *= 0.95  # Moderate angular damping

        # Clamp velocities to prevent excessive speeds - increased limits
        self.state[3] = float(np.clip(self.state[3], -8.0, 8.0))  # Doubled from 4.0
        self.state[4] = float(np.clip(self.state[4], -8.0, 8.0))  # Doubled from 4.0
        self.state[5] = float(np.clip(self.state[5], -2.0, 2.0))  # Increased from 1.2

        self.state[0] += self.state[3] * TIME_STEP
        self.state[1] += self.state[4] * TIME_STEP
        self.state[2] += self.state[5] * TIME_STEP
        
        self.steps_survived += 1
        self.total_steps += 1
        
        # Update race time (only if not completed)
        if not self.race_completed:
            if self.start_time == 0:
                self.start_time = time.time()
            self.current_time = time.time() - self.start_time

        # --- Logic & Reward ---
        reward = 0.0
        terminated = False
        truncated = False
        
        # Maximum episode length - give enough time to complete the lap
        MAX_STEPS = 1500  # ~30 seconds - enough time to learn
        if self.total_steps >= MAX_STEPS:
            truncated = True
            reward -= 500.0  # Heavy penalty for timeout
        
        if self.current_gate_idx < len(self.gates):
            target = self.gates[self.current_gate_idx]
            dist = np.linalg.norm(self.state[0:2] - target)
            
            # Progress Reward (Shaping)
            reward += (self.prev_dist_to_gate - dist) * 100.0
            self.prev_dist_to_gate = dist
            
            # Speed bonus
            velocity = np.sqrt(self.state[3]**2 + self.state[4]**2)
            reward += velocity * 0.5
            
            # Gate Crossing
            if dist < 0.5: 
                # Speed bonus for fast gate crossings
                speed_bonus = max(0, 300 - self.total_steps * 0.5)
                reward += 300.0 + speed_bonus
                self.steps_near_gate = 0
                
                # Check if we just completed the race (passed gate 0 after all others)
                if self.current_gate_idx == 0 and self.passed_all_gates:
                    # Finished! Bonus for fast completion
                    time_bonus = max(0, 5000 - self.total_steps * 8)
                    reward += 2000.0 + time_bonus
                    self.race_completed = True
                    terminated = True
                else:
                    # Move to next gate
                    self.current_gate_idx += 1
                    
                    # After passing gate 4 (index becomes 5), go back to gate 0
                    if self.current_gate_idx >= len(self.gates):
                        self.current_gate_idx = 0  # Now target is gate 0 (start/finish)
                        self.passed_all_gates = True  # Mark that we need to return
                    
                    self.prev_dist_to_gate = np.linalg.norm(self.state[0:2] - self.gates[self.current_gate_idx])

        # Time penalty
        reward -= 0.1
        # Removed stability penalty - we want speed not stability
        
        # Crash / Out of bounds
        world_h = SCREEN_HEIGHT/SCALE
        world_w = SCREEN_WIDTH/SCALE
        
        # Floor or Ceiling collision - עונש מופחת
        if self.state[1] < 0.5 or self.state[1] > world_h - 0.5:
            terminated = True
            reward -= 20.0  # הופחת מ-50 ל-20
            
        # Side walls - עונש מופחת
        if self.state[0] < 0.5 or self.state[0] > world_w - 0.5:
             terminated = True
             reward -= 20.0  # הופחת מ-50 ל-20

        # Too far from target (straying) - מרחק מוגדל ועונש מופחת
        if self.current_gate_idx < len(self.gates):
            target = self.gates[self.current_gate_idx]
            if np.linalg.norm(self.state[0:2] - target) > 35.0:  # הגדלנו מ-25 ל-35
                terminated = True 
                reward -= 20.0  # הופחת מ-50 ל-20

        if self.render_mode == "human" and PYGAME_AVAILABLE:
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _init_render(self):
        if not PYGAME_AVAILABLE: return
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

    def render(self):
        if not PYGAME_AVAILABLE: return
        if self.screen is None: self._init_render()
        
        self.screen.fill((30, 30, 30))
        
        def to_pix(x, y):
            return int(x * SCALE), int(SCREEN_HEIGHT - (y * SCALE))
        
        # Draw Gates
        for i, g in enumerate(self.gates):
            pos = to_pix(g[0], g[1])
            
            if self.race_completed:
                # All gates green when finished
                color = (0, 255, 0)
            elif i == self.current_gate_idx:
                # Current target gate - bright green
                color = (0, 255, 0)
            elif i == 0:
                # Start/finish gate - yellow (will turn green at end)
                if self.current_gate_idx == 0:
                    color = (0, 255, 0)  # Green when it's the target
                else:
                    color = (255, 255, 0)  # Yellow when waiting for return
            else:
                # Other gates - check if passed
                # Gates 1-4: passed if current_gate_idx > i, OR if we're back at 0
                if self.current_gate_idx == 0 or self.current_gate_idx > i:
                    color = (100, 100, 100)  # Passed - gray
                else:
                    color = (255, 0, 0)  # Not yet - red
            
            pygame.draw.circle(self.screen, color, pos, 15, 3)
            
            # REMOVED: Guide line to active gate (the "web")

        # Draw Drone
        pos = to_pix(self.state[0], self.state[1])
        
        # Drone Body (Central Hub)
        pygame.draw.circle(self.screen, (50, 50, 200), pos, 10)
        
        # Arms
        arm_length_pix = int((DRONE_WIDTH / 2.0) * SCALE)
        angle = self.state[2]
        
        # Left Arm End
        lx = pos[0] - math.cos(angle) * arm_length_pix
        ly = pos[1] - math.sin(angle) * arm_length_pix
        # Right Arm End
        rx = pos[0] + math.cos(angle) * arm_length_pix
        ry = pos[1] + math.sin(angle) * arm_length_pix
        
        pygame.draw.line(self.screen, (200, 200, 200), (lx, ly), (rx, ry), 4)
        
        # Motors (Propellers)
        # Left Motor
        pygame.draw.circle(self.screen, (0, 255, 255), (int(lx), int(ly)), 6)
        # Right Motor
        pygame.draw.circle(self.screen, (0, 255, 255), (int(rx), int(ry)), 6)
        
        # Orientation Indicator (small line on body)
        head_x = pos[0] + math.sin(angle) * 12
        head_y = pos[1] - math.cos(angle) * 12
        pygame.draw.line(self.screen, (255, 255, 0), pos, (head_x, head_y), 2)

        # Info
        info = f"Gate: {self.current_gate_idx}/{len(self.gates)}"
        surf = self.font.render(info, True, (255, 255, 255))
        self.screen.blit(surf, (10, 10))
        
        # Timer
        if self.race_completed:
            time_info = f"Time: {self.current_time:.2f}s (FINISHED!)"
            time_color = (0, 255, 0)  # Green
        else:
            time_info = f"Time: {self.current_time:.2f}s"
            time_color = (255, 255, 255)  # White
        time_surf = self.font.render(time_info, True, time_color)
        self.screen.blit(time_surf, (10, 40))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if PYGAME_AVAILABLE and self.screen is not None:
            pygame.quit()