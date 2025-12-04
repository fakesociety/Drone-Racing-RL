"""
Drone Racing Environment for Reinforcement Learning.

A custom Gymnasium environment simulating a 2D racing drone navigating through gates.
The drone uses differential thrust for control (left/right motors) and must complete
a circular track by passing through all gates and returning to the start.

Environment Details:
- Physics: 2D with gravity, thrust, torque, and velocity damping
- Control: Two continuous actions [-1, 1] for left/right motor thrust
- Observation: 8D vector with drone state and relative position to target gate
- Reward: Progress-based shaping with gate crossing bonuses

Author: Your Name
License: MIT
"""

import math
import time
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 30.0  # Pixels per meter

# Physics
GRAVITY = -9.81  # m/s^2
MASS = 1.0       # kg
TIME_STEP = 0.02 # seconds (50 Hz simulation)

# Drone Properties
DRONE_WIDTH = 1.0   # meters (motor-to-motor distance)
DRONE_HEIGHT = 0.25 # meters
INERTIA = (1/12) * MASS * (DRONE_WIDTH ** 2) * 1.5  # Moment of inertia
MAX_THRUST = 12.0   # Newtons per motor
TORQUE_SCALE = 1.5  # Torque multiplier for better turning
THRUST_HOVER = (-GRAVITY * MASS) / 2.0  # Thrust per motor for hovering

# Velocity Limits
MAX_LINEAR_VELOCITY = 8.0   # m/s
MAX_ANGULAR_VELOCITY = 2.0  # rad/s

# Damping Coefficients
LINEAR_DAMPING = 0.98
ANGULAR_DAMPING = 0.95

# Track Parameters
TRACK_CENTER_X = 13.0
TRACK_CENTER_Y = 10.0
TRACK_RADIUS = 8.0
NUM_GATES = 5

# Episode Limits
MAX_EPISODE_STEPS = 1500  # ~30 seconds
GATE_CROSSING_DISTANCE = 0.5  # meters
MAX_DISTANCE_FROM_GATE = 35.0  # meters


class DroneRacingEnv(gym.Env):
    """
    A Gymnasium environment for training a drone to race through gates.
    
    The drone starts at gate 0, must pass through gates 1-4 in order,
    then return to gate 0 to complete the lap.
    
    Observation Space (8D):
        - angle: Current drone angle (radians)
        - angular_velocity: Angular velocity (rad/s)
        - vx: Horizontal velocity (m/s)
        - vy: Vertical velocity (m/s)
        - dx: Relative X distance to target gate (m)
        - dy: Relative Y distance to target gate (m)
        - sin_rel_angle: Sine of angle to target
        - cos_rel_angle: Cosine of angle to target
    
    Action Space (2D):
        - left_motor: Left motor thrust command [-1, 1]
        - right_motor: Right motor thrust command [-1, 1]
        
        Note: 0 corresponds to hover thrust, not zero thrust.
    
    Rewards:
        - Progress: (previous_dist - current_dist) * 100
        - Velocity bonus: speed * 0.5
        - Gate crossing: 300 + speed_bonus
        - Finish: 2000 + time_bonus
        - Time penalty: -0.1 per step
        - Crash penalty: -20
        - Timeout penalty: -500
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the drone racing environment.
        
        Args:
            render_mode: "human" for pygame visualization, "rgb_array" for frame output
        """
        super().__init__()
        
        self.render_mode = render_mode
        self._screen = None
        self._clock = None
        self._font = None

        if self.render_mode == "human" and not PYGAME_AVAILABLE:
            print("Warning: Render mode 'human' requires pygame. Falling back to None.")
            self.render_mode = None

        # Action Space: 2 motors with continuous control
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )

        # Observation Space: 8D state vector
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,), 
            dtype=np.float32
        )

        # State variables
        self._gates: list = []
        self._state: np.ndarray = np.zeros(6, dtype=np.float32)  # [x, y, angle, vx, vy, angular_vel]
        self._current_gate_idx: int = 0
        self._prev_dist_to_gate: float = 0.0
        self._total_steps: int = 0
        self._start_time: float = 0.0
        self._current_time: float = 0.0
        self._race_completed: bool = False
        self._passed_all_gates: bool = False

    @property
    def current_gate_idx(self) -> int:
        """Current target gate index."""
        return self._current_gate_idx
    
    @property
    def gates(self) -> list:
        """List of gate positions."""
        return self._gates

    def _generate_track(self) -> None:
        """Generate a semi-circular track of gates."""
        self._gates = []
        
        start_angle = math.pi  # 180 degrees (left side)
        end_angle = 0.0        # 0 degrees (right side)
        
        for i in range(NUM_GATES):
            t = i / (NUM_GATES - 1)
            angle = start_angle + t * (end_angle - start_angle)
            
            gate_x = TRACK_CENTER_X + TRACK_RADIUS * math.cos(angle)
            gate_y = TRACK_CENTER_Y + TRACK_RADIUS * math.sin(angle)
            
            self._gates.append(np.array([gate_x, gate_y]))

    def _get_observation(self) -> np.ndarray:
        """
        Compute the observation vector.
        
        Returns:
            8D numpy array with drone state and relative gate position
        """
        if self._current_gate_idx >= len(self._gates):
            return np.zeros(8, dtype=np.float32)
            
        target = self._gates[self._current_gate_idx]
        
        # Relative position to target
        dx = target[0] - self._state[0]
        dy = target[1] - self._state[1]
        
        # Relative angle to target
        angle_to_target = math.atan2(dy, dx)
        rel_angle = angle_to_target - self._state[2]
        
        return np.array([
            self._state[2],       # Angle
            self._state[5],       # Angular velocity
            self._state[3],       # Vx
            self._state[4],       # Vy
            dx,                   # Distance X to gate
            dy,                   # Distance Y to gate
            math.sin(rel_angle),  # Sine of relative angle
            math.cos(rel_angle)   # Cosine of relative angle
        ], dtype=np.float32)

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
            
        Returns:
            observation: Initial observation
            info: Empty info dictionary
        """
        super().reset(seed=seed)
        
        # Generate track
        self._generate_track()
        
        # Start at gate 0
        start_pos = self._gates[0]
        self._state = np.array([
            start_pos[0], start_pos[1],  # Position
            0.0,                          # Angle
            0.0, 0.0,                     # Velocity
            0.0                           # Angular velocity
        ], dtype=np.float32)
        
        # Race state: start targeting gate 1, return to gate 0 to finish
        self._current_gate_idx = 1
        self._total_steps = 0
        self._start_time = 0.0
        self._current_time = 0.0
        self._race_completed = False
        self._passed_all_gates = False
        self._prev_dist_to_gate = np.linalg.norm(self._state[0:2] - self._gates[1])

        if self.render_mode == "human" and self._screen is None:
            self._init_render()
            
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: 2D array [left_motor, right_motor] in range [-1, 1]
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: True if episode ended (crash, finish)
            truncated: True if episode truncated (timeout)
            info: Empty info dictionary
        """
        # =========================
        # PHYSICS SIMULATION
        # =========================
        
        # Convert actions to thrust (with hover trim)
        left_thrust = (action[0] * MAX_THRUST) + THRUST_HOVER
        right_thrust = (action[1] * MAX_THRUST) + THRUST_HOVER
        left_thrust = float(np.clip(left_thrust, 0.0, MAX_THRUST))
        right_thrust = float(np.clip(right_thrust, 0.0, MAX_THRUST))

        angle = self._state[2]
        total_thrust = left_thrust + right_thrust

        # Force calculations
        fx = total_thrust * math.sin(angle)
        fy = total_thrust * math.cos(angle) + (GRAVITY * MASS)
        torque = (right_thrust - left_thrust) * (DRONE_WIDTH / 2.0) * TORQUE_SCALE

        # Accelerations
        ax = fx / MASS
        ay = fy / MASS
        alpha = torque / INERTIA

        # Velocity integration
        self._state[3] += ax * TIME_STEP
        self._state[4] += ay * TIME_STEP
        self._state[5] += alpha * TIME_STEP

        # Damping
        self._state[3] *= LINEAR_DAMPING
        self._state[4] *= LINEAR_DAMPING
        self._state[5] *= ANGULAR_DAMPING

        # Velocity clamping
        self._state[3] = float(np.clip(self._state[3], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY))
        self._state[4] = float(np.clip(self._state[4], -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY))
        self._state[5] = float(np.clip(self._state[5], -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY))

        # Position integration
        self._state[0] += self._state[3] * TIME_STEP
        self._state[1] += self._state[4] * TIME_STEP
        self._state[2] += self._state[5] * TIME_STEP
        
        self._total_steps += 1
        
        # Update race time
        if not self._race_completed:
            if self._start_time == 0:
                self._start_time = time.time()
            self._current_time = time.time() - self._start_time

        # =========================
        # REWARD CALCULATION
        # =========================
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Timeout check
        if self._total_steps >= MAX_EPISODE_STEPS:
            truncated = True
            reward -= 500.0
        
        # Gate progress reward
        if self._current_gate_idx < len(self._gates):
            target = self._gates[self._current_gate_idx]
            dist = np.linalg.norm(self._state[0:2] - target)
            
            # Progress reward (shaping)
            reward += (self._prev_dist_to_gate - dist) * 100.0
            self._prev_dist_to_gate = dist
            
            # Speed bonus
            velocity = np.sqrt(self._state[3]**2 + self._state[4]**2)
            reward += velocity * 0.5
            
            # Gate crossing
            if dist < GATE_CROSSING_DISTANCE:
                speed_bonus = max(0, 300 - self._total_steps * 0.5)
                reward += 300.0 + speed_bonus
                
                # Check for race completion
                if self._current_gate_idx == 0 and self._passed_all_gates:
                    time_bonus = max(0, 5000 - self._total_steps * 8)
                    reward += 2000.0 + time_bonus
                    self._race_completed = True
                    terminated = True
                else:
                    # Advance to next gate
                    self._current_gate_idx += 1
                    
                    # After gate 4, return to gate 0
                    if self._current_gate_idx >= len(self._gates):
                        self._current_gate_idx = 0
                        self._passed_all_gates = True
                    
                    self._prev_dist_to_gate = np.linalg.norm(
                        self._state[0:2] - self._gates[self._current_gate_idx]
                    )

        # Time penalty
        reward -= 0.1
        
        # =========================
        # BOUNDARY CHECKS
        # =========================
        
        world_h = SCREEN_HEIGHT / SCALE
        world_w = SCREEN_WIDTH / SCALE
        
        # Floor/ceiling collision
        if self._state[1] < 0.5 or self._state[1] > world_h - 0.5:
            terminated = True
            reward -= 20.0
            
        # Wall collision
        if self._state[0] < 0.5 or self._state[0] > world_w - 0.5:
            terminated = True
            reward -= 20.0

        # Too far from target
        if self._current_gate_idx < len(self._gates):
            target = self._gates[self._current_gate_idx]
            if np.linalg.norm(self._state[0:2] - target) > MAX_DISTANCE_FROM_GATE:
                terminated = True
                reward -= 20.0

        if self.render_mode == "human" and PYGAME_AVAILABLE:
            self.render()
            
        return self._get_observation(), reward, terminated, truncated, {}

    def _init_render(self) -> None:
        """Initialize pygame rendering."""
        if not PYGAME_AVAILABLE:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Drone Racing - TQC Agent")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 24)

    def render(self) -> None:
        """Render the current environment state."""
        if not PYGAME_AVAILABLE:
            return
        if self._screen is None:
            self._init_render()
        
        self._screen.fill((30, 30, 30))
        
        def to_pixels(x: float, y: float) -> Tuple[int, int]:
            return int(x * SCALE), int(SCREEN_HEIGHT - (y * SCALE))
        
        # Draw gates
        for i, gate in enumerate(self._gates):
            pos = to_pixels(gate[0], gate[1])
            
            if self._race_completed:
                color = (0, 255, 0)  # All green when finished
            elif i == self._current_gate_idx:
                color = (0, 255, 0)  # Target gate - green
            elif i == 0:
                color = (255, 255, 0) if self._current_gate_idx != 0 else (0, 255, 0)
            else:
                # Check if passed
                if self._current_gate_idx == 0 or self._current_gate_idx > i:
                    color = (100, 100, 100)  # Passed - gray
                else:
                    color = (255, 0, 0)  # Not yet - red
            
            pygame.draw.circle(self._screen, color, pos, 15, 3)

        # Draw drone
        pos = to_pixels(self._state[0], self._state[1])
        angle = self._state[2]
        
        # Body
        pygame.draw.circle(self._screen, (50, 50, 200), pos, 10)
        
        # Arms
        arm_length = int((DRONE_WIDTH / 2.0) * SCALE)
        lx = pos[0] - math.cos(angle) * arm_length
        ly = pos[1] - math.sin(angle) * arm_length
        rx = pos[0] + math.cos(angle) * arm_length
        ry = pos[1] + math.sin(angle) * arm_length
        
        pygame.draw.line(self._screen, (200, 200, 200), (lx, ly), (rx, ry), 4)
        
        # Motors
        pygame.draw.circle(self._screen, (0, 255, 255), (int(lx), int(ly)), 6)
        pygame.draw.circle(self._screen, (0, 255, 255), (int(rx), int(ry)), 6)
        
        # Direction indicator
        head_x = pos[0] + math.sin(angle) * 12
        head_y = pos[1] - math.cos(angle) * 12
        pygame.draw.line(self._screen, (255, 255, 0), pos, (head_x, head_y), 2)

        # HUD
        gate_text = f"Gate: {self._current_gate_idx}/{len(self._gates)}"
        self._screen.blit(self._font.render(gate_text, True, (255, 255, 255)), (10, 10))
        
        if self._race_completed:
            time_text = f"Time: {self._current_time:.2f}s (FINISHED!)"
            time_color = (0, 255, 0)
        else:
            time_text = f"Time: {self._current_time:.2f}s"
            time_color = (255, 255, 255)
        self._screen.blit(self._font.render(time_text, True, time_color), (10, 40))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Clean up pygame resources."""
        if PYGAME_AVAILABLE and self._screen is not None:
            pygame.quit()
            self._screen = None
