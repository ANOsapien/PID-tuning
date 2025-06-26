import numpy as np
import gymnasium as gym
import pandas as pd
import time
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import EnvSpec
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class LongCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.render_mode = "human"
        self.spec = EnvSpec("Custom-CartPole-v1")
        self.gravity = 9.82
        self.masscart = 0.5
        self.masspole = 0.5
        self.friction_cart = 0.1
        self.friction_pole = 0.0
        self.length = 0.6  # Full pole length is 1.2 m (this is half-length from pivot)
        self.force_mag = 10.0
        self.tau = 0.02
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.max_episode_steps = 5000

def make_long_cartpole_env():
    return TimeLimit(LongCartPoleEnv(), max_episode_steps=3000)

# PID controller
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt=0.02):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# Simulate a single PID set
def run_simulation(params, render=True):
    Kp_theta, Ki_theta, Kd_theta, Kp_pos, Ki_pos, Kd_pos = params
    pid_theta = PIDController(Kp_theta, Ki_theta, Kd_theta)
    pid_pos = PIDController(Kp_pos, Ki_pos, Kd_pos)

    env = make_long_cartpole_env()
    observation, _ = env.reset()
    pid_theta.reset()
    pid_pos.reset()

    steps = 0
    for _ in range(5000):
        if render:
            env.render()
            time.sleep(0.02)

        x, x_dot, theta, theta_dot = observation
        control_theta = pid_theta.update(theta)
        control_pos = pid_pos.update(x)
        control = control_theta + control_pos

        action = 0 if control < 0 else 1
        observation, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    env.close()
    return steps * 0.02  # seconds

# Load top parameters
csv_file = "top_100_pid_parameters.csv"
df = pd.read_csv(csv_file)

# Simulate top N
N = 10
for i in range(N):
    print(f"\n--- Simulation {i+1}/{N} ---")
    row = df.iloc[i]
    params = row[["Kp_theta", "Ki_theta", "Kd_theta", "Kp_pos", "Ki_pos", "Kd_pos"]].values
    duration = run_simulation(params, render=True)
    print(f"PID #{i+1} stayed upright for {duration:.2f} seconds")
