import numpy as np
import gymnasium as gym
import csv
import os
import time
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, TimeoutError
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import EnvSpec

# --- Custom CartPole ---
class LongCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.spec = EnvSpec("Custom-CartPole-v1")
        self.gravity = 9.82
        self.masscart = 0.5
        self.masspole = 0.5
        self.friction_cart = 0.1
        self.friction_pole = 0.0
        self.length = 0.6
        self.force_mag = 10.0
        self.tau = 0.02
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.max_episode_steps = 5000

def make_env():
    return TimeLimit(LongCartPoleEnv(), max_episode_steps=5000)

# --- PID Controller ---
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

# --- Evaluation Functions ---
def evaluate_pid(params):
    Kp_theta, Ki_theta, Kd_theta, Kp_pos, Ki_pos, Kd_pos = params
    pid_theta = PIDController(Kp_theta, Ki_theta, Kd_theta)
    pid_pos = PIDController(Kp_pos, Ki_pos, Kd_pos)

    env = make_env()
    observation, _ = env.reset()
    pid_theta.reset()
    pid_pos.reset()

    total_theta = 0.0
    steps = 0

    for _ in range(5000):
        x, x_dot, theta, theta_dot = observation
        control_theta = pid_theta.update(theta)
        control_pos = pid_pos.update(x)
        control = control_theta + control_pos
        action = 0 if control < 0 else 1

        total_theta += abs(theta)
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    score = 0.5 * (1 - total_theta / steps) + 0.5 * (steps / 5000)
    return score, total_theta / steps, steps

def evaluate_pid_multiple_trials(params_trials):
    params, trials = params_trials
    total_score, total_theta, total_steps = 0, 0, 0
    perfect_trials = 0

    for _ in range(trials):
        score, avg_theta, steps = evaluate_pid(params)
        total_score += score
        total_theta += avg_theta
        total_steps += steps
        if steps == 5000:
            perfect_trials += 1

    avg_score = total_score / trials
    avg_theta = total_theta / trials
    avg_time_s = (total_steps * 0.02) / trials
    return (params, avg_score, avg_theta, avg_time_s, perfect_trials)

def safe_evaluate_pid(args):
    try:
        with Pool(1) as p:
            result = p.apply_async(evaluate_pid_multiple_trials, args=(args,))
            return result.get(timeout=100)
    except TimeoutError:
        return (args[0], 0, 999, 0.0, 0)
    except Exception as e:
        print(f"Error: {e}")
        return (args[0], 0, 999, 0.0, 0)

# --- GRID CONFIG ---
TRIALS_PER_PID = 5
GRID_STEPS = [10, 10, 5, 10, 10, 2]  # Total = 100000 combinations

bounds = [
    (-300, 300),     # Kp_theta
    (-3000, 3000),   # Ki_theta
    (0, 100),        # Kd_theta
    (-1.0, 1.0),     # Kp_pos
    (-1.0, 1.0),     # Ki_pos
    (0.0, 0.5)       # Kd_pos
]

# Build grid
param_grids = [
    np.linspace(start, end, num=steps).tolist()
    for (start, end), steps in zip(bounds, GRID_STEPS)
]
grid_combinations = list(product(*param_grids))

# Save paths
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
result_file = os.path.join(results_dir, "grid_search_results.csv")

# Create CSV header
if not os.path.exists(result_file):
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Kp_theta", "Ki_theta", "Kd_theta",
            "Kp_pos", "Ki_pos", "Kd_pos",
            "Score", "Avg_Theta", "Avg_Time_s", "Perfect_Trials"
        ])

# --- CHUNKED EVALUATION LOOP ---
CHUNK_SIZE = 1000
total = len(grid_combinations)

print(f"\n Starting Grid Search: {total} PID combinations × {TRIALS_PER_PID} trials each")

for i in range(0, total, CHUNK_SIZE):
    chunk = grid_combinations[i:i + CHUNK_SIZE]
    print(f"\n🔁 Evaluating chunk {i} to {i + len(chunk) - 1}...")

    results = []
    for params in tqdm(chunk):
        result = safe_evaluate_pid((list(params), TRIALS_PER_PID))
        results.append(result)

    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        for ind, score, avg_theta, time_s, perfect in results:
            writer.writerow(ind + [score, avg_theta, time_s, perfect])

print("\nGrid Search Complete. Results saved to:", result_file)
