import numpy as np
import gymnasium as gym
import random
import csv
import time
import os
from tqdm import tqdm
from multiprocessing import Pool, TimeoutError
from scipy.stats.qmc import LatinHypercube, scale
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import EnvSpec

# Custom CartPole environment
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

# PID Controller
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

# Genetic Algorithm Config
POP_SIZE = 200
GENERATIONS = 80
TRIALS_PER_PID = 10
MUTATION_RATE = 1.0
ELITE_SIZE = 0
INJECT_RANDOM = 50

bounds = [
    (100, 2000), (100, 1700), (2, 10),
    (0.001, 0.1), (0.0, 0.1), (0.0001, 0.01)
]

# Latin Hypercube Sampling
def lhs_population(n_samples, bounds):
    sampler = LatinHypercube(d=len(bounds))
    sample = sampler.random(n=n_samples)
    return scale(sample, [b[0] for b in bounds], [b[1] for b in bounds]).tolist()

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            low, high = bounds[i]
            ind[i] += np.random.normal(0, (high - low) * 0.4)
            ind[i] = np.clip(ind[i], low, high)
    return ind

def crossover(p1, p2):
    return [(a + b) / 2 + np.random.normal(0, abs(a - b) * 0.3) for a, b in zip(p1, p2)]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initial population using Latin Hypercube
    population = lhs_population(POP_SIZE, bounds)
    total_simulations = 0

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1} ---")
        results = []
        for ind in tqdm(population):
            result = safe_evaluate_pid((ind, TRIALS_PER_PID))
            results.append(result)

        total_simulations += len(population) * TRIALS_PER_PID
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        best = ranked[0]

        with open("logs/generation_log.csv", "a", newline="") as logfile:
            log = csv.writer(logfile)
            if gen == 0:
                log.writerow(["Generation", "Best_Score", "Avg_Theta", "Avg_Time", "Perfect_Trials"])
            log.writerow([gen+1, best[1], best[2], best[3], best[4]])

        with open(f"results/results_gen_{gen}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Kp_theta", "Ki_theta", "Kd_theta",
                "Kp_pos", "Ki_pos", "Kd_pos",
                "Score", "Avg_Theta", "Avg_Time_s", "Perfect_Trials"
            ])
            for ind, score, avg_theta, time_s, perfect in results:
                writer.writerow(ind + [score, avg_theta, time_s, perfect])

        # Generate next generation
        next_gen = []
        selection_pool = random.choices(ranked, weights=[r[1] + 1e-6 for r in ranked], k=100)

        while len(next_gen) < POP_SIZE - INJECT_RANDOM:
            p1, p2 = random.sample(selection_pool, 2)
            child = mutate(crossover(p1[0], p2[0]))
            next_gen.append(child)

        next_gen += lhs_population(INJECT_RANDOM, bounds)
        population = next_gen
