
import numpy as np
import gymnasium as gym
import random
import csv
import time
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, TimeoutError
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import EnvSpec

# Custom CartPole Environment
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
    total_reward = 0
    steps = 0
    pid_theta.reset()
    pid_pos.reset()

    for _ in range(5000):
        x, x_dot, theta, theta_dot = observation
        control_theta = pid_theta.update(theta)
        control_pos = pid_pos.update(x)
        control = control_theta + control_pos
        action = 0 if control < 0 else 1

        reward = 1.0 - abs(theta)
        total_reward += reward

        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return total_reward, steps

def evaluate_pid_multiple_trials(params_trials):
    params, trials = params_trials
    total_reward = 0
    total_steps = 0
    perfect_trials = 0

    for _ in range(trials):
        reward, steps = evaluate_pid(params)
        total_reward += reward
        total_steps += steps
        if steps == 5000:
            perfect_trials += 1

    avg_reward = total_reward / trials
    avg_steps = total_steps / trials
    avg_time_s = avg_steps * 0.02
    return (params, avg_reward, avg_steps, avg_time_s, perfect_trials)

def safe_evaluate_pid(args):
    try:
        with Pool(1) as p:
            result = p.apply_async(evaluate_pid_multiple_trials, args=(args,))
            return result.get(timeout=120)
    except TimeoutError:
        return (args[0], 0, 0, 0.0, 0)
    except Exception as e:
        print(f"Error: {e}")
        return (args[0], 0, 0, 0.0, 0)

POP_SIZE = 150
GENERATIONS = 66
MUTATION_RATE = 0.25
ELITE_SIZE = 10
TRIALS_PER_PID = 8
CHECKPOINT_INTERVAL = 10
INJECT_RANDOM = 10

bounds = [
    (100, 2000), (100, 1700), (2, 10),
    (0.001, 0.1), (0.0, 0.1), (0.0001, 0.01)
]

def random_individual():
    return [random.uniform(low, high) for (low, high) in bounds]

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            low, high = bounds[i]
            if random.random() < 0.15:
                ind[i] = random.uniform(low, high)
            else:
                ind[i] += np.random.normal(0, (high - low) * 0.35)
                ind[i] = np.clip(ind[i], low, high)
    return ind

def crossover(p1, p2):
    return [random.choice(pair) for pair in zip(p1, p2)]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    population = [random_individual() for _ in range(POP_SIZE)]
    total_simulations = 0

    for gen in range(GENERATIONS):
        start_time = time.time()
        results = []

        for ind in tqdm(population):
            result = safe_evaluate_pid((ind, TRIALS_PER_PID))
            results.append(result)

        total_simulations += len(population) * TRIALS_PER_PID
        print(f"Gen {gen+1} | Best Avg Reward: {results[0][1]:.2f} | Total Simulations: {total_simulations}")

        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        best = ranked[0]

        with open("logs/generation_log.csv", "a", newline="") as logfile:
            log = csv.writer(logfile)
            if gen == 0:
                log.writerow(["Generation", "Best_Reward", "Avg_Time_s", "Perfect_Trials"])
            log.writerow([gen+1, best[1], best[3], best[4]])

        with open(f"results/results_gen_{gen}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Kp_theta", "Ki_theta", "Kd_theta",
                "Kp_pos", "Ki_pos", "Kd_pos",
                "Avg_Reward", "Avg_Steps", "Avg_Time_s", "Perfect_Trials"
            ])
            for ind, reward, steps, time_s, perfect in results:
                writer.writerow(ind + [reward, steps, time_s, perfect])

        if gen % CHECKPOINT_INTERVAL == 0:
            with open(f"checkpoints/checkpoint_gen_{gen}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Kp_theta", "Ki_theta", "Kd_theta",
                    "Kp_pos", "Ki_pos", "Kd_pos",
                    "Avg_Reward", "Avg_Steps", "Avg_Time_s", "Perfect_Trials"
                ])
                for ind, reward, steps, time_s, perfect in ranked[:100]:
                    writer.writerow(ind + [reward, steps, time_s, perfect])

        next_gen = [ind for ind, *_ in ranked[:ELITE_SIZE]]
        while len(next_gen) < POP_SIZE - INJECT_RANDOM:
            p1, p2 = random.sample(ranked[:30], 2)
            child = mutate(crossover(p1[0], p2[0]))
            next_gen.append(child)
        next_gen += [random_individual() for _ in range(INJECT_RANDOM)]
        population = next_gen
