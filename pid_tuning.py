import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
import csv
from tqdm import tqdm
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import EnvSpec

# Custom environment with real cartpole physics
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

def make_long_cartpole_env():
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

# One run of the simulation
def evaluate_pid(params):
    Kp_theta, Ki_theta, Kd_theta, Kp_pos, Ki_pos, Kd_pos = params
    pid_theta = PIDController(Kp_theta, Ki_theta, Kd_theta)
    pid_pos = PIDController(Kp_pos, Ki_pos, Kd_pos)

    env = make_long_cartpole_env()
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
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return total_reward, steps

# Run multiple trials
def evaluate_pid_multiple_trials(params, trials=20):
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
    return avg_reward, avg_steps, perfect_trials

# Genetic Algorithm config
POP_SIZE = 150
GENERATIONS = 300
MUTATION_RATE = 0.05
ELITE_SIZE = 10
TRIALS_PER_PID = 20

bounds = [
    (300, 2000),    # Kp_theta
    (300, 1700),    # Ki_theta
    (2, 10),        # Kd_theta
    (0.001, 0.1),   # Kp_pos
    (0.0, 0.1),     # Ki_pos
    (0.0001, 0.005) # Kd_pos
]

def random_individual():
    return [random.uniform(low, high) for (low, high) in bounds]

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            low, high = bounds[i]
            individual[i] += np.random.normal(0, (high - low) * 0.1)
            individual[i] = np.clip(individual[i], low, high)
    return individual

def crossover(parent1, parent2):
    return [random.choice(gene_pair) for gene_pair in zip(parent1, parent2)]

# Run GA
if __name__ == "__main__":
    population = [random_individual() for _ in range(POP_SIZE)]
    all_results = []
    best_scores = []

    for gen in range(GENERATIONS):
        results = [evaluate_pid_multiple_trials(ind, trials=TRIALS_PER_PID) for ind in tqdm(population)]
        avg_rewards = [score for score, _, _ in results]
        avg_durations = [steps for _, steps, _ in results]
        perfect_trials = [p for _, _, p in results]
        avg_seconds = [round(s * 0.02, 2) for s in avg_durations]

        ranked = sorted(zip(population, avg_rewards, avg_durations, avg_seconds, perfect_trials),
                        key=lambda x: x[1], reverse=True)

        best_scores.append(ranked[0][1])
        all_results.extend(ranked)

        print(f"Gen {gen+1}: Best Avg Reward = {ranked[0][1]:.2f} | Avg Time = {ranked[0][3]}s | Perfect Trials = {ranked[0][4]}")

        next_gen = [ind for ind, _, _, _, _ in ranked[:ELITE_SIZE]]
        while len(next_gen) < POP_SIZE:
            p1, p2 = random.sample(ranked[:30], 2)
            child = crossover(p1[0], p2[0])
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    # Save top 100 robust PIDs
    top_100 = sorted(all_results, key=lambda x: x[1], reverse=True)[:1000]
    with open("pid_parameters.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Kp_theta", "Ki_theta", "Kd_theta",
            "Kp_pos", "Ki_pos", "Kd_pos",
            "Avg_Reward", "Avg_Steps", "Avg_Time_s", "Perfect_Trials"
        ])
        for ind, reward, steps, seconds, perfect in top_100:
            writer.writerow(ind + [reward, steps, seconds, perfect])

    # Plot
    plt.figure()
    plt.plot(best_scores)
    plt.title("Best Avg Reward per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("genetic_pid_cartpole_results.png")
