import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from invertedpendulummdp import InvertedPendulum

class PrioritizedSweepingAgent:

    def __init__(self, env, gamma=0.99, theta=1e-3, max_iterations=500, updates_per_iter=500):
        self.env = env
        self.gamma = gamma
        self.model = {}
        self.states = [(i,j) for i in range(self.env.NUM_OMEGA_BINS) for j in range(self.env.NUM_OMEGA_DOT_BINS)]
        self.build_model()
        self.V = {s:0.0 for s in self.states}
        self.predecessors = {s:set() for s in self.states}
        for s in self.states:
            for a_i in range(self.env.NUM_ACTIONS):
                for (prob, s_next, r, done) in self.model[(s,a_i)]:
                    if prob > 0:
                        self.predecessors[s_next].add(s)
        self.theta = theta
        self.max_iterations = max_iterations
        self.updates_per_iter = updates_per_iter

    def build_model(self):
        for s in self.states:
            i, j = s
            omega_val, omega_dot_val = self.env.state_from_idx(i,j)
            for a_i, a_val in enumerate(self.env.ACTIONS):
                self.env.set_state(omega_val, omega_dot_val)
                next_s_cont, r, done = self.env.step(a_val)
                next_s = self.env.discretize_state(next_s_cont)
                self.model[(s, a_i)] = [(1.0, next_s, r, done)]

    def compute_max_action_value(self, s):
        if s not in self.V:
            return 0.0
        best_val = -math.inf
        transitions_done = True
        for a_i in range(self.env.NUM_ACTIONS):
            for (prob, s_next, r, done) in self.model[(s, a_i)]:
                val = r + (0 if done else self.gamma * self.V[s_next])
                if val > best_val:
                    best_val = val
                if not done:
                    transitions_done = False
        return best_val

    def priority_sweeping(self, n):
        pq = []
        for s in self.states:
            old_val = self.V[s]
            new_val = self.compute_max_action_value(s)
            diff = abs(old_val - new_val)
            if diff > self.theta:
                heapq.heappush(pq, (-diff, s))

        for _ in range(n):
            if not pq:
                break
            priority, s = heapq.heappop(pq)
            old_val = self.V[s]
            new_val = self.compute_max_action_value(s)
            self.V[s] = new_val
            diff_s = abs(old_val - new_val)

            for p in self.predecessors[s]:
                old_val_p = self.V[p]
                new_val_p = self.compute_max_action_value(p)
                diff_p = abs(old_val_p - new_val_p)
                if diff_p > self.theta:
                    heapq.heappush(pq, (-diff_p, p))

    def derive_policy(self):
        policy = {}
        for s in self.states:
            best_a = None
            best_val = -math.inf
            for a_i in range(self.env.NUM_ACTIONS):
                for (prob, s_next, r, done) in self.model[(s,a_i)]:
                    val = r + (0 if done else self.gamma * self.V[s_next])
                    if val > best_val:
                        best_val = val
                        best_a = a_i
            policy[s] = best_a
        return policy

    def evaluate_policy(self, policy, episodes=200):
        total_return = 0.0
        for _ in range(episodes):
            self.env.reset_state()
            ep_return = 0.0
            for _ in range(self.env.MAX_STEPS):
                s = self.env.discretize_state(np.array([self.env.omega, self.env.omega_dot]))
                a_i = policy[s]
                a = self.env.ACTIONS[a_i]
                next_s, r, done = self.env.step(a)
                ep_return += r
                if done:
                    break
            total_return += ep_return
        return total_return/episodes


    def learning_rate(self):
        performance = []
        for i in range(self.max_iterations):
            self.priority_sweeping(self.updates_per_iter)
            pol = self.derive_policy()
            avg_return = self.evaluate_policy(pol, episodes=200)
            performance.append(avg_return)
            if (i+1)%1000 == 0:
                print(f"Iteration {i+1}/{self.max_iterations}, Average Return: {avg_return:.2f}")

        plt.figure(figsize=(8,5))
        plt.plot(range(1, self.max_iterations+1), performance)
        plt.xlabel("Iteration")
        plt.ylabel("Average Return")
        plt.title("Prioritized Sweeping on Inverted Pendulum: Performance vs Iterations")
        plt.grid(True)
        plt.show()

    def print_prioritized_sweeping(self):
        final_policy = self.derive_policy()
        print("\nSample of State-Value Function and Policy:")
        print("Format: (omega_idx, omega_dot_idx): V  PolicyAction")
        for i in range(0, self.env.NUM_OMEGA_BINS, 5):
            for j in range(0, self.env.NUM_OMEGA_DOT_BINS, 5):
                s = (i,j)
                val = self.V[s]
                a_i = final_policy[s]
                a = self.env.ACTIONS[a_i]
                print(f"State({i},{j}): V={val:.2f}, a={a}")

if __name__ == "__main__":
    env = InvertedPendulum()
    agent = PrioritizedSweepingAgent(env, gamma=0.925, theta=1e-3, max_iterations=10000, updates_per_iter=1000)
    agent.learning_rate()
    agent.print_prioritized_sweeping()
