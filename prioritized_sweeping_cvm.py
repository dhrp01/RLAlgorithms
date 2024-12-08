import heapq
import math
import numpy as np
from catvsmonstermdp import CatVsMonster
import matplotlib.pyplot as plt

class PrioritizedSweepingAgent:
    def __init__(self, env: CatVsMonster, theta=1e-6, max_iterations=1000):
        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations

        self.states = [(r, c) for r in range(self.env.rows) for c in range(self.env.cols) if (r,c) not in self.env.furniture]

        self.V = {s: 0.0 for s in self.states}

        self.model = {}
        self.build_model()

        self.predecessors = {s: set() for s in self.states}
        for s in self.states:
            for a in self.env.actions:
                transitions = self.model[(s, a)]
                for (prob, s_next, r) in transitions:
                    if prob > 0:
                        self.predecessors[s_next].add(s)

    def build_model(self):
        action_map = {
            "AU": (-1, 0),
            "AD": (1, 0),
            "AL": (0, -1),
            "AR": (0, 1)
        }

        left_turn = {"AU": "AL", "AD": "AR", "AL": "AD", "AR": "AU"}
        right_turn = {"AU": "AR", "AD": "AL", "AL": "AU", "AR": "AD"}

        for s in self.states:
            for a in self.env.actions:
                if self.env.is_terminal(s):
                    self.model[(s, a)] = [(1.0, s, self.env.get_reward(s))]
                    continue

                dr, dc = action_map[a]
                intended = self.attempt_move(s, dr, dc)

                a_left = left_turn[a]
                dr_left, dc_left = action_map[a_left]
                left_st = self.attempt_move(s, dr_left, dc_left)

                a_right = right_turn[a]
                dr_right, dc_right = action_map[a_right]
                right_st = self.attempt_move(s, dr_right, dc_right)

                stay_st = s

                transitions = [
                    (0.70, intended, self.env.get_reward(intended)),
                    (0.12, right_st, self.env.get_reward(right_st)),
                    (0.12, left_st, self.env.get_reward(left_st)),
                    (0.06, stay_st, self.env.get_reward(stay_st))
                ]

                combined = {}
                for (p, ns, r) in transitions:
                    if ns not in combined:
                        combined[ns] = (p, r)
                    else:
                        combined[ns] = (combined[ns][0] + p, r)
                transitions = [(combined_ns[0], ns, combined_ns[1]) for ns, combined_ns in combined.items()]

                self.model[(s, a)] = transitions

    def attempt_move(self, s, dr, dc):
        nr, nc = s[0] + dr, s[1] + dc
        if nr < 0 or nr >= self.env.rows or nc < 0 or nc >= self.env.cols or (nr, nc) in self.env.furniture:
            return s
        return (nr, nc)

    def compute_max_action_value(self, s):
        if self.env.is_terminal(s):
            return 0.0
        best_val = -math.inf
        for a in self.env.actions:
            val = 0.0
            for (prob, s_next, r) in self.model[(s, a)]:
                val += prob * (r + self.env.gamma * self.V[s_next])
            if val > best_val:
                best_val = val
        return best_val

    def priority_sweeping(self, n=5):
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
            priority = -priority
            old_val = self.V[s]
            self.V[s] = self.compute_max_action_value(s)
            diff = abs(old_val - self.V[s])

            for p in self.predecessors[s]:
                if self.env.is_terminal(p):
                    continue
                old_val_p = self.V[p]
                new_val_p = self.compute_max_action_value(p)
                diff_p = abs(old_val_p - new_val_p)
                if diff_p > self.theta:
                    heapq.heappush(pq, (-diff_p, p))

    def compute_mse(self):
        errors = []
        for s, opt_val in self.env.optimal_values.items():
            if opt_val is not None and s in self.V:
                errors.append((self.V[s] - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def print_optimal_values(self):
        print("\nFinal Value Function:")
        for r in range(self.env.rows):
            row = []
            for c in range(self.env.cols):
                if (r, c) in self.env.furniture:
                    row.append("    X   ")
                else:
                    row.append(f"{self.V[(r, c)]:8.4f}")
            print(" ".join(row))

    def derive_policy(self):
        policy = {}
        for s in self.states:
            if self.env.is_terminal(s):
                policy[s] = None
            else:
                best_a = None
                best_val = -math.inf
                for a in self.env.actions:
                    val = 0.0
                    for (prob, s_next, r) in self.model[(s, a)]:
                        val += prob * (r + self.env.gamma * self.V[s_next])
                    if val > best_val:
                        best_val = val
                        best_a = a
                policy[s] = best_a
        return policy

    def print_policy(self):
        policy_grid = []
        policy = self.derive_policy()
        for r in range(self.env.rows):
            row = []
            for c in range(self.env.cols):
                state = (r, c)
                if state in self.env.furniture:
                    row.append(" ")
                elif state == self.env.food:
                    row.append("G")
                elif state in self.env.monsters:
                    best_action = policy.get(state, [0,0,0,0])
                    row.append(self.env.action_symbols[best_action])
                else:
                    best_action = policy.get(state, [0,0,0,0])
                    if best_action:
                        row.append(self.env.action_symbols[best_action])
            policy_grid.append(row)

        print("Policy:")
        for row in policy_grid:
            print(" ".join(row))


if __name__ == "__main__":
    env = CatVsMonster()
    agent = PrioritizedSweepingAgent(env, theta=1e-9, max_iterations=50)

    mse_values = []
    num_sweeps = 25
    updates_per_sweep =10

    for i in range(num_sweeps):
        agent.priority_sweeping(n=updates_per_sweep)
        mse = agent.compute_mse()
        mse_values.append(mse)
        print(f"Sweep {i+1}, MSE: {mse:.6f}")

    agent.print_optimal_values()
    agent.print_policy()

    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_sweeps+1), mse_values, marker='o')
    plt.title("Prioritized Sweeping Learning Curve (MSE)")
    plt.xlabel("Sweep")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()
