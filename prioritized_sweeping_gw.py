import math
import heapq
import matplotlib.pyplot as plt
from gridworldmdp import GridWorldMDP

class PrioritizedSweepingAgent:
    def __init__(self, mdp, theta=1e-5, max_iterations=2000):
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        self.V = {s:0.0 for s in self.states}

        self.model = {}
        self.build_model()

        self.predecessors = {s: set() for s in self.states}
        for s in self.states:
            for a_i, a in enumerate(self.mdp.actions):
                transitions = self.model[(s,a)]
                for (prob, s_next, r, done) in transitions:
                    if prob > 0:
                        self.predecessors[s_next].add(s)

    def build_model(self):
        for s in self.states:
            for a in self.mdp.actions:
                self.model[(s,a)] = self.mdp.get_next_state_distribution(s,a)

    def compute_max_action_value(self, s):
        if self.mdp.is_terminal(s):
            return 0.0
        best_val = -math.inf
        for a in self.mdp.actions:
            val = 0.0
            for (prob, s_next, r, done) in self.model[(s,a)]:
                val += prob*(r + (0 if done else self.gamma*self.V[s_next]))
            if val > best_val:
                best_val = val
        return best_val

    def priority_sweeping(self, updates=10):
        pq = []
        for s in self.states:
            old_val = self.V[s]
            new_val = self.compute_max_action_value(s)
            diff = abs(old_val - new_val)
            if diff > self.theta:
                heapq.heappush(pq, (-diff, s))

        for _ in range(updates):
            if not pq:
                break
            priority, s = heapq.heappop(pq)
            old_val = self.V[s]
            self.V[s] = self.compute_max_action_value(s)
            for p in self.predecessors[s]:
                old_val_p = self.V[p]
                new_val_p = self.compute_max_action_value(p)
                diff_p = abs(old_val_p - new_val_p)
                if diff_p > self.theta:
                    heapq.heappush(pq, (-diff_p, p))

    def derive_policy(self):
        policy = {}
        for s in self.states:
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            best_a = None
            best_val = -math.inf
            for a in self.mdp.actions:
                val = 0.0
                for (prob, s_next, r, done) in self.model[(s,a)]:
                    val += prob*(r + (0 if done else self.gamma*self.V[s_next]))
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s] = best_a
        return policy

    def compute_mse(self):
        errors = []
        for s, opt_val in self.mdp.optimal_values.items():
            if s in self.V and opt_val is not None:
                errors.append((self.V[s] - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def print_values(self):
        print("Value Function:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                val = self.V[(r,c)]
                row_str.append(f"{val:6.2f}")
            print(" ".join(row_str))

    def print_policy(self, policy):
        print("Policy:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                s = (r,c)
                if s == self.mdp.goal:
                    row_str.append("G")
                elif s == self.mdp.start:
                    row_str.append("S")
                elif s in self.mdp.obstacles:
                    row_str.append("X")
                elif s == self.mdp.water:
                    row_str.append("W")
                else:
                    a = policy[s]
                    if a is None:
                        row_str.append("T")
                    else:
                        row_str.append(self.mdp.action_symbols[a])
            print(" ".join(row_str))


if __name__ == "__main__":
    mdp = GridWorldMDP(gamma=0.9)
    agent = PrioritizedSweepingAgent(mdp, theta=1e-9, max_iterations=2000)

    mse_values = []
    iterations = 20
    updates_per_iter = 20

    for i in range(iterations):
        agent.priority_sweeping(updates=updates_per_iter)
        mse = agent.compute_mse()
        mse_values.append(mse)

    agent.print_values()
    final_policy = agent.derive_policy()
    agent.print_policy(final_policy)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, iterations+1), mse_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Prioritized Sweeping Learning Curve (MSE vs Iterations)")
    plt.grid(True)
    plt.show()
