import math
import heapq
import matplotlib.pyplot as plt
from mazemdp import MazeWorldMDP

class PrioritizedSweepingMaze:
    def __init__(self, mdp, gamma=0.99, theta=1e-5):#, max_iterations=1000):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        # self.max_iterations = max_iterations
        self.actions = self.mdp.actions

        self.V = {}
        for r in range(self.mdp.rows):
            for c in range(self.mdp.cols):
                self.V[(r,c)] = 0.0

        self.model = {}
        self.build_model()

        self.predecessors = {s:set() for s in self.V.keys()}
        for s in self.V.keys():
            for a in self.actions:
                transitions = self.model[(s,a)]
                for (prob, s_next, r, done) in transitions:
                    if prob > 0 and s_next in self.predecessors:
                        self.predecessors[s_next].add(s)


    def build_model(self):
        for s in self.V.keys():
            for a in self.actions:
                transitions = self.mdp.get_next_state_distribution(s, a)
                self.model[(s,a)] = transitions

    def compute_max_action_value(self, s):
        if self.mdp.is_terminal(s):
            return 0.0
        best_val = -math.inf
        for a in self.actions:
            val = 0.0
            for (prob, s_next, r, done) in self.model[(s,a)]:
                val += prob*(r + (0 if done else self.gamma*self.V[s_next]))
            if val > best_val:
                best_val = val
        return best_val

    def priority_sweeping(self, updates=10):
        pq = []
        for s in self.V.keys():
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
                if self.mdp.is_terminal(p):
                    continue
                old_val_p = self.V[p]
                new_val_p = self.compute_max_action_value(p)
                diff_p = abs(old_val_p - new_val_p)
                if diff_p > self.theta:
                    heapq.heappush(pq, (-diff_p, p))

    def derive_policy(self):
        policy = {}
        for s in self.V.keys():
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            best_action = None
            best_value = -math.inf
            for a in self.actions:
                val = 0.0
                for (prob, s_next, r, done) in self.model[(s,a)]:
                    val += prob*(r + (0 if done else self.gamma*self.V[s_next]))
                if val > best_value:
                    best_value = val
                    best_action = a
            policy[s] = best_action
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
                if s == self.mdp.start:
                    row_str.append("S")
                elif s == self.mdp.exit:
                    row_str.append("G")
                elif s in self.mdp.walls:
                    row_str.append("X")
                else:
                    a = policy[s]
                    char = self.mdp.action_symbols[a]
                    row_str.append(char)
            print(" ".join(row_str))

if __name__ == "__main__":
    mdp = MazeWorldMDP(gamma=0.9)

    theta = 0.01
    updates_per_sweeps = [5, 10, 15, 20] #15 best

    results = {}  

    for updates_per_sweep in updates_per_sweeps:
        print(f"\nTesting with theta={theta}")
        agent = PrioritizedSweepingMaze(mdp, gamma=mdp.gamma, theta=theta)
        
        mse_values = []
        iterations = 25  # Number of sweeps
        
        for i in range(iterations):
            agent.priority_sweeping(updates=updates_per_sweep)
            mse = agent.compute_mse()
            mse_values.append(mse)
        
        # Store the results
        results[updates_per_sweep] = mse_values
        
        # Print final values and policy
        print("Final Values:")
        agent.print_values()
        print("Final Policy:")
        final_policy = agent.derive_policy()
        agent.print_policy(final_policy)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    for updates_per_sweep, mse_values in results.items():
        plt.plot(range(1, iterations+1), mse_values, marker='o', label=f"Updates per sweep={updates_per_sweep}")
    
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Prioritized Sweeping Learning Curve (MSE vs Iterations) for Different θ")
    plt.legend()
    plt.grid(True)
    plt.show()

