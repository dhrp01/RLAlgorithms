import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mazemdp import MazeWorldMDP

class MCTSAgent:
    def __init__(self, mdp, gamma=0.99, c=1.0, num_simulations=500, max_depth=50):
        self.mdp = mdp
        self.gamma = gamma
        self.c = c
        self.num_simulations = num_simulations
        self.max_depth = max_depth

        self.actions = self.mdp.actions
        # Q-values and counts
        # Use dictionaries keyed by (s,a) for Q and N, and by s for N_s
        self.Q = {}
        self.Nsa = {}
        self.Ns = {}

    def reset_stats(self):
        self.Q.clear()
        self.Nsa.clear()
        self.Ns.clear()

    def UCB(self, s, a):
        # Upper Confidence Bound for trees
        sa = (s,a)
        if self.Nsa.get(sa,0) == 0:
            return math.inf
        return self.Q.get(sa,0.0) + self.c * math.sqrt(math.log(self.Ns.get(s,1))/self.Nsa[sa])

    def select_action(self, s):
        # Choose action that maximizes UCB
        best_a = None
        best_val = -math.inf
        for a in self.actions:
            val = self.UCB(s,a)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def simulate(self, s, depth=0):
        # Simulation function for MCTS
        # If terminal or depth limit reached, return 0
        if self.mdp.is_terminal(s) or depth >= self.max_depth:
            return 0.0

        # If state not expanded yet, do a random rollout
        if s not in self.Ns:
            # Initialize counts
            self.Ns[s] = 0
            for a in self.actions:
                self.Q[(s,a)] = 0.0
                self.Nsa[(s,a)] = 0

            # Do a random rollout from this node
            return self.rollout(s, depth)

        # Otherwise, select action using UCB
        a = self.select_action(s)
        # Sample next state from the model
        transitions = self.mdp.get_next_state_distribution(s,a)
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        p, s_next, r, done = transitions[idx]

        # Recursively simulate
        G = r + self.gamma * self.simulate(s_next, depth+1)

        # Backpropagate
        self.Ns[s] += 1
        self.Nsa[(s,a)] += 1
        self.Q[(s,a)] += (G - self.Q[(s,a)]) / self.Nsa[(s,a)]

        return G

    def rollout(self, s, depth):
        # Random rollout until terminal or depth limit
        if self.mdp.is_terminal(s) or depth >= self.max_depth:
            return 0.0
        a = random.choice(self.actions)
        transitions = self.mdp.get_next_state_distribution(s,a)
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        p, s_next, r, done = transitions[idx]
        return r + self.gamma * self.rollout(s_next, depth+1)

    def run_mcts_iterations(self, iterations=50):
        # Run multiple iterations:
        # Each iteration: run num_simulations MCTS simulations from start state
        # Compute MSE after each iteration
        mse_list = []
        for i in range(iterations):
            for _ in range(self.num_simulations):
                self.simulate(self.mdp.start)
            mse = self.compute_mse()
            mse_list.append(mse)
        return mse_list

    def compute_value_function(self):
        # Value = max_a Q(s,a)
        V = {}
        # Some states might not have been visited; for them V=0
        states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        for s in states:
            q_vals = []
            for a in self.actions:
                q_vals.append(self.Q.get((s,a), 0.0))
            V[s] = max(q_vals)
        return V

    def derive_policy(self):
        policy = {}
        states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        for s in states:
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            # Pick action with max Q(s,a)
            best_a = None
            best_val = -math.inf
            for a in self.actions:
                val = self.Q.get((s,a),0.0)
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s] = best_a
        return policy

    def compute_mse(self):
        # Compare current V(s) with mdp.optimal_values
        V = self.compute_value_function()
        errors = []
        for s, opt_val in self.mdp.optimal_values.items():
            if opt_val is not None:
                v_est = V[s]
                errors.append((v_est - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def print_values(self):
        V = self.compute_value_function()
        print("Value Function:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                val = V[(r,c)]
                row_str.append(f"{val:6.2f}")
            print(" ".join(row_str))

    def print_policy(self, policy):
        print("Policy:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                s = (r,c)
                if s == self.mdp.start:
                    char = "S"
                elif s == self.mdp.exit:
                    char = "G"
                elif s in self.mdp.walls:
                    char = "X"
                else:
                    a = policy[s]
                    char = self.mdp.action_symbols[a]
                row_str.append(char)
            print(" ".join(row_str))


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Create the MDP environment
    mdp = MazeWorldMDP(gamma=0.9)

    # MCTS Agent
    agent = MCTSAgent(mdp, gamma=mdp.gamma, c=1.4, num_simulations=300, max_depth=100)
    iterations = 10000
    mse_list = agent.run_mcts_iterations(iterations=iterations)

    # Print final values and policy
    agent.print_values()
    final_policy = agent.derive_policy()
    agent.print_policy(final_policy)

    # Plot MSE vs Iterations
    plt.figure(figsize=(8,5))
    plt.plot(range(1, iterations+1), mse_list, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MCTS Learning Curve (MSE vs Iterations)")
    plt.grid(True)
    plt.show()
