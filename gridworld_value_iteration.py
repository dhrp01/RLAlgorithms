import math
from gridworldmdp import GridWorldMDP

class ValueIterationAgent:
    def __init__(self, mdp, gamma=0.99, theta=1e-6, max_iterations=1000):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = {}
        for r in range(self.mdp.rows):
            for c in range(self.mdp.cols):
                if self.mdp.in_bounds(r, c) or self.mdp.is_obstacle((r,c)):
                    self.V[(r,c)] = 0.0

    def run_value_iteration(self):
        for i in range(self.max_iterations):
            delta = 0
            for s in self.V.keys():
                if self.mdp.is_terminal(s):
                    continue

                action_values = []
                old_V = self.V[s]
                for a in self.mdp.actions:
                    transitions = self.mdp.get_next_state_distribution(s,a)
                    val = 0.0
                    for (prob, s_next, r, done) in transitions:
                        val += prob * (r + self.gamma * self.V[s_next])
                    action_values.append(val)

                self.V[s] = max(action_values) if len(action_values) > 0 else self.V[s]

                delta = max(delta, abs(self.V[s] - old_V))

            if delta < self.theta:
                print(f"Value Iteration converged after {i+1} iterations.")
                break

    def derive_policy(self):
        policy = {}
        for s in self.V.keys():
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            best_action = None
            best_value = -math.inf
            for a in self.mdp.actions:
                val = 0.0
                transitions = self.mdp.get_next_state_distribution(s,a)
                for (prob, s_next, r, done) in transitions:
                    val += prob*(r + self.gamma*self.V[s_next])
                if val > best_value:
                    best_value = val
                    best_action = a
            policy[s] = best_action
        return policy

    def print_values(self):
        print("Value Function:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                v = self.V[(r,c)]
                row_str.append(f"{v:6.2f}")
            print(" ".join(row_str))

    def print_policy(self, policy):
        print("Policy:")
        for r in range(self.mdp.rows):
            row_str = []
            for c in range(self.mdp.cols):
                s = (r,c)
                if s == self.mdp.start:
                    char = "S"
                elif s == self.mdp.goal:
                    char = "G"
                elif s in self.mdp.obstacles:
                    char = "X"
                elif s == self.mdp.water:
                    char = "W"
                else:
                    char = self.mdp.action_symbols[policy[s]]
                row_str.append(char)
            print(" ".join(row_str))

if __name__ == "__main__":
    mdp = GridWorldMDP(gamma=0.9)
    agent = ValueIterationAgent(mdp, gamma=mdp.gamma, theta=1e-9, max_iterations=1000)
    agent.run_value_iteration()
    final_policy = agent.derive_policy()
    print("\nPolicy:")
    print(final_policy)
    print("\nValues:")
    print(agent.V)
    agent.print_values()
    agent.print_policy(final_policy)
