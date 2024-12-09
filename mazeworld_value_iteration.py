import math

class MazeValueIterationAgent:
    def __init__(self, mdp, gamma=0.99, theta=1e-6, max_iterations=1000):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = {}
        for r in range(self.mdp.rows):
            for c in range(self.mdp.cols):
                self.V[(r,c)] = 0.0

    def run_value_iteration(self):
        for i in range(self.max_iterations):
            delta = 0
            for s in self.V.keys():
                if self.mdp.is_terminal(s):
                    continue

                action_values = []
                old_val = self.V[s]

                for a in self.mdp.actions:
                    transitions = self.mdp.get_next_state_distribution(s,a)
                    val = 0.0
                    for (prob, s_next, r, done) in transitions:
                        val += prob * (r + self.gamma * self.V[s_next])
                    action_values.append(val)

                if len(action_values) > 0:
                    self.V[s] = max(action_values)
                else:
                    self.V[s] = self.V[s]

                delta = max(delta, abs(self.V[s] - old_val))

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
        action_symbols = {
            'AU': '↑',
            'AD': '↓',
            'AL': '←',
            'AR': '→',
            None: 'G'
        }
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
                    if a is None:
                        char = "G"
                    else:
                        char = action_symbols[a]
                row_str.append(char)
            print(" ".join(row_str))

if __name__ == "__main__":
    from mazemdp import MazeWorldMDP
    mdp = MazeWorldMDP(gamma=0.99)
    agent = MazeValueIterationAgent(mdp, gamma=mdp.gamma, theta=1e-9, max_iterations=1000)
    agent.run_value_iteration()
    final_policy = agent.derive_policy()
    print("\nPolicy:")
    print(final_policy)
    print("\nValues:")
    print(agent.V)
    agent.print_values()
    agent.print_policy(final_policy)
