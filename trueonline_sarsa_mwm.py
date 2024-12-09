import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mazemdp import MazeWorldMDP

class TrueOnlineSarsaAgent:
    def __init__(self, mdp, gamma=0.99, alpha=0.1, lambd=0.9, epsilon=0.1, num_episodes=5000):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        self.actions = self.mdp.actions
        self.num_actions = len(self.actions)
        self.states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        self.state_to_id = {s:i for i,s in enumerate(self.states)}
        self.num_states = len(self.states)

        # Q-table: shape [num_states, num_actions]
        self.Q = np.zeros((self.num_states, self.num_actions))

    def state_id(self, s):
        return self.state_to_id[s]

    def epsilon_greedy(self, s_id):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            qvals = self.Q[s_id]
            max_q = np.max(qvals)
            best_actions = [i for i,a_q in enumerate(qvals) if a_q == max_q]
            return random.choice(best_actions)

    def run_true_online_sarsa(self):
        MSE_list = []
        for ep in range(self.num_episodes):
            s = self.mdp.reset()
            s_id = self.state_id(s)
            a_id = self.epsilon_greedy(s_id)
            q_old = self.Q[s_id,a_id]

            # Eligibility traces
            e = np.zeros((self.num_states, self.num_actions))

            done = False
            while not done:
                # Take action a
                action = self.actions[a_id]
                s_next, r, done = self.mdp.step(action)
                s_next_id = self.state_id(s_next)
                a_next_id = self.epsilon_greedy(s_next_id)

                q = self.Q[s_id,a_id]
                q_next = self.Q[s_next_id,a_next_id] if not done else 0.0

                delta = r + self.gamma * q_next - q

                # Update e
                e[s_id,a_id] += 1.0

                # True online Sarsa update:
                # For all x,y:
                # Q(x,y) += alpha * (delta + q - q_old) * e(x,y)
                #          - alpha * (q - q_old)
                # The simpler tabular variant derived from official True Online Sarsa eq:
                # Actually, in tabular form, the official formula is simpler:
                # Q <- Q + alpha * (delta + q - q_old)*e - alpha(q - q_old)

                # We'll follow the derivation from True Online Sarsa paper for tabular form:
                # The tabular form reduces nicely. Reference: "True online TD(λ)" by van Seijen & Sutton (2014).
                # For tabular:
                # Q(x,y) <- Q(x,y) + α[δ + q - q_old]e(x,y) - α(q - q_old)
                # But the term -α(q - q_old) is applied only to the current state-action to ensure correctness.
                # Actually, let's implement directly from main formula provided in literature:
                # According to the paper, the update is:
                # Q <- Q + alpha[delta + q - q_old]*e - alpha(q - q_old)*I
                # where I is an indicator vector of length Q-size with 1 for the current s,a and 0 otherwise.

                # We'll store q_old to handle next step.
                # The only difference in tabular is that "I" is just for s,a.

                # Compute the correction term:
                correction = self.alpha * (q - q_old)
                # Update all Q with main part:
                self.Q += self.alpha * (delta + q - q_old) * e
                # Subtract correction from current s,a:
                self.Q[s_id,a_id] -= correction

                q_old = q_next

                # Decay e
                e *= self.gamma * self.lambd

                # Move to next state
                s_id = s_next_id
                a_id = a_next_id

            # After each episode compute MSE
            mse = self.compute_mse()
            MSE_list.append(mse)
        return MSE_list

    def compute_mse(self):
        V = {}
        for s in self.states:
            s_id = self.state_id(s)
            V[s] = np.max(self.Q[s_id])
        errors = []
        for s, opt_val in self.mdp.optimal_values.items():
            if opt_val is not None:
                v_est = V[s]
                errors.append((v_est - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def get_value_function(self):
        V = {}
        for s in self.states:
            s_id = self.state_id(s)
            V[s] = np.max(self.Q[s_id])
        return V

    def derive_policy(self):
        policy = {}
        for s in self.states:
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            s_id = self.state_id(s)
            qvals = self.Q[s_id]
            max_q = np.max(qvals)
            best_actions = [i for i,a_q in enumerate(qvals) if a_q == max_q]
            best_a_id = random.choice(best_actions)
            policy[s] = self.actions[best_a_id]
        return policy

    def print_values(self):
        V = self.get_value_function()
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
    from matplotlib import pyplot as plt

    mdp = MazeWorldMDP(gamma=0.9)
    agent = TrueOnlineSarsaAgent(mdp, gamma=mdp.gamma, alpha=2e-4, lambd=0.9, epsilon=0.1, num_episodes=1000000)
    mse_list = agent.run_true_online_sarsa()

    # Print final values and policy
    agent.print_values()
    final_policy = agent.derive_policy()
    agent.print_policy(final_policy)

    # Plot MSE vs Episodes
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("True Online Sarsa Learning Curve (MSE vs Episodes)")
    plt.grid(True)
    plt.show()