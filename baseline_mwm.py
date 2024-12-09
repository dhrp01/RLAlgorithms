import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mazemdp import MazeWorldMDP
from matplotlib import pyplot as plt

class REINFORCEWithBaselineAgent:
    def __init__(self, mdp, gamma=0.99, alpha_theta=0.01, alpha_w=0.01, num_episodes=5000):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.num_episodes = num_episodes

        self.states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        self.num_states = len(self.states)
        self.state_to_id = {s:i for i,s in enumerate(self.states)}
        self.num_actions = len(self.mdp.actions)
        self.actions = self.mdp.actions

        # Initialize parameters for policy (theta) and baseline (w)
        self.theta = np.zeros((self.num_states, self.num_actions))
        self.w = np.zeros(self.num_states)  # baseline weights

    def softmax_policy(self, s_id):
        prefs = self.theta[s_id]
        max_pref = np.max(prefs)
        exp_prefs = np.exp(prefs - max_pref)
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, s):
        s_id = self.state_to_id[s]
        probs = self.softmax_policy(s_id)
        a_id = np.random.choice(range(self.num_actions), p=probs)
        a = self.actions[a_id]
        return a, a_id, probs

    def generate_episode(self):
        s = self.mdp.reset()
        episode = []
        done = False
        while not done:
            a, a_id, probs = self.select_action(s)
            s_next, r, done = self.mdp.step(a)
            episode.append((s, a_id, r))
            s = s_next
        return episode

    def run_reinforce_with_baseline(self):
        MSE_list = []
        for ep in range(self.num_episodes):
            episode = self.generate_episode()
            # Compute returns G_t
            T = len(episode)
            G = np.zeros(T)
            returns = 0.0
            for t in reversed(range(T)):
                returns = self.gamma * returns + episode[t][2]
                G[t] = returns

            # Update each step of the episode
            for t in range(T):
                s = episode[t][0]
                a_id = episode[t][1]
                g = G[t]
                s_id = self.state_to_id[s]

                baseline = self.w[s_id]
                # Update baseline
                self.w[s_id] += self.alpha_w * (g - baseline)

                # Update policy
                probs = self.softmax_policy(s_id)
                grad_log_pi = -probs
                grad_log_pi[a_id] += 1.0

                self.theta[s_id,:] += self.alpha_theta * (g - baseline) * grad_log_pi

            # Compute MSE after each episode
            mse = self.compute_mse()
            MSE_list.append(mse)
        return MSE_list

    def compute_mse(self):
        errors = []
        for s, opt_val in self.mdp.optimal_values.items():
            if opt_val is not None:
                s_id = self.state_to_id[s]
                v_est = self.w[s_id]
                errors.append((v_est - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def get_value_function(self):
        V = {}
        for s in self.states:
            s_id = self.state_to_id[s]
            V[s] = self.w[s_id]
        return V

    def derive_greedy_policy(self):
        policy = {}
        for s in self.states:
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue
            s_id = self.state_to_id[s]
            probs = self.softmax_policy(s_id)
            a_id = np.argmax(probs)
            policy[s] = self.actions[a_id]
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
    random.seed(0)
    np.random.seed(0)

    mdp = MazeWorldMDP(gamma=0.9)
    agent = REINFORCEWithBaselineAgent(mdp, gamma=mdp.gamma, alpha_theta=2e-4, alpha_w=0.1, num_episodes=1000000)
    mse_list = agent.run_reinforce_with_baseline()

    # Print final values and policy
    agent.print_values()
    final_policy = agent.derive_greedy_policy()
    agent.print_policy(final_policy)

    # Plot MSE vs Episodes
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("REINFORCE with Baseline Learning Curve (MSE vs Episodes)")
    plt.grid(True)
    plt.show()
