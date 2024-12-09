import numpy as np
import random
import matplotlib.pyplot as plt
import math
from gridworldmdp import GridWorldMDP


class REINFORCEWithBaselineAgent:
    def __init__(self, mdp, gamma=0.9, alpha_theta=0.01, alpha_w=0.01, num_episodes=5000):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.num_episodes = num_episodes

        self.states = [(r,c) for r in range(self.mdp.rows) for c in range(self.mdp.cols)]
        self.num_states = len(self.states)
        self.state_to_id = {s:i for i,s in enumerate(self.states)}
        self.num_actions = self.mdp.n_actions
        self.actions = self.mdp.actions

        self.theta = np.zeros((self.num_states, self.num_actions))
        self.w = np.zeros(self.num_states)

    def softmax_policy(self, s_id):
        prefs = self.theta[s_id]
        max_pref = np.max(prefs)
        exp_prefs = np.exp(prefs - max_pref)
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, s):
        s_id = self.state_to_id[s]
        probs = self.softmax_policy(s_id)
        a_id = np.random.choice(range(self.num_actions), p=probs)
        return self.actions[a_id], a_id, probs

    def generate_episode(self):
        s = self.mdp.start
        episode = []
        done = False
        while not done:
            a, a_id, probs = self.select_action(s)
            transitions = self.mdp.get_next_state_distribution(s,a)
            p = [t[0] for t in transitions]
            idx = np.random.choice(len(transitions), p=p)
            p_t, s_next, r, done = transitions[idx]

            episode.append((s, a, a_id, r))
            s = s_next
        return episode

    def run_reinforce_with_baseline(self):
        MSE_list = []
        for ep in range(self.num_episodes):
            episode = self.generate_episode()
            T = len(episode)
            G = np.zeros(T)
            returns = 0.0
            for t in reversed(range(T)):
                returns = self.gamma * returns + episode[t][3]
                G[t] = returns

            for t in range(T):
                s = episode[t][0]
                a_id = episode[t][2]
                g = G[t]
                s_id = self.state_to_id[s]
                b = self.w[s_id]
                self.w[s_id] += self.alpha_w * (g - b)
                probs = self.softmax_policy(s_id)
                grad_log_pi = -probs
                grad_log_pi[a_id] += 1.0
                self.theta[s_id, :] += self.alpha_theta * (g - b) * grad_log_pi

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
    random.seed(0)
    np.random.seed(0)

    mdp = GridWorldMDP(gamma=0.925)
    agent = REINFORCEWithBaselineAgent(mdp, gamma=mdp.gamma, alpha_theta=2e-4, alpha_w=0.1, num_episodes=500000)
    mse_list = agent.run_reinforce_with_baseline()

    agent.print_values()
    final_policy = agent.derive_greedy_policy()
    agent.print_policy(final_policy)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("REINFORCE with Baseline Learning Curve (MSE vs Episodes)")
    plt.grid(True)
    plt.show()
