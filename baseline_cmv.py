import numpy as np
import random
import matplotlib.pyplot as plt
import math
from catvsmonstermdp import CatVsMonster

class REINFORCEWithBaselineCatVsMonster:
    def __init__(self, env, gamma=0.925, alpha_theta=0.01, alpha_w=0.01, num_episodes=5000):
        self.env = env
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.num_episodes = num_episodes

        self.actions = self.env.actions
        self.n_actions = self.env.n_actions

        self.states = [(r,c) for r in range(self.env.rows) for c in range(self.env.cols)
                       if (r,c) not in self.env.furniture]
        self.state_to_id = {s:i for i,s in enumerate(self.states)}
        self.num_states = len(self.states)

        self.theta = np.zeros((self.num_states, self.n_actions))
        self.w = np.zeros(self.num_states)

    def is_terminal(self, s):
        return self.env.is_terminal(s) or (s in self.env.monsters)

    def softmax_policy(self, s_id):
        prefs = self.theta[s_id]
        max_pref = np.max(prefs)
        exp_prefs = np.exp(prefs - max_pref)
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, s):
        s_id = self.state_to_id[s]
        probs = self.softmax_policy(s_id)
        a_id = np.random.choice(range(self.n_actions), p=probs)
        a = self.actions[a_id]
        return a, a_id, probs

    def generate_episode(self):
        s = self.env.initial_state()
        episode = []
        while True:
            a, a_id, probs = self.select_action(s)
            next_s = self.env.get_next_state(s, a)
            r = self.env.get_reward(next_s)
            done = self.is_terminal(next_s)
            episode.append((s, a, a_id, r))
            s = next_s
            if done:
                break
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
        for s, opt_val in self.env.optimal_values.items():
            if opt_val is not None and s in self.state_to_id:
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
            if self.is_terminal(s):
                policy[s] = None
                continue
            s_id = self.state_to_id[s]
            probs = self.softmax_policy(s_id)
            best_a_id = np.argmax(probs)
            policy[s] = self.actions[best_a_id]
        return policy

    def print_values(self):
        V = self.get_value_function()
        print("Value Function:")
        for r in range(self.env.rows):
            row_str = []
            for c in range(self.env.cols):
                s = (r,c)
                if s in self.env.furniture:
                    row_str.append("  X   ")
                else:
                    val = V[s] if s in V else 0.0
                    row_str.append(f"{val:6.2f}")
            print(" ".join(row_str))

    def print_policy(self, policy):
        print("Policy:")
        for r in range(self.env.rows):
            row_str = []
            for c in range(self.env.cols):
                s = (r,c)
                if s in self.env.furniture:
                    row_str.append("X")
                elif s == self.env.food:
                    row_str.append("G")
                elif s in self.env.monsters:
                    row_str.append("M")
                else:
                    a = policy[s]
                    if a is None:
                        row_str.append("T")
                    else:
                        row_str.append(self.env.action_symbols[a])
            print(" ".join(row_str))


if __name__ == "__main__":
    from collections import defaultdict
    random.seed(0)
    np.random.seed(0)

    env = CatVsMonster(gamma=0.925)
    agent = REINFORCEWithBaselineCatVsMonster(env, gamma=env.gamma, alpha_theta=5e-4, alpha_w=0.01, num_episodes=1000000)
    mse_list = agent.run_reinforce_with_baseline()

    agent.print_values()
    final_policy = agent.derive_greedy_policy()
    agent.print_policy(final_policy)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("REINFORCE with Baseline (CatVsMonster): MSE vs Episodes")
    plt.grid(True)
    plt.show()
