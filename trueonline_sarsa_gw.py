import numpy as np
import random
import math
import matplotlib.pyplot as plt
from gridworldmdp import GridWorldMDP

class TrueOnlineSarsaGridWorld:
    def __init__(self, mdp, gamma=0.9, alpha=0.1, lambd=0.9, epsilon=0.1, num_episodes=5000):
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

        self.Q = np.zeros((self.num_states, self.num_actions))

    def state_id(self, s):
        return self.state_to_id[s]

    def is_terminal(self, s):
        return self.mdp.is_terminal(s)

    def epsilon_greedy(self, s_id):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            qvals = self.Q[s_id]
            max_q = np.max(qvals)
            best_actions = [i for i,a_q in enumerate(qvals) if a_q == max_q]
            return random.choice(best_actions)

    def initial_state(self):
        return self.mdp.start

    def run_true_online_sarsa(self):
        MSE_list = []
        for ep in range(self.num_episodes):
            s = self.initial_state()
            s_id = self.state_id(s)
            a_id = self.epsilon_greedy(s_id)
            q_old = self.Q[s_id,a_id]

            e = np.zeros((self.num_states, self.num_actions))

            done = False
            while not done:
                action = self.actions[a_id]
                transitions = self.mdp.get_next_state_distribution(s, action)
                probs = [t[0] for t in transitions]
                idx = np.random.choice(len(transitions), p=probs)
                p, s_next, r, done = transitions[idx]

                if not done:
                    s_next_id = self.state_id(s_next)
                    a_next_id = self.epsilon_greedy(s_next_id)
                    q_next = self.Q[s_next_id,a_next_id]
                else:
                    q_next = 0.0

                q = self.Q[s_id,a_id]
                delta = r + self.gamma * q_next - q
                e[s_id,a_id] += 1.0

                correction = self.alpha * (q - q_old)
                self.Q += self.alpha * (delta + q - q_old) * e
                self.Q[s_id,a_id] -= correction

                q_old = q_next
                e *= self.gamma * self.lambd

                s = s_next
                if not done:
                    s_id = s_next_id
                    a_id = a_next_id

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
            if self.is_terminal(s):
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
                v = V[(r,c)]
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
                elif self.mdp.is_terminal(s):
                    char = "G"
                elif s in self.mdp.obstacles:
                    char = "X"
                else:
                    a = policy[s]
                    if a is None:
                        char = "T"
                    else:
                        char = self.mdp.action_symbols[a]
                row_str.append(char)
            print(" ".join(row_str))

if __name__ == "__main__":
    mdp = GridWorldMDP(gamma=0.9)
    agent = TrueOnlineSarsaGridWorld(mdp, gamma=mdp.gamma, alpha=5e-4, lambd=0.9, epsilon=0.2, num_episodes=1000000)
    mse_list = agent.run_true_online_sarsa()

    agent.print_values()
    final_policy = agent.derive_policy()
    agent.print_policy(final_policy)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("True Online Sarsa on GridWorld: MSE vs Episodes")
    plt.grid(True)
    plt.show()
