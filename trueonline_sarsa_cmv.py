import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math
from catvsmonstermdp import CatVsMonster

class TrueOnlineSarsaCatVsMonster:
    def __init__(self, env, gamma=0.925, alpha=0.1, lambd=0.9, epsilon=0.1, num_episodes=5000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        self.actions = self.env.actions
        self.num_actions = len(self.actions)

        # Collect all valid states (including monsters and goal), except furniture are also part of state space but unreachable?
        # We'll just store Q for all non-furniture states:
        self.states = [(r,c) for r in range(self.env.rows) for c in range(self.env.cols) if (r,c) not in self.env.furniture]
        self.state_to_id = {s:i for i,s in enumerate(self.states)}
        self.num_states = len(self.states)

        # Q-table: shape [num_states, num_actions]
        self.Q = np.zeros((self.num_states, self.num_actions))

    def state_id(self, s):
        return self.state_to_id[s]

    def is_terminal(self, s):
        return self.env.is_terminal(s)

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
            s = self.initial_state()
            s_id = self.state_id(s)
            a_id = self.epsilon_greedy(s_id)
            q_old = self.Q[s_id,a_id]

            # Eligibility traces
            e = np.zeros((self.num_states, self.num_actions))

            done = False
            while not done:
                action = self.actions[a_id]
                next_s = self.env.get_next_state(s, action)
                r = self.env.get_reward(next_s)
                done = self.is_terminal(next_s) or (next_s in self.env.monsters)

                if not done:
                    next_s_id = self.state_id(next_s)
                    a_next_id = self.epsilon_greedy(next_s_id)
                    q_next = self.Q[next_s_id, a_next_id]
                else:
                    q_next = 0.0

                q = self.Q[s_id,a_id]
                delta = r + self.gamma * q_next - q
                e[s_id,a_id] += 1.0

                # True Online Sarsa update
                # Q <- Q + alpha [delta + q - q_old] e - alpha(q - q_old)*I
                # where I is indicator vector for (s,a)
                # For tabular: same formula as given in previous code
                correction = self.alpha * (q - q_old)
                self.Q += self.alpha * (delta + q - q_old) * e
                self.Q[s_id,a_id] -= correction

                q_old = q_next
                e *= self.gamma * self.lambd

                s = next_s
                if not done:
                    s_id = next_s_id
                    a_id = a_next_id

            # After each episode compute MSE
            mse = self.compute_mse()
            MSE_list.append(mse)

        return MSE_list

    def initial_state(self):
        # The environment provides initial_state function
        # or we pick a random valid initial state:
        return self.env.initial_state()

    def compute_mse(self):
        # Compute MSE between V(s)=max_a Q(s,a) and env.optimal_values
        V = {}
        for s in self.states:
            s_id = self.state_id(s)
            V[s] = np.max(self.Q[s_id])
        errors = []
        for s, opt_val in self.env.optimal_values.items():
            # Some states might be furniture (None in opt_val?), skip if not in Q
            if opt_val is not None and s in self.state_to_id:
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
                    a = policy.get(s, None)
                    if a is None:
                        row_str.append("T")
                    else:
                        row_str.append(self.env.action_symbols[a])
            print(" ".join(row_str))


if __name__ == "__main__":
    env = CatVsMonster(gamma=0.925)
    agent = TrueOnlineSarsaCatVsMonster(env, gamma=env.gamma, alpha=2e-4, lambd=0.9, epsilon=0.1, num_episodes=1000000)
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
    plt.title("True Online Sarsa on CatVsMonster: MSE vs Episodes")
    plt.grid(True)
    plt.show()
