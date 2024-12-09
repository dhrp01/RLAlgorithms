import numpy as np
import matplotlib.pyplot as plt
import math
import random
from gridworldmdp import GridWorldMDP

class SemiGradientNStepSarsaAgent:
    def __init__(self, mdp, n=5, alpha=0.01, epsilon=0.1, gamma=None, num_episodes=5000):
        self.mdp = mdp
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = mdp.gamma if gamma is None else gamma
        self.num_episodes = num_episodes

        # State-Action feature dimension: 25 states * 4 actions = 100
        self.num_states = self.mdp.rows * self.mdp.cols
        self.num_actions = self.mdp.n_actions
        self.feature_dim = self.num_states * self.num_actions

        # Initialize weights
        self.w = np.zeros(self.feature_dim)

        # Actions
        self.action_list = self.mdp.actions

    def state_to_id(self, s):
        r, c = s
        return r * self.mdp.cols + c

    def sa_to_feature(self, s, a):
        # One-hot encoding:
        # index = state_id * num_actions + action_id
        s_id = self.state_to_id(s)
        a_id = self.action_list.index(a)
        x = np.zeros(self.feature_dim)
        index = s_id * self.num_actions + a_id
        x[index] = 1.0
        return x

    def Q(self, s, a):
        x = self.sa_to_feature(s,a)
        return np.dot(self.w, x)

    def epsilon_greedy(self, s):
        if random.random() < self.epsilon:
            return random.choice(self.action_list)
        else:
            # Choose best action w.r.t current Q
            q_values = [self.Q(s,a) for a in self.action_list]
            max_q = max(q_values)
            best_actions = [a for a, qv in zip(self.action_list, q_values) if qv == max_q]
            return random.choice(best_actions)

    def generate_episode(self):
        # Start from start state
        s = self.mdp.start
        episode = []
        a = self.epsilon_greedy(s)

        done = False
        steps = 0
        while not done:
            # Take action a
            transitions = self.mdp.get_next_state_distribution(s,a)
            # This MDP is stochastic, but transitions is a distribution. We sample from it.
            probs = [tr[0] for tr in transitions]
            idx = np.random.choice(range(len(transitions)), p=probs)
            p, s_next, r, done = transitions[idx]

            episode.append((s,a,r))
            if not done:
                a_next = self.epsilon_greedy(s_next)
            else:
                a_next = None

            s = s_next
            a = a_next
            steps += 1
            if done:
                break
        return episode

    def run_n_step_sarsa(self):
        # n-step Sarsa using semi-gradient update
        MSE_list = []
        for ep in range(self.num_episodes):
            episode = self.generate_episode()
            # episode is a list of (S_t, A_t, R_{t+1})
            T = len(episode)
            # We will use indices t for states and actions: S_t,A_t,R_{t+1}
            # For convenience, define an array of states, actions, rewards:
            S = [e[0] for e in episode] + [None]  # final S_{T} is None
            A = [e[1] for e in episode] + [None]
            R = [e[2] for e in episode]

            for t in range(T):
                tau = t - self.n + 1
                if tau < 0:
                    continue
                # Compute G
                G = 0.0
                gamma_power = 1.0
                for k in range(tau+1, min(tau+self.n+1, T+1)):
                    G += gamma_power * R[k-1]  # R_{k} is at R[k-1] since R_1=R[0]
                    gamma_power *= self.gamma
                if tau + self.n < T:
                    # add gamma^n Q(S_{tau+n},A_{tau+n})
                    S_tau_n = S[tau+self.n]
                    A_tau_n = A[tau+self.n]
                    if A_tau_n is not None:
                        G += gamma_power * self.Q(S_tau_n, A_tau_n)

                # Update weights
                S_tau = S[tau]
                A_tau = A[tau]
                if A_tau is not None:
                    Q_tau = self.Q(S_tau, A_tau)
                    x = self.sa_to_feature(S_tau, A_tau)
                    self.w += self.alpha * (G - Q_tau) * x

            # After each episode, compute MSE
            mse = self.compute_mse()
            MSE_list.append(mse)

        return MSE_list

    def compute_mse(self):
        # Compute MSE for state values.
        # State value V(s) = max_a Q(s,a) and compare with mdp.optimal_values.
        errors = []
        for s, opt_val in self.mdp.optimal_values.items():
            if opt_val is not None:
                q_vals = [self.Q(s,a) for a in self.action_list]
                V_est = max(q_vals)
                errors.append((V_est - opt_val)**2)
        if len(errors) == 0:
            return 0.0
        return sum(errors)/len(errors)

    def get_value_function(self):
        # V(s)=max_a Q(s,a)
        V = {}
        for r in range(self.mdp.rows):
            for c in range(self.mdp.cols):
                s = (r,c)
                q_vals = [self.Q(s,a) for a in self.action_list]
                V[s] = max(q_vals)
        return V

    def derive_policy(self):
        policy = {}
        for r in range(self.mdp.rows):
            for c in range(self.mdp.cols):
                s = (r,c)
                if self.mdp.is_terminal(s):
                    policy[s] = None
                    continue
                q_vals = [self.Q(s,a) for a in self.action_list]
                max_q = max(q_vals)
                best_actions = [a for a, qv in zip(self.action_list, q_vals) if qv == max_q]
                policy[s] = random.choice(best_actions)
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
    # Assuming GridWorldMDP is defined as per user's code.
    from collections import defaultdict
    random.seed(0)
    np.random.seed(0)

    mdp = GridWorldMDP(gamma=0.9)
    agent = SemiGradientNStepSarsaAgent(mdp, n=2, alpha=0.05, epsilon=0.5, num_episodes=1000000, gamma=mdp.gamma)
    mse_list = agent.run_n_step_sarsa()

    # Print final values
    agent.print_values()

    # Print final policy
    final_policy = agent.derive_policy()
    agent.print_policy(final_policy)

    # Plot learning curve (MSE vs Episodes)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(mse_list)+1), mse_list)
    plt.xlabel("Episodes")
    plt.ylabel("MSE")
    plt.title("Episodic Semi-Gradient n-step Sarsa Learning Curve")
    plt.grid(True)
    plt.show()
