import numpy as np
import random

class MazeWorldMDP:
    def __init__(self, gamma=0.99):
        self.rows = 5
        self.cols = 5
        self.gamma = gamma

        self.start = (0,0)
        self.exit = (4,4)

        self.walls = {(0,1), (2,0), (1,3), (2,2), (3,3), (4,1), (4,3)}

        self.actions = ['AU', 'AD', 'AL', 'AR']

        self.prob_intended = 0.80
        self.prob_perp = 0.05
        self.prob_stay = 0.10

        self.reward_exit = 10.0
        self.reward_step = -1.0
        self.reward_wall = -5.0
        self.reward_revisit = -2.5

        self.visited = set()
        self.current_state = None

        self.optimal_values = {
            (0, 0): -10.76709536613459, (0, 1): -4.51334066119277, (0, 2): -2.420613169758752, (0, 3): -0.30017897894003986, (0, 4): 1.9687581551642528,
            (1, 0): -8.630316267212365, (1, 1): -6.585534193061787, (1, 2): -4.51334066119277, (1, 3): 2.754829185248174, (1, 4): 4.155626803762491,
            (2, 0): -9.939961920379002, (2, 1): -8.747809327813936, (2, 2): 1.6286415299103347, (2, 3): 4.155626803762491, (2, 4): 6.480824111890805,
            (3, 0): -12.130101967353228, (3, 1): -10.132236495489161, (3, 2): -12.130101967353228, (3, 3): 6.480824111890805, (3, 4): 8.728179551122194,
            (4, 0): -14.223242841045892, (4, 1): -11.57917019518519, (4, 2): -14.223242841045893, (4, 3): 8.728179551122194, (4, 4): 0.0
        }

        self.action_symbols = {
            'AU': '↑',
            'AD': '↓',
            'AL': '←',
            'AR': '→',
            None: 'G'
        }

    def reset(self):
        self.current_state = self.start
        self.visited = {self.current_state}
        return self.current_state

    def is_terminal(self, state):
        return state == self.exit

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def attempt_move(self, state, action):
        (r,c) = state
        if action == 'AU':
            nr, nc = r-1, c
        elif action == 'AD':
            nr, nc = r+1, c
        elif action == 'AL':
            nr, nc = r, c-1
        else:
            nr, nc = r, c+1

        if not self.in_bounds(nr, nc) or (nr, nc) in self.walls:
            return state, True
        else:
            return (nr,nc), False

    def left_turn(self, action):
        if action == 'AU':
            return 'AL'
        elif action == 'AD':
            return 'AR'
        elif action == 'AL':
            return 'AD'
        else:
            return 'AU'

    def right_turn(self, action):
        if action == 'AU':
            return 'AR'
        elif action == 'AD':
            return 'AL'
        elif action == 'AL':
            return 'AU'
        else:
            return 'AD'

    def get_next_state_distribution(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        intended_state, wall_hit = self.attempt_move(state, action)
        action_left = self.left_turn(action)
        left_state, left_wall = self.attempt_move(state, action_left)
        action_right = self.right_turn(action)
        right_state, right_wall = self.attempt_move(state, action_right)

        stay_state = state

        transitions = [
            (self.prob_intended, intended_state, wall_hit),
            (self.prob_perp, left_state, left_wall),
            (self.prob_perp, right_state, right_wall),
            (self.prob_stay, stay_state, False)
        ]

        combined = {}
        for (p, s_next, wallflag) in transitions:
            if s_next not in combined:
                combined[s_next] = (p, wallflag)
            else:
                combined[s_next] = (combined[s_next][0] + p, combined[s_next][1] or wallflag)

        result = []
        for s_next, (p, wflag) in combined.items():
            r = self.get_reward(state, s_next, wflag)
            done = self.is_terminal(s_next)
            result.append((p, s_next, r, done))
        return result

    def get_reward(self, s, s_next, wall_attempt):
        if s_next == self.exit:
            return self.reward_exit

        if wall_attempt:
            return self.reward_wall

        r = self.reward_step
        if s_next in self.visited and s_next != self.exit:
            r += self.reward_revisit

        return r

    def step(self, action):
        transitions = self.get_next_state_distribution(self.current_state, action)
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        p, s_next, r, done = transitions[idx]

        self.current_state = s_next
        self.visited.add(s_next)
        return s_next, r, done

    def print_maze(self):
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                s = (r,c)
                if s == self.start:
                    row_str.append('S')
                elif s == self.exit:
                    row_str.append('G')
                elif s in self.walls:
                    row_str.append('X')
                else:
                    row_str.append('.')
            print(" ".join(row_str))


if __name__ == "__main__":
    mdp = MazeWorldMDP()
    mdp.print_maze()
    state = mdp.reset()
    done = False
    total_reward = 0
    while not done:
        action = random.choice(mdp.actions)
        next_state, r, done = mdp.step(action)
        total_reward += r
    print(f"Episode ended. Total reward: {total_reward}")
