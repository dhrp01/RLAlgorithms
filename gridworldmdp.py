import random

class GridWorldMDP:
    def __init__(self, gamma=0.9):
        self.rows = 5
        self.cols = 5
        self.gamma = gamma
        self.actions = ['AU', 'AD', 'AL', 'AR']
        self.n_actions = len(self.actions)

        self.start = (0,0)
        self.goal = (4,4)
        self.water = (4,2)
        self.obstacles = [(2,2), (3,2)]

        self.prob_intended = 0.80
        self.prob_right = 0.05
        self.prob_left = 0.05
        self.prob_stay = 0.10

        self.reward_water = -10
        self.reward_goal = 10
        self.reward_step = -1

        self.left_turn = {"AU": "AL", "AD": "AR", "AL": "AD", "AR": "AU"}
        self.right_turn = {"AU": "AR", "AD": "AL", "AL": "AU", "AR": "AD"}

        self.action_symbols = {
            "AU": "↑",
            "AD": "↓",
            "AL": "←",
            "AR": "→"
        }

        self.optimal_values = {
            (0, 0): -1.9626199661914625, (0, 1): -0.8904317898358831, (0, 2): 0.31508917414718624, (0, 3): 1.6672715831268463, (0, 4): 2.910575789148558,
            (1, 0): -1.256786266549504, (1, 1): 0.06471769813916807, (1, 2): 1.6025910174955873, (1, 3): 3.2945308573572976, (1, 4): 4.781417828295584,
            (2, 0): -2.2656521487860557, (2, 1): -1.2200665942695872, (2, 2): 3.217178355978997, (2, 3): 5.153809280742425, (2, 4): 6.927322962353615,
            (3, 0): -3.163465982281034, (3, 1): -2.336189499530575, (3, 2): 4.524444223933264, (3, 3): 7.147660409078173, (3, 4): 9.38918464555898,
            (4, 0): -3.966076302894255, (4, 1): -3.399122103453835, (4, 2): 4.829527981466422, (4, 3): 9.389184645558979, (4, 4): 0.0
        }
        self.optimal_policy = {
            (0, 0): 'AR', (0, 1): 'AR', (0, 2): 'AR', (0, 3): 'AD', (0, 4): 'AD',
            (1, 0): 'AR', (1, 1): 'AR', (1, 2): 'AR', (1, 3): 'AD', (1, 4): 'AD',
            (2, 0): 'AU', (2, 1): 'AU', (2, 2): 'AR', (2, 3): 'AD', (2, 4): 'AD',
            (3, 0): 'AU', (3, 1): 'AU', (3, 2): 'AR', (3, 3): 'AD', (3, 4): 'AD',
            (4, 0): 'AU', (4, 1): 'AU', (4, 2): 'AR', (4, 3): 'AR', (4, 4): None
        }

    def is_terminal(self, state):
        return state == self.goal

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, s):
        return s in self.obstacles

    def attempt_move(self, state, action):
        (r,c) = state
        if action == "AU":
            nr, nc = r-1, c
        elif action == "AD":
            nr, nc = r+1, c
        elif action == "AL":
            nr, nc = r, c-1
        else:
            nr, nc = r, c+1

        if not self.in_bounds(nr, nc) or self.is_obstacle((nr,nc)):
            return state
        return (nr, nc)

    def get_next_state_distribution(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        intended_state = self.attempt_move(state, action)
        action_left = self.left_turn[action]
        state_left = self.attempt_move(state, action_left)
        action_right = self.right_turn[action]
        state_right = self.attempt_move(state, action_right)
        state_stay = state

        transitions = [
            (self.prob_intended, intended_state),
            (self.prob_left, state_left),
            (self.prob_right, state_right),
            (self.prob_stay, state_stay)
        ]

        combined = {}
        for p, s_next in transitions:
            if s_next not in combined:
                combined[s_next] = p
            else:
                combined[s_next] += p

        result = []
        for s_next, p in combined.items():
            r = self.get_reward(s_next)
            done = self.is_terminal(s_next)
            result.append((p, s_next, r, done))

        return result

    def get_reward(self, state):
        if state == self.goal:
            return self.reward_goal
        elif state == self.water:
            return self.reward_water
        else:
            return self.reward_step

    def print_grid(self):
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                s = (r,c)
                if s == self.start:
                    row_str.append('S')
                elif s == self.goal:
                    row_str.append('G')
                elif s == self.water:
                    row_str.append('W')
                elif s in self.obstacles:
                    row_str.append('X')
                else:
                    row_str.append('.')
            print(' '.join(row_str))

if __name__ == "__main__":
    mdp = GridWorldMDP()
    mdp.print_grid()

    transitions = mdp.get_next_state_distribution(mdp.start, 'AR')
    print("Transitions from start going right:")
    for t in transitions:
        p, s_next, r, done = t
        print(f"  prob={p:.2f}, next={s_next}, reward={r}, done={done}")
