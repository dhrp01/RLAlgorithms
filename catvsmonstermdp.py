import numpy as np
import random

class CatVsMonster:

    def __init__(self, gamma: float = 0.925):
        self.rows = 5
        self.cols = 5
        self.gamma = gamma
        self.actions = ['AU', 'AD', 'AL', 'AR']
        self.n_actions = len(self.actions)
        self.furniture = [(2,1), (2,2), (2,3), (3,2)]
        self.monsters = [(0,3), (4,1)]
        self.food = (4,4)
        self.reward_step = -0.05
        self.reward_food = 10
        self.reward_monster = -8
        self.action_symbols = {
            "AU": "↑",
            "AD": "↓",
            "AL": "←",
            "AR": "→"
        }


    def get_next_state(self, state, action):
        if state == self.food:
            return state

        row, col = state
        prob = random.random()
        if prob < 0.70:
            if action == "AU": new_row, new_col = row - 1, col
            elif action == "AD": new_row, new_col = row + 1, col
            elif action == "AL": new_row, new_col = row, col - 1
            else: new_row, new_col = row, col + 1
        elif prob < 0.82:
            if action == "AU": new_row, new_col = row, col + 1
            elif action == "AD": new_row, new_col = row, col - 1
            elif action == "AL": new_row, new_col = row - 1, col
            else: new_row, new_col = row + 1, col
        elif prob < 0.94:
            if action == "AU": new_row, new_col = row, col - 1
            elif action == "AD": new_row, new_col = row, col + 1
            elif action == "AL": new_row, new_col = row + 1, col
            else: new_row, new_col = row - 1, col
        else:
            return state

        if (new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols or (new_row, new_col) in self.furniture):
            return state

        return (new_row, new_col)

    def get_reward(self, next_state):
        if next_state == self.food:
            return self.reward_food
        elif next_state in self.monsters:
            return self.reward_monster
        else:
            return self.reward_step

    def is_terminal(self, state):
        return state == self.food

    def compute_max_norm(self, v1, v2):
        max_diff = 0
        for state in v1:
            if v2.get(state) is not None:
                max_diff = max(max_diff, abs(v1[state] - v2[state]))
        return max_diff

    def generate_episode(self, state):
        episode = []
        while not self.is_terminal(state):
            action = self.optimal_policy[state]
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            episode.append((state, action, reward, next_state))
            state = next_state
        return episode

    def initial_state(self):
        states =  [(r,c) for r in range(self.rows) for c in range(self.cols) if (r,c) not in self.furniture and (r,c) != self.food]
        return random.choice(states)

    def generate_episode_epsilon(self, policy):
        episode = []
        state = self.initial_state()
        while True:
            action_prob = policy[state]
            action = random.choices(self.actions, weights=action_prob, k=1)[0]
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            episode.append((state, action, reward, next_state))
            if self.is_terminal(next_state) or next_state in self.monsters:
                break
            state = next_state
        return episode
