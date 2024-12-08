import numpy as np

class InvertedPendulum:
    G = 10
    L = 1
    M = 1
    MAX_SPEED = 8
    MAX_TORQUE = 2
    dt = 0.05
    MAX_STEPS = 200
    NUM_OMEGA_BINS = 61
    NUM_OMEGA_DOT_BINS = 61
    ACTIONS = [-2.0, 0.0, 2.0]
    NUM_ACTIONS = len(ACTIONS)
    OMEGA_BINS = np.linspace(-np.pi, np.pi, NUM_OMEGA_BINS)
    OMEGA_DOT_BINS = np.linspace(-8, 8, NUM_OMEGA_DOT_BINS)

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.omega = np.random.uniform(-5*np.pi/6, 5 * np.pi/6)
        self.omega_dot = np.random.uniform(-1,1)
        self.steps = 0
        return np.array([self.omega, self.omega_dot])

    def step(self, action):
        action = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)
        omega_dot_dot = ((3*self.G)/(2*self.L)*np.sin(self.omega)) + ((3*action)/(self.M*self.L**2))
        self.omega_dot = np.clip(self.omega_dot + omega_dot_dot*self.dt, -self.MAX_SPEED, self.MAX_SPEED)
        self.omega = self.omega + self.omega_dot*self.dt

        self.steps += 1
        terminal_state = self.steps >= self.MAX_STEPS

        omega_normalized = ((self.omega+np.pi)%(2*np.pi)) - np.pi
        reward = -((omega_normalized)**2 + 0.1*(self.omega_dot**2) + 0.001*(action**2))

        return np.array([self.omega, self.omega_dot]), reward, terminal_state

    def set_state(self, omega, omega_dot):
        self.omega = omega
        self.omega_dot = omega_dot

    def discretize_state(self, state):
        omega, omega_dot = state
        omega_idx = np.digitize([omega], self.OMEGA_BINS)[0] - 1
        omega_dot_idx = np.digitize([omega_dot], self.OMEGA_DOT_BINS)[0] - 1
        omega_idx = np.clip(omega_idx, 0, self.NUM_OMEGA_BINS-1)
        omega_dot_idx = np.clip(omega_dot_idx, 0, self.NUM_OMEGA_DOT_BINS-1)
        return (omega_idx, omega_dot_idx)

    def state_from_idx(self, i, j):
        omega_val = (self.OMEGA_BINS[i] + (self.OMEGA_BINS[i+1] if i+1<self.NUM_OMEGA_BINS else self.OMEGA_BINS[i]))/2 if i+1<self.NUM_OMEGA_BINS else self.OMEGA_BINS[i]
        omega_dot_val = (self.OMEGA_DOT_BINS[j] + (self.OMEGA_DOT_BINS[j+1] if j+1<self.NUM_OMEGA_DOT_BINS else self.OMEGA_DOT_BINS[j]))/2 if j+1<self.NUM_OMEGA_DOT_BINS else self.OMEGA_DOT_BINS[j]
        return omega_val, omega_dot_val
