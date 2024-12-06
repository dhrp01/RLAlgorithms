import numpy as np

class InvertedPendulum:
    G = 10
    L = 1
    M = 1
    MAX_SPEED = 8
    MAX_TORQUE = 2
    dt = 0.05
    MAX_STEPS = 200

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
