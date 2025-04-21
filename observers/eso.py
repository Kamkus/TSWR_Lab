from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z_est = np.reshape(self.state, (len(self.state), 1))
        print(u)
        z_est_dot = self.A @ z_est + self.B*u + self.L @ (q - self.W @ z_est)
        self.state = self.state + self.Tp * np.reshape(z_est_dot, (z_est_dot.shape[0]))

    def get_state(self):
        return self.state
