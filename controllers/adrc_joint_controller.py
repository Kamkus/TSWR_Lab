import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [b],
                      [0]])
        L = np.array([[3 * p], [3 * p**2], [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        _, q_h_dot, f = self.eso.get_state()
        v = self.kp * (q_d - x[0]) +  self.kd * (q_d_dot - q_h_dot) + q_d_ddot 
        u = (v - f) / self.b
        self.eso.update(x[0], u)
        return u