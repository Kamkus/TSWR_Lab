import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
# from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array(
             [
                 [3 * p[0], 0],
                 [0, 3 * p[1]],
                 [3 * p[0] ** 2, 0],
                 [0, 3 * p[1] ** 2],
                 [p[0] ** 3, 0],
                 [0, p[1] ** 3],
             ]
        )
        W = W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        self.A = np.array(
             [
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             ]
        )
        self.B = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.eso = ESO(self.A, self.B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.array([q[0], q[1], q_dot[0], q_dot[1]]) 
        M = self.model.M(x)
        C = self.model.C(x)
        inv_M = np.linalg.inv(M)
        A = self.A.copy()
        B = self.B.copy()
        A[2:4, 2:4] = -inv_M @ C
        B[2:4, 0:2] = inv_M
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[:2]
        q_dot = x[2:]
        state = self.eso.get_state()
        q_h = state[:2]
        q_h_dot = state[2:4]
        f = state[4:]
        x_hat = np.array([q_h[0], q_h[1], q_h_dot[0], q_h_dot[1]])
        M = self.model.M(x_hat)
        C = self.model.C(x_hat)
        v = q_d_ddot + self.Kd @ (q_d_dot - q_dot) + self.Kp @ (q_d - q)
        u = M @ (v - f) + C @ q_h_dot
        self.update_params(q_h, q_h_dot)
        self.eso.update(q, u)
        return u