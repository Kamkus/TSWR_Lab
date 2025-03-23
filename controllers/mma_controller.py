import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        self.models = [ManiuplatorModel(Tp, m3=0.1, r3=0.05), ManiuplatorModel(Tp, m3=0.01, r3=0.01), ManiuplatorModel(Tp, m3=1.0, r3=0.3)]
        self.i = 0
        self.Tp = Tp
        self.prevtau = np.array([[0], [0]]) 
        self.prevx = [0, 0, 0, 0]

    def choose_model(self, x):
        # q_dot = x[2:]
        # errors = np.array([])
        # for model in self.models:
        #     model_prediction = model.M(x) @ self.v + model.C(x) @ q_dot[:, np.newaxis]
        #     errors = np.append(errors, np.sum(np.abs(q_dot - model_prediction)))
        # return np.argmin(errors)

        errors = np.array([])
        for model in self.models:
            q_ddot = model.q_ddot(self.prevx, self.prevtau)
            q_dot = np.array([[self.prevx[2]], [self.prevx[3]]]) + self.Tp*q_ddot
            q = np.array([[self.prevx[0]], [self.prevx[1]]]) + self.Tp*q_dot
            errors = np.append(errors, np.abs(np.sum(x - np.array([q[0], q[1], q_dot[0], q_dot[1]]))))
        return np.argmin(errors)


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.i = self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        Kd = 1.5
        Kp = 4

        v = q_r_ddot + Kd * (q_r_dot - q_dot) + Kp * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prevtau = u
        self.prevx = x
        return u
