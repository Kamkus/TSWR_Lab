import numpy as np

from models.manipulator_model import ManiuplatorModel
from .adrc_joint_controller import ADRCJointController
from .controller import Controller


class ADRController(Controller):
    def __init__(self, Tp, params):
        self.joint_controllers = []
        self.model = ManiuplatorModel(Tp)
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        u = []
        M = self.model.M(x)
        invM = np.linalg.inv(M)
        # print(np.diagonal(M))
        for i, controller in enumerate(self.joint_controllers):
            controller.set_b(np.diagonal(M)[i])
            u.append(controller.calculate_control([x[i], x[i+2]], q_d[i], q_d_dot[i], q_d_ddot[i]))
        u = np.array(u)[:, np.newaxis]
        return u

