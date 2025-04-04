import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = x[:2]
        q_dot = x[2:]

        KD, KP = -1, 0.5
        v = q_r_ddot + KD*(q_dot - q_r_dot) + KP*(q - q_r)
        return self.model.M(x)@v + self.model.C(x)@q_dot