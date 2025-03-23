import numpy as np
import math

class ManiuplatorModel:
    def __init__(self, Tp, m3 = None, r3 = None):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3 or 0.5
        self.r3 = r3 or 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        d1 = 1/2 * self.l1
        d2 = 1/2 * self.l2

        alpha = self.m1*(d1**2) + self.I_1 + self.m2*(self.l1**2 + d2**2) + self.I_2 + self.m3*(self.l1**2+self.l2**2) + self.I_3
        beta = self.m2*self.l1*d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2*d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        c2 = math.cos(q2)

        m11 = alpha + 2*beta*c2
        m12 = gamma + beta*c2

        m21 = gamma + beta*c2
        m22 = gamma

        M = np.array([[m11, m12], [m21, m22]])

        return M

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        d2 = 1/2 * self.l2

        beta = self.m2*self.l1*d2 + self.m3 * self.l1 * self.l2

        s2 = math.sin(q2)

        return np.array([[-beta * s2*q2_dot, -beta*s2*(q1_dot + q2_dot)], [beta*s2*q1_dot, 0]])



    def q_ddot(self, x, u):
        M = self.M(x)
        C = self.C(x)
        invM = np.linalg.inv(M)
        A = -invM@C
        B = invM

        q_dot = np.array([[x[2]], [x[3]]])

        return A@q_dot + B@u