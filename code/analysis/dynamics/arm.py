""" Two-link arm dynamics model.

    See:
    Iterative Linear Quadratic Regulator Design
    for Nonlinear Biological Movement Systems.
        Weiwei Li, Emanuel Todorov
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

# In pytorch:
# stack() for new dimension
# cat() for existing dimension
        

class TwoLinkArm(object):
    def __init__(self):
        self.b11 = 0.05
        self.b22 = 0.05
        self.b12 = 0.025
        self.b21 = 0.025

        self.m1 = 1.4
        self.m2 = 1.0

        self.l1 = 0.30  # m
        self.l2 = 0.33  # m

        self.s1 = 0.11  # m
        self.s2 = 0.16  # m

        self.I1 = 0.025
        self.I2 = 0.045

        self.theta = torch.tensor([0.0, 0.0])
        self.dtheta = torch.tensor([0.0, 0.0])
  

    def _ddtheta(self, u):
        """ Angle velocity dynamics
            u: tensor[2] - torque (control)
        """
        a1 = self.I1 + self.I2 + self.m2 * self.l1**2
        a2 = self.m2 * self.l1 * self.s2
        a3 = self.I2

        theta1, theta2 = self.theta
        dtheta1, dtheta2 = self.dtheta

        M = torch.stack([torch.tensor([a1 + 2.0 * a2 * torch.cos(theta2), a3 + a2*torch.cos(theta2)]),
                         torch.tensor([a3 + a2*torch.cos(theta2), a3])])
        C = torch.stack([-dtheta2*(2.0*dtheta1 + dtheta2), dtheta1**2]) * a2*torch.sin(theta2)
        B = torch.tensor([[self.b11, self.b12], [self.b21, self.b22]])

        return torch.mv(M.inverse(), (u - C - torch.mv(B, self.dtheta)))


    def get_x(self):
        return torch.cat([self.theta, self.dtheta])

    def set_x(self, x):
        self.theta = x[:2]
        self.dtheta = x[2:]


    def get_dx(self, u):
        """ State dynamics
            u: tensor[2] - torque (control)
        """
        return torch.cat([self.dtheta, self._ddtheta(u)])


    def step(self, u, dt=0.01):
        """ Integrate the state
            u: tensor[2] - torque (control)
        """
        self.set_x(self.get_x() + dt * self.get_dx(u))


    def get_position(self):
        def m_rot(angle):
            return np.array([[np.cos(angle), np.sin(angle)], 
                             [-np.sin(angle), np.cos(angle)]])
        theta = self.theta.data.numpy()
        theta1, theta2 = theta
        root = np.zeros(2)
        elbow = np.dot(m_rot(theta1), np.array([self.l1, 0.0]))
        wrist = np.dot(m_rot(theta1), np.array([self.l1, 0.0]) + np.dot(m_rot(theta2), np.array([self.l2, 0.0])))
        return root, elbow, wrist


    def plot(self, axes):
        root, elbow, wrist = self.get_position()
        path = np.stack([root, elbow, wrist])
        axes.plot(path[:, 0], path[:, 1], alpha=0.3)
        axes.scatter(path[:, 0], path[:, 1], alpha=0.5)


class MuscleActivation(object):
    def __init__(self):
        pass

    def _tension(self, a, l, v):
        """ Muscle tension
            a: activation
            l: lenght
            v: velocity
        """
        Fp = -0.02 * torch.exp(13.8 - 18.7*l)
        Fv = torch.where(v <= 0, 
                (-5.72-v) / (-5.72 + (1.38 + 2.09)*l),
                (0.62 - (-3.12+4.21*l-2.67*l**2)) / (0.62+v))
        Fl = torch.exp(-(torch.abs((l**1.93-1)/1.03))**1.87)
        Nf = 2.11 + 4.16*(1/l - 1)
        A = 1.0 - torch.exp(-(a/(0.56*Nf))**Nf)
        T = A * (Fl * Fv + Fp)
        return T

    

if __name__ == "__main__":
    
    arm = TwoLinkArm()
    u = torch.tensor([-4.0, -2.0])

    plt.figure()
    for i in range(80):
        arm.step(u)
        arm.plot(plt.gca())

    plt.axis("equal")
    plt.show()






