""" This file contains utility functions to generate min jerk optimal control trajectories
"""
import numpy as np
import torch


def gen_trajectory_min_jerk(t, 
        x0=(0, 0), 
        x0p=(0, 0), 
        x0pp=(0, 0), 
        x1=(1, 0),
        x1p=(0, 0), 
        x1pp=(0, 0)):
    """ Minimum jerk trajectory.
        The optimal solution is a 5-th order polyomial.
        Boundary conditions:
        start:      x(0) = (0, 0)
        end:        x(1) = (1, 0)
        x'_start:   x'(0) = (0, 0)
        x'_end:     x'(1) = (0, 0)
        x''_start:  x''(0) = (0, 0)
        x''_end:    x''(1) = (0, 0)
        Find 6 coefficients for every dimennsion.
    """
    n = 5
    def basis(t, d=0):
        """ t: time (trajectory parameter)
            d: derivative order
            Retuns: factors for the polynomial coefficients
        """
        a = np.ones(n+1)
        p = np.arange(n+1)
        for i in range(d):
            a *= p
            p = np.clip(p-1, 0, None)
        b = a * np.power([t] * (n+1), p)
        return b

    t0, t1 = t[0], t[-1]  # time interval is [0...1]
    # Basis matrix
    b = np.vstack([
            basis(t0, 0), 
            basis(t1, 0), 
            basis(t0, 1), 
            basis(t1, 1), 
            basis(t0, 2), 
            basis(t1, 2)])
    # Boundary conditions for the basis matrix
    c = np.array([
            x0, 
            x1, 
            x0p,
            x1p,
            x0pp,
            x1pp])
    # Coefficients
    r = np.dot(np.linalg.inv(b), c)
    # Trajectory
    x = np.sum(np.stack([r[i, :] * np.power(t[:, None], i) for i in range(n+1)]), axis=0)
    xp = np.sum(np.stack([r[i, :] * i * np.power(t[:, None], i-1) for i in range(1, n+1)]), axis=0)
    xpp = np.sum(np.stack([r[i, :] * i * (i-1) * np.power(t[:, None], i-2) for i in range(2, n+1)]), axis=0)
    return x, xp, xpp


def gen_trajectory_min_jerk_disturbed(t, fraction=0.5, x_target=(0, 1), x_target_disturbed=(-0.3, 1.0)):
    ptstart = (0, 0)
    x, xp, xpp = gen_trajectory_min_jerk(t, x0=ptstart, x1=x_target)
    n = len(x)
    i = int(fraction * n)  # integer division
    x_1 = x[:i]
    xp_1 = xp[:i]  # integer division
    xpp_1 = xpp[:i]  # integer division
    x_2, xp_2, xpp_2 = gen_trajectory_min_jerk(t[i-1:], x0=x_1[-1], x0p=xp_1[-1], x1=x_target_disturbed)
    return np.concatenate([x_1, x_2[1:]], axis=0), \
           np.concatenate([xp_1, xp_2[1:]], axis=0), \
           np.concatenate([xpp_1, xpp_2[1:]], axis=0)


def add_noise(x, scale=0.1):
    return x + np.reshape(np.random.normal(scale=scale, size=x.size), x.shape)


if __name__ == "__main__":
    ## Test plots

    # Min-jerk straight trajectory
    import matplotlib.pyplot as plt
    ptstart = (0, 0)
    ptend = (0, 1)
    t=np.arange(0.0, 1.0, 0.01)
    x, xp, xpp = gen_trajectory_min_jerk(t, x0=ptstart, x1=ptend)
    plt.plot(x[:, 0], x[:, 1], "o-", alpha=0.3)
    plt.show()

    # Min-jerk disturbed trajectory
    x, xp, xpp = gen_trajectory_min_jerk_disturbed(t)
    plt.plot(x[:, 0], x[:, 1], "o-", alpha=0.3)
    plt.axis("equal")
    plt.show()
    