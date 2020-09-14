""" Monte Carlo integration.
    Estimating Pi constant with MC.
    Requires python v3.x

  Author: Dmytro Velychko, Philipps University of Marburg
  velychko@staff.uni-marburg.de
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def mc_integrate_uniform(f, support, N):
    """ Monte Carlo integration by uniform sampling.
        f: callable function, accepts array of imputs
        support: list of boundaries for every dimensoin
        N: int, number of samples
    """
    D = len(support)
    particles = np.stack([np.random.uniform(s[0], s[1], N) for s in support], axis=-1)
    y = f(particles)
    return np.prod([s[1]-s[0] for s in support]) * np.sum(y) / N


def mc_integrate_importance(f, D, N):
    """ Monte Carlo integration by importance sampling.
        f: callable function, accepts array of imputs
        D: number of dimensions
        N: int, number of samples
    """
    loc = 0.0
    scale = 1.0
    particles = np.stack([np.random.normal(loc=loc, scale=scale, size=N) for d in range(D)], axis=-1)
    p_particles = np.prod(st.norm.pdf(particles, loc=loc, scale=scale), axis=-1)
    f_particles = f(particles)
    return np.sum(f_particles/p_particles) / N


disk = lambda x: 1.0 * np.less(np.linalg.norm(x, axis=-1), 1.0)
f_pi_hats = [("Uniform sampling", lambda N: mc_integrate_uniform(f=disk, support=[(-1, 2), (-2, 2)], N=N)), 
             ("Importance sampling", lambda N: mc_integrate_importance(f=disk, D=2, N=N))]

for title, f_pi_hat in f_pi_hats:
    log_n = np.linspace(1, 14, 100)
    pi_hats = [f_pi_hat(int(n)) for n in np.exp(log_n)]
    plt.plot(log_n, np.pi * np.ones_like(pi_hats), "--")
    plt.plot(log_n, pi_hats)
    plt.xlabel("log(N)")
    plt.title("Monte Carlo Pi. " + title)
    plt.show()

