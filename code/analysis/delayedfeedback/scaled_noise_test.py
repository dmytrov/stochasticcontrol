""" Test for the control-scaled noise covariance matrix

Reference: "Christopher M. Harris and Daniel M. Wolpert - 1998 - Signal-dependent 
noise determines motor planning":

    "We assume that neural commands have signal-dependent noise
    whose standard deviation increases linearly with the absolute value
    of the neural control signal."
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import analysis.delayedfeedback.targetswitching.model as tsm

torch.set_default_dtype(torch.float64)


vcontrol = 1.0* torch.tensor([3.0, 0.0])  # max covariance direction
scale = torch.tensor([4.0, 1.0])  # covariance scale [max, min]
covariance_matrix, m_control_globalscaled = tsm.signal_dependent_noise_covar_torch(vcontrol, scale)

#vcontrol = 1.0* torch.tensor([3.0, 0.0])  # max covariance direction
#covariance_matrix, m_control_globalscaled = tsm.signal_dependent_noise_covar_xaligned_torch(vcontrol, scale)
#covariance_matrix = torch.diag(covariance_matrix)

u, sigma, v = covariance_matrix.svd()
print("u:", u)
print("sigma^2:", sigma)
std = torch.sqrt(sigma)
assert torch.abs(std[0] / std[1] - scale[0] / scale[1]) < 1.0e-3
print(u @ torch.diag(sigma) @ v)

loc = torch.tensor([0.0, 0.0])


matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
X = torch.tensor(X)
Y = torch.tensor(Y)
gaussian = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
XY = torch.stack([X, Y], dim=-1)
Z = torch.exp(gaussian.log_prob(XY))


plt.figure()
m = 0.2 * m_control_globalscaled
plt.arrow(0, 0, m[0, 0], m[1, 0])
plt.arrow(0, 0, m[0, 1], m[1, 1])
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Control scaled noise')
plt.axis("equal")
plt.grid(True)
plt.show()