"""

  Author: Dmytro Velychko, Philipps University of Marburg
  velychko@staff.uni-marburg.de
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation


fig = plt.gcf()
axis = plt.gca()

points, = axis.plot([], [], 'bo', ms=3)
particles = np.stack([np.random.uniform(-1, 1, 1000), np.random.uniform(-1, 1, 1000)], axis=-1)
support = mpatches.Rectangle(xy=[-1, -1], width=2, height=2, color="orange", alpha=0.2) 
axis.add_patch(support)
circle = mpatches.Circle(xy=[0, 0], radius=1, ec="none", alpha=0.5)
axis.add_patch(circle)
info_text = axis.text(0.02, 0.9, '', transform=axis.transAxes)
plt.axis('equal')


def init():
    points.set_data([], [])
    return points,


def animate(N):
    N = N + 1
    disk = lambda x: 1.0 * np.less(np.linalg.norm(x, axis=-1), 1.0)
    pi_hat = 4 * np.sum(disk(particles[:N])) / N 
    points.set_data(particles[:N, 0], particles[:N, 1])
    info_text.set_text("N={}\nPi={:.4f}".format(N, pi_hat))
    return points, info_text


anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=1000, interval=20, blit=True)
plt.show()
