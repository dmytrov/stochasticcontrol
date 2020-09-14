import numpy as np


def kernel(x0, x1):
    a1 = 0.1  # scale
    a2 = 0.1  # length
    a3 = 100.01 # linear
    a4 = 0.01   # noise

    distsqr = np.sum((x0[np.newaxis, :, :] - x1[:, np.newaxis, :])**2, axis=2)
    k = a1 * np.exp(-0.5 * distsqr * (1.0/a2))
    k += a3 * np.dot(x0, x1.T).T
    if x0.shape == x1.shape:
        k += a4 * np.identity(x0.shape[0])
    return k.T


def conditional(x, y, xstar, kfunction=kernel):
    xmean = np.mean(x, axis=0)
    xzeromean = x - xmean
    xstarzeromean = xstar - xmean
    ymean = np.mean(y, axis=0)
    yzeromean = y - ymean
    Kxstarx = kfunction(xstarzeromean, xzeromean)
    Kinv = np.linalg.inv(kfunction(xzeromean, xzeromean))
    ystar_mean = ymean + Kxstarx.dot(Kinv.dot(yzeromean))
    return ystar_mean
    #ystar_cov = kfunction(xstar, xstar) - Kxstarx.dot(Kinv.dot(Kxstarx.T))

if __name__ == "__main__":
    x = 1.0 / 120 * np.array([[0.0, 1.0, 3.0, 4.0]]).T
    y = 0 + np.array([[0.0, 1.0, 3.0, 4.0],
                     [0.00, 0.01, 0.03, 0.04]]).T

    xstar = 1.0 / 120 * np.array([[2.0]])
    ystar = conditional(x, y, xstar)
    print(ystar)

    xstar = 1.0 / 120 * np.array([[100.0]])
    ystar = conditional(x, y, xstar)
    print(ystar)