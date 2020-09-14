import numpy as np


def rbf_kernel(x0, x1, tau):
    assert len(x0.shape) == 2
    assert len(x1.shape) == 2
    return np.exp( (-0.5/(tau**2)) * np.sum(x0[:, np.newaxis, :] - x1[np.newaxis, :, :], axis=-1)**2)

def uniform_kernel(x0, x1, tau):
    assert len(x0.shape) == 2
    assert len(x1.shape) == 2
    return (np.abs(x0[:, np.newaxis] - x1[np.newaxis, :]) < tau).astype(float)


class LocallyLinearRegression(object):
    def __init__(self, x, y, tau=0.1):
        assert len(x) == len(y)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        xy = np.hstack([x, y])
        valid = np.logical_not(np.any(np.isnan(xy), axis=-1))
        x = x[valid]
        y = y[valid]
        assert np.alltrue(np.logical_not(np.isnan(y)))
        self.x = x.copy()
        self.y = y.copy()
        self.tau = tau
        self._xones = np.hstack([self.x, np.ones([len(self.x), 1])])
        self.kernel = rbf_kernel

    def regress(self, xstar):
        
        xstar = np.array([xstar])
        if len(xstar.shape) == 1:
            xstar = xstar[:, np.newaxis] 
        xstarones = np.vstack([xstar, np.ones(len(xstar))]).T
        k = self.kernel(self.x, xstar, self.tau)
        # Estimate mean
        xones_k = self._xones * k
        ab = np.linalg.pinv(xones_k.T.dot(self._xones)).dot(xones_k.T).dot(self.y)
        ystar = xstarones.dot(ab)
        
        # Estimate covariance
        residuals = (self._xones.dot(ab) - self.y)
        residualssqr_k = residuals**2 * k
        covarstar = np.sum(residualssqr_k, axis=0) / np.sum(k)
        return ystar[0], covarstar 


    def __call__(self, xstar):
        return self.regress(xstar)

