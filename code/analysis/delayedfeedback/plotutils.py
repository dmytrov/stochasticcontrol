
import numpy as np
import matplotlib.pyplot as plt
import analysis.delayedfeedback.regression as re


def plot_mean_var(x, var, color="b", ax=plt):
    """ Plot mean line and confidence intervals.
            x : [N, 2]
            var : [N] - variance along the line (+-)
    """
    assert len(x) == len(var)
    x = np.array(x)
    var = np.array(var)
    # Compute normals to the mean line
    dx = np.diff(x, axis=0)
    dx = np.vstack([dx, dx[-1]])
    dx = dx / np.linalg.norm(dx, axis=-1)[:, np.newaxis]
    mnormal = np.array([[0, 1], [-1, 0]])
    dxnormal = dx.dot(mnormal)
    if len(var.shape) == 2:
        # Transform covariance vector from local to global coordinates
        vbasis = np.stack([dx, dxnormal], axis=2)
        vbasisinv = np.linalg.inv(vbasis)
        var_global = np.array([basis.T.dot(np.diag(c)).dot(basis) for c, basis in zip(var, vbasis)])
        var = var_global[:, 1, 1]
    
    xleft = x - dxnormal * var[:, np.newaxis]
    xright = x + dxnormal * var[:, np.newaxis]

    polyline = np.vstack([xleft, np.flip(xright, axis=0)])
    p = plt.Polygon(polyline, closed=True, fill=True, alpha=0.2, color=color)
    ax.plot(x[:, 0], x[:, 1], color=color)
    try:
        ax.add_patch(p)
    except:
        plt.gca().add_patch(p)
    ax.axis("equal")


def plot_average(xys, tau=0.05, color="b", nsigma=1, ax=plt):
    """ Plots x/y data and regressed average with variance.
            xys : [(x, y)] - list of (x, y) tuples.
                x : [N] - vector
                y : [N, D] - N data points. D \in [1..2] or
                    [N] - vector
            nsigma : number of sigmas to include into error band
    """
    def ensure_2d(x):
        if len(x.shape) == 1:
            return x[..., np.newaxis]
        elif len(x.shape) == 2:
            return x
        else:
            raise ValueError("Must be 1 or 2 dimensinoal!")

    if len(xys) == 0:
        return
    llr = re.LocallyLinearRegression(
            np.hstack([x for x, y in xys]),
            np.vstack([ensure_2d(y) for x, y in xys]),
            tau=tau)
    
    # Compute x0, y0, c0
    x0 = np.linspace(np.min(llr.x), np.max(llr.x), 100)
    y0_mean_covar = [llr.regress(x) for x in x0]
    y0 = np.array([m for m, c in y0_mean_covar])
    var0 = np.squeeze(np.sqrt(np.array([c for m, c in y0_mean_covar])))
    
    if y0.shape[1] == 1:
        # 1D data
        y0 = y0[:, 0]
        ax.plot(x0, y0, color=color)
        ax.fill_between(x0, y0-nsigma*var0, y0+nsigma*var0, alpha=0.3)
        ax.plot(llr.x, llr.y[:, 0], ".", markersize=1, alpha=0.4, color=color)
    else:
        # 2D data
        plot_mean_var(y0, nsigma*var0, ax=ax)
        ax.plot(llr.y[:, 0], llr.y[:, 1], ".", markersize=1, alpha=0.4, color=color)
        
    



if __name__ == "__main__":
    if False:
        x = [[-1, 0, ], [0, 0], [1, 1], [1, 2], [0, 3]]
        c = [0.1, 0.1, 0.2, 0.3, 0.4]
        plot_mean_var(x, c)
        plt.show()

    if True:
        x = np.linspace(0, 2*np.pi, 100)
        ym = np.sin(x)
        xy = [(x, ym + np.random.normal(size=ym.size)) for i in range(20)]
        plot_average(xy, tau=0.6, nsigma=1)
        plt.show()
        
    if True:
        # 2D circle
        x = np.linspace(0, 2 * np.pi, 100)
        ym = 10 * np.vstack([np.sin(x), np.cos(x)]).T
        xy = [(x, ym + np.reshape(np.random.normal(size=ym.size), newshape=ym.shape)) for i in range(20)]
        plot_average(xy, tau=0.2, nsigma=1)
        plt.show()
    