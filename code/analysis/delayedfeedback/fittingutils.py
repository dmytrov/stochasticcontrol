import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
import scipy.optimize as opt



def data_to_vector(args):
    return np.hstack([arg.data.numpy().flatten() for arg in args])


def grad_to_vector(args):
    return np.hstack([arg.grad.numpy().flatten() for arg in args])


def from_vector(args, v):
    for arg in args:
        s = arg.data.numpy().size
        arg.data.copy_(torch.Tensor(np.reshape(v[:s], newshape=arg.shape)))
        v = v[s:]
        if arg.grad is not None:
            arg.grad.data.zero_()


def optimize_l_bfgs_scipy(f, params, maxiter=1000, callback=None):
    params = list(params)
    
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def f_df_params(v, params):
        from_vector(params, v)
        zero_grad(params)
        loss = f()
        loss.backward()
        return loss.data.numpy(), grad_to_vector(params)
    
    f_df = lambda v: f_df_params(v, params)
    xopt, fopt, d = opt.fmin_l_bfgs_b(f_df, x0=data_to_vector(params), disp=1, maxiter=maxiter, 
            maxls=100, factr=10, pgtol=1.0e-8, callback=callback)
    from_vector(params, xopt)
    zero_grad(params)
    return xopt, fopt, d


def optimize_l_bfgs_torch(f, params, maxiter=100):
    optimizer = torch.optim.LBFGS(params)
    for i in range(maxiter):
        def closure():
            optimizer.zero_grad()
            loss = f()
            print("{}, Loss: {}".format(i, loss.data.numpy()))
            loss.backward()
            return loss
        optimizer.step(closure)


class Polynomial(object):
    """ Polynomial function
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs
        
    def __call__(self, x):
        return np.sum([a *np.power(x, i) for i, a in zip(range(len(self.coeffs)), self.coeffs)], axis=0)


class PolynomialExponential(object):
    """ Polynomial + exponential function
    """
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def __call__(self, x):
        res = np.sum([a *np.power(x, i) for i, a in zip(range(len(self.a)), self.a)], axis=0)
        if len(self.b) > 0:
            res += np.sum([b *np.power(x, i) * np.exp(c * x) for i, b, c in zip(range(len(self.b)), self.b, self.c)], axis=0)
        return res

        

def fit_ploynomial(x, y, order):
    """ MSE fit of data with a polynonial f(x):x->y of some order.
    
    Parameters:
        x : [N] - vector of inputs
        y : [N] - vector of outputs
        order : scalar - polynomial order
    
    Returns:
        (coeffs, residuals, MSE)
        res : [order] - list of polynomial coefficients
        ystar : [N] - vector of fitted values
    """
    # Construct the matrix of bases
    assert len(x) == len(y)
    N = len(x)
    B = np.vstack([np.power(x, i) for i in range(order+1)]).T
    
    # MSE fit
    # B*A = Y + noise
    # A = pinv(B) * Y
    A = np.linalg.pinv(B).dot(y)
    ystar = B.dot(A)
    
    return A, ystar



def fit_polynomial_exponential(t, x, maxpoly=1, maxexp=0):
    """ MSE fit of data with a polynonial + exponentials f(x):x->y of some order.
    
    Parameters:
        t : [N] - vector of inputs, time
        x : [N] - vector of outputs, state
        maxpoly : scalar - max oplynomial $a_i t^i$ order
        maxexp : scalar - max exponential $b_i t^i e^{c_i t}$ order
    
    Returns:
        (coeffs, residuals, MSE)
        respoly : [order] - list of polynomial coefficients a
        resexp : ([order], [order]) - two lists of polynomial coefficients b and c
        xstar : [N] - vector of fitted values
    """

    N = len(t)
    
    # Construct tencors of polynomial coefficients
    astart, _ = fit_ploynomial(t, x, maxpoly)
    a = torch.tensor(astart, dtype=torch.double)
    b = torch.tensor(np.zeros([maxexp + 1]), dtype=torch.double)
    c = torch.tensor(np.zeros([maxexp + 1]), dtype=torch.double)
    a.requires_grad = True
    b.requires_grad = True
    c.requires_grad = True

    def xstar(t):
        polybasis = torch.tensor(np.vstack([np.power(t, i) for i in range(maxpoly+1)]).T)  # t^i
        xs = torch.mv(polybasis, a)
        
        if maxexp > 0:
            expbasis = torch.stack([torch.exp(c[i] * torch.tensor(t)) * torch.tensor(np.power(t, i)) for i in range(1, maxexp+1)], -1)
            exponential = torch.mv(expbasis, b[1:])
            xs += exponential
        return xs 

    def floss(t):
        loss = torch.sum((xstar(t) - torch.tensor(x))**2)
        return loss
    
    optimize_l_bfgs_torch(floss, params)
    optimize_l_bfgs_scipy(floss, params)
    return data_to_vector([a]), data_to_vector([b]), data_to_vector([c]), xstar(t).data.numpy()


if __name__ == "__main__":
    np.random.seed(555)
    x = np.linspace(0, 2*np.pi, 100)
    ym = np.sin(x)
    y = ym + 0.1*np.random.normal(size=ym.size)
    
    if False:
        A, ystar = fit_ploynomial(x, y, order=10)
        residuals = y-ystar
        MSE = np.sum(residuals**2)
        f = Polynomial(A)

        plt.plot(x, y, ".")
        plt.plot(x, ystar)
        plt.plot(x, f(x), "r--")
        plt.show()
    
    if True:
        a, b, c, ystar = fit_polynomial_exponential(x, y, maxpoly=5, maxexp=3)
        print(a, b, c)
        plt.plot(x, y, ".")
        plt.plot(x, ystar)
        f = PolynomialExponential(a, b, c)
        plt.plot(x, f(x), "r--")
        plt.show()
        
        

