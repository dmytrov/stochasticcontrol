import numpy as np
import matplotlib.pyplot as plt
import analysis.delayedfeedback.fittingutils as fu



def InverseOptimalControl(object):
    def __init__(self):
        pass


def infer_cost(trajectories):
    """ Infers the goal cost of the optimal controller.
        
        Assumptions:
            - cost is linear w.r.t. some terms:
                - integral over time (total time)
                - integral over energy used for the control (F*S).
                  Force is proportional to acceleration, mass is constant:
                    F = m*a
                - integral over jerk (path third derivative)
            - control u is force;
            - control is optimal.

        Optimal control minimizes the functional of cost.
        We assume the control is optimal.
        Find the weights of the cost terms which minimize the cost functional
        for all trajectories plus some noise.
    """
    
    # Construct the cost terms
    pass


def fit_trajectory_nonesense(trials, maxpoly=5, maxexp=2, ax=None):
    """ WARNING!!! NONESENSE!!!
        
        Fit polynomial and exponentials as a solution to an optimal 
        reaching problem.

        Fit a function f: t -> x
        
        Assumptions:
            - time is approximately the same. Motion is from time 0 to 1.
            - total cost is a sum of quadratic costs 
                of derivatives of different orders up to maxorder.
            - trajectories are optimal solutions + noise.
        
        Arguments:
            - start point
            - end point
            - trials
        Returns:
            - callable fitted function x(t)
    """
    
    # Normalize time to [0, 1]
    traces = [trial.motiontrajectoryinterpolated[trial.a:trial.b] for trial in trials]
    times = np.hstack([np.linspace(0, 1, len(trace)) for trace in traces])

    # For diagonal quadratic cost, coordinates are independent.
    xs = np.hstack([trace[:, 0] for trace in traces])
    ys = np.hstack([trace[:, 1] for trace in traces])
    
    a, b, c, ystar = fu.fit_polynomial_exponential(times, xs, maxpoly, maxexp)
    print("X:", a, b, c)
    fx = fu.PolynomialExponential(a, b, c)

    a, b, c, ystar = fu.fit_polynomial_exponential(times, ys, maxpoly, maxexp)
    print("Y:", a, b, c)
    fy = fu.PolynomialExponential(a, b, c)

    for trace in traces:
        ax.plot(trace[:, 0], trace[:, 1], "b.", alpha=0.3)

    t = np.linspace(0, 1, 100)
    ax.plot(fx(t), fy(t), "r")
    
    
