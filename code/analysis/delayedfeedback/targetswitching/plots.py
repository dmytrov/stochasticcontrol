""" Plotting utlities for the switching model
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import analysis.delayedfeedback.targetswitching.diagnostics as diag

def plot_training(axes, data):
    """ X-Y plot with 
    """
    for i, trajectory, start, target, _, target_dist in enumerate(data):
        axes.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
        axes.plot(target[0], target[1], "xb")
        axes.plot(target_dist[0], target_dist[1], "or")    
        axes.plot(start[0], start[1], "*g")


def plot_trajectory(axes, data, sigmoid=None):
    """ X-Y plot, sigmoid-colored
    """
    trajectory, _, target, _, target_dist = data
    axes.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.1)
    axes.scatter(trajectory[:, 0], trajectory[:, 1], 
            marker="o", color=sigmoid, alpha=0.3, cmap=plt.cm.coolwarm)
    axes.plot(target[0], target[1], "xb")
    axes.plot(target_dist[0], target_dist[1], "or")
    axes.plot(start[0], start[1], "*g")


def plot_sampled(axes, sampled=None):
    if sampled is not None:
        axes.plot(sampled[:, 0], sampled[:, 1], "-o", alpha=0.3)


def fig_plot_trajectory(data, sigmoid=None, sampled=None, filename=None):
    """ Create a new figure, plot, save or show
    """
    plt.figure()
    axes = plt.gca()
    plot_trajectory(axes, data, sigmoid)
    plot_sampled(axes, sampled=sampled)
    axes.axis("equal")
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_sigmoid_probmap(axes, model, initvals, i):
    sigma_as = np.linspace(0.01, 10.0, 100)
    x_is = np.linspace(0.0, 1.0, 50)
    logprobmap = diag.sigmoid_logprob_profile(model, initvals, i, sigma_as=sigma_as, x_is=x_is)
    logprobmap = logprobmap - np.max(logprobmap)
    axes.axis([0.0, 1.0, 1.0, 10.0])
    axes.imshow(np.exp(logprobmap), extent=[0.0, 1.0, 1.0, 10.0], origin="lower", aspect="auto")
    axes.set_xlabel("inflection")
    axes.set_ylabel("a")   


def plot_sigmoid_posterior_probmap(axes, proposal, initvals, i):
    sigma_as = np.linspace(0.01, 10.0, 100)
    x_is = np.linspace(0.0, 1.0, 50)
    logprobmap = diag.sigmoid_posterior_density(proposal, initvals, i, sigma_as=sigma_as, x_is=x_is)
    logprobmap = logprobmap - np.max(logprobmap)
    axes.axis([0.0, 1.0, 0.0, 10.0])
    axes.imshow(np.exp(logprobmap), extent=[0.0, 1.0, 0.0, 10.0], origin="lower", aspect="auto")
    axes.set_xlabel("inflection")
    axes.set_ylabel("a") 


if __name__ == "__main__":
    T = 20
    tr = np.vstack([np.sort(np.random.uniform(0, 20, T)), np.linspace(0, 10, T)]).T
    trial = (tr[:], tr[0] , tr[-1], tr[0] , tr[-1])
    data = [trial]
    fig_plot_trajectory(data[0])