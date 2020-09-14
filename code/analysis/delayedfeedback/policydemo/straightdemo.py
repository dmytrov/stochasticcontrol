""" This file demonstrates the ability of a simple policy, which is a simple 
linear combimation of features, to mimic different control strategies:
 - optimal feedback control after a disturbance,
 - trajectory control after a disturbance,
 - minimum jerk trajectory.
Method:
 - create trining data,
 - optimize the policy,
 - generate a trajectory by running the policy.
"""
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import analysis.variationalbayes.mcvb as mcvb
import analysis.delayedfeedback.targetswitching.model as tsm
import analysis.delayedfeedback.targetswitching.plots as tsp
from analysis.delayedfeedback.policydemo.minjerkutils import *


def straight_trajectory(t, ptstart=(0, 0), ptend=(0, 1)):
    # Generate straight training trajectory
    x, xp, xpp = gen_trajectory_min_jerk(t, x0=ptstart, x1=ptend)
    return x

def bent_trajectory(t, ptstart=(0, 0), ptstartpp=None, ptend=None):
    # Generate randomly bent trajectory
    if ptstartpp is None:
        ptstartpp = [0, 0.1] + [20, 0.1] * np.random.uniform(-1.0, 1.0, 2)
    if ptend is None:
        ptend = [0.0, 1.0] + [0.5, 0.2] * np.random.uniform(-1.0, 1.0, 2) 
    x, xp, xpp = gen_trajectory_min_jerk(t, x0=ptstart, x0pp=ptstartpp, x1=ptend)
    return x


def disturbed_trajectory(t, ptstart=(0, 0), ptend=(0, 1), x_target_disturbed=None):
    # Generate randomly disturbed trajectory
    if x_target_disturbed is None:
        x_target_disturbed = [0, 1] + [0.5, 0.2] * np.random.uniform(-1.0, 1.0, 2) 
    x, xp, xpp = gen_trajectory_min_jerk_disturbed(t, x_target_disturbed=x_target_disturbed)
    return x


def trajectory_to_data(x, ptend=None):
    # Data is noise-corrupted trajectory
    if ptend is None:
        ptend = x[-1]
    return (add_noise(x, scale=0.002), x[0], ptend, x[0] , x[-1])


if __name__ == "__main__":
    # NAN anomaly detection
    from torch import autograd
    #torch.set_anomaly_enabled(True)

    ## Comman line parameters
    
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Min-jerk control policy learning demo")
    parser.add_argument("-c", dest="cuda", action="store_true", help="Use CUDA if possible. Default is CPU")
    parser.add_argument("-d", dest="double", action="store_true", help="Use float64 type. Default is float32")
    parser.add_argument("-r", dest="recompute", action="store_true", help="Force recompute. Default is reload")
    args = parser.parse_args()

    devicename = "cuda" if args.cuda else "cpu"
    device = torch.device(devicename)
    dtype = torch.float64 if args.double else torch.float32
    torch.set_default_dtype(dtype)
    print("Using", device, dtype)

    # Namual seed
    np.random.seed(0)
    torch.manual_seed(0)

    ### Min-jerk trajectory learning
    
    savepath = "./out/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    ## Generate training data
    t = np.arange(0.0, 1.0, 0.05)
    
    # Straight
    data_straight = []
    for i in range(20):
        x = straight_trajectory(t)
        data_straight.append(trajectory_to_data(x))
        plt.plot(x[:, 0], x[:, 1], "-o", alpha=0.1)

    # Bent
    for i in range(0):
        xb = bent_trajectory(t)
        data_straight.append(trajectory_to_data(xb))
        plt.plot(xb[:, 0], xb[:, 1], "-o", alpha=0.3)

    # Save the training data
    plt.axis("equal")
    plt.savefig(os.path.join(savepath, "trainig.pdf"))
    plt.close()
    
    # Disturbed
    data_disturbed = []
    for i in range(5):
        xd = disturbed_trajectory(t)
        data_disturbed.append(trajectory_to_data(xd, ptend=(0, 1)))
        plt.plot(xd[:, 0], xd[:, 1], "-o", alpha=0.3)
    plt.axis("equal")
    plt.savefig(os.path.join(savepath, "disturbed.pdf"))
    plt.close()
    
    data = data_straight + data_disturbed

    # Create and configure the main VB and model objects
    vb = mcvb.VB()
    pl = tsm.Plotter()
    regressortypes = [ \
            tsm.RegressorType.Velocity1D, 
            tsm.RegressorType.Acceleration, 
            tsm.RegressorType.OptimalTarget, 
            tsm.RegressorType.OptimalTrajectory,
            tsm.RegressorType.Const,
            ]
    targetswitching = tsm.TargetSwitchingModel(data=data, regressortypes=regressortypes, 
            memsize=1, delaysize=0, policyactivation=False)
    
    targetswitching.to_device(device, dtype)
    vb.to_device(device, dtype)
    
    # Infer the model parameters
    maxiter = 3000
    nparticles = 300
    initvals = mcvb.TensorContainer()
    paramsfilename = os.path.join(savepath, "params.json")
        
    try:
        if args.recompute:
            raise  # go to recompute
        with open(paramsfilename, "r") as f:
            initvals.json_load(f)
            initvals.to_device(device, dtype)
        print("Parameters loaded successfully!")
    except: 
        model_sampler, proposal_sampler = vb.infer(targetswitching.model_fast, targetswitching.proposal,
                nparticles=nparticles, lr=0.01, maxiter=maxiter, callback=pl.iter_callback)
        initvals.update(model_sampler._param)
        initvals.update(proposal_sampler._param)
        with open(paramsfilename, "w") as f:
            initvals.json_dump(f, indent=2)
        print("Parameters saved successfully!")

    # Move to CPU
    device = torch.device("cpu")
    vb.to_device(device, dtype)
    initvals.to_device(device, dtype)
    targetswitching.to_device(device, dtype)
    
    # Run deterministic policy
    policy = initvals["policy.loc"].to("cpu", torch.float64).data.numpy()
    tsm.pprint_policy(policy, regressortypes)
    xstar = tsm.run_policy(policy, x[:3], x[-1], n=len(x), regressortypes=regressortypes)
        
    x = data[0][0]
    plt.plot(t, x[:, 1], "-o", alpha=0.7)
    plt.plot(t, xstar[:, 1], "-o", alpha=1.0)
    plt.xlabel("time")
    plt.title("Learned min-jerk trajectory")
    plt.ylabel("position")
    plt.savefig(savepath + "min-jerk-mcvb-position.pdf")
    plt.close()

    dx = np.diff(x, axis=0)
    dxstar = np.diff(xstar, axis=0)
    plt.plot(t[:-1], dx[:, 1], "-o", alpha=0.7)
    plt.plot(t[:-1], dxstar[:, 1], "-o", alpha=1.0)
    plt.xlabel("time")
    plt.title("Learned min-jerk trajectory")
    plt.ylabel("velocity")
    plt.savefig(savepath + "min-jerk-mcvb-velocity.pdf")
    plt.close()

    ddx = np.diff(dx, axis=0)
    ddxstar = np.diff(dxstar, axis=0)
    plt.plot(t[1:-1], ddx[:, 1], "-o", alpha=0.7)
    plt.plot(t[1:-1], ddxstar[:, 1], "-o", alpha=1.0)
    plt.xlabel("time")
    plt.title("Learned min-jerk acceleration")
    plt.ylabel("acceleration")
    plt.savefig(savepath + "min-jerk-mcvb-acceleration.pdf")
    plt.close()
    
    
    ## Sample trajectories from the model
    print("Sampling trajectories...")
    len_x = targetswitching["x"].shape[1]
    ignore_obs = set(["x_{}".format(k) for k in range(3, len_x)])
    model_sample, proposal_sample = vb.sample(targetswitching.model, targetswitching.proposal,
                initvals=initvals, ignore_obs=ignore_obs)
    xstars = torch.stack([model_sample.value("x_{}".format(k)) for k in range(0, len_x)], dim=-2)
    xstars = xstars.to("cpu", torch.float64).data.numpy()
    sigmoid = torch.stack([model_sample.get_stored("sigmoid_{}".format(k))[0] for k in range(3, len_x)], dim=-1)
    for i in range(3):
        sigmoid = torch.cat([np.NaN * sigmoid[:, 0][:, None], sigmoid], dim=-1)
    sigmoid = sigmoid.data.numpy()
    is_disturbed = targetswitching["disturbed"].data.numpy()

    ## Plot likelihood w.r.t sigmoid parameters 
    for i, data_i in enumerate(data):
        if not is_disturbed[i]:
            continue
        
        print("Plotting trajectory", i)
        trajectory, start, target, _, target_dist = data_i
        
        T, D = trajectory.shape
        t = torch.arange(0, T, dtype=dtype).to("cpu")
        
        psize = 5
        fig = plt.figure(constrained_layout=True, figsize=(2*psize, 2*psize)) 
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        
        fax = fig.add_subplot(gs[0, 0])
        fax.scatter(trajectory[:, 0], trajectory[:, 1], 
                marker="o", 
                c=sigmoid[i], 
                alpha=0.8, cmap=plt.cm.coolwarm)
        fax.plot(target[0], target[1], "xb")
        fax.plot(target_dist[0], target_dist[1], "or")
        xstar = xstars[i]
        fax.plot(xstar[:, 0], xstar[:, 1], "-o", color="darkorange")
        fax.set_xlabel("x")
        fax.set_ylabel("y")
        fax.set_title("Training vs model sampled data")
        plt.axis("equal")

        fax = fig.add_subplot(gs[0, 1])
        fax.plot(trajectory[:, 0])
        fax.plot(trajectory[:, 1])
        #fax.plot(sigmoid[i])
        fax.set_xlabel("time")
        fax.set_ylabel("coordinate")
        fax.set_title("Training vs model sampled data")
        
        fax = fig.add_subplot(gs[1, 0])
        tsp.plot_sigmoid_probmap(fax, targetswitching.model_fast, initvals, i)
        fax.set_title("Model sigmoid logprob")
        
        fax = fig.add_subplot(gs[1, 1])
        tsp.plot_sigmoid_posterior_probmap(fax, targetswitching.proposal, initvals, i)
        fax.set_title("Posterior")
        
        plt.savefig(os.path.join(savepath, "switching-fitted-({}).pdf".format(i)))
        plt.close()
       
  