""" This file demonstrates the ability of a simple policy, which is a simple 
linear combimation of features, to mimic different control strategies:
 - minimum jerk trajectory.
Method:
 - create trining data,
 - optimize the policy,
 - generate a trajectory by running the policy.

  Author: Dmytro Velychko, Philipps University of Marburg
  velychko@staff.uni-marburg.de
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mcvariationalbayes.mcvb as mcvb
import mcvariationalbayes.torchjson as torchjson
import model as tsm
from minjerkutils import *


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
    return (add_noise(x, scale=0.002), x[0], ptend)


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
    data = []

    # Straight
    for i in range(10):
        x = straight_trajectory(t)
        data.append(trajectory_to_data(x))
        plt.plot(x[:, 0], x[:, 1], "-o", alpha=0.1)

    # Bent
    for i in range(20):
        xb = bent_trajectory(t)
        data.append(trajectory_to_data(xb))
        plt.plot(xb[:, 0], xb[:, 1], "-o", alpha=0.3)

    # Save the training data
    plt.axis("equal")
    plt.savefig(os.path.join(savepath, "trainig.pdf"))
    plt.close()

    #with open("reaching(H).json", "r") as f:
    #    data = torchjson.load(f) 
    
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
    lpm = tsm.LinearPolicyModel(data, regressortypes=regressortypes)
    
    lpm.to_device(device, dtype)
    vb.to_device(device, dtype)
    
    # Infer the model parameters
    maxiter = 1500
    nparticles = 10
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
        model_sampler, proposal_sampler = vb.infer(lpm.model, lpm.proposal,
                nparticles=nparticles, lr=0.01, maxiter=maxiter, callback=pl.iter_callback)
        initvals.update(model_sampler._param)
        initvals.update(proposal_sampler._param)
        with open(paramsfilename, "w") as f:
            initvals.json_dump(f, indent=2)
        print("Parameters saved successfully!")

    device = torch.device("cpu")
    vb.to_device(device, dtype)
    initvals.to_device(device=device, dtype=dtype)
    
    policy = initvals["policy.loc"].to("cpu", torch.float64).data.numpy()
    tsm.pprint_policy(policy, regressortypes)    

    # Sample trajectories
    print("Sampling trajectories...")
    len_x = lpm["x"].shape[1]
    ignore_obs = set(["x_{}".format(k) for k in range(3, len_x)])
    model_sample, proposal_sample = vb.sample(lpm.model, lpm.proposal,
                initvals=initvals, ignore_obs=ignore_obs)
    xstars = torch.stack([model_sample.value("x_{}".format(k)) for k in range(0, len_x)], dim=-2)
    xstars = xstars.to("cpu", torch.float64)
    for i, data_i in enumerate(lpm.data):
        print("Saving trajectory", i)
        x = data_i[0]
        xstar = xstars[i].data.numpy()
        plt.plot(x[:, 0], x[:, 1], "-o", alpha=0.2)
        plt.plot(xstar[:, 0], xstar[:, 1], "-o")
        plt.axis("equal")
        plt.savefig(savepath + "min-jerk-sampled-({}).pdf".format(i))
        plt.close()
        
   