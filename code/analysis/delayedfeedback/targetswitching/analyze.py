import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import states.utils as su
import analysis.delayedfeedback.regression as re
import analysis.delayedfeedback.patherrors as pe
import analysis.delayedfeedback.trial as tr
import analysis.delayedfeedback.datalayer as dl
import analysis.delayedfeedback.database as db
import analysis.delayedfeedback.averagetrajectory as at
import analysis.delayedfeedback.fittingutils as fu
import analysis.delayedfeedback.optimalcontrolutils as ocu
import analysis.delayedfeedback.targetswitching.model as tsm
import analysis.delayedfeedback.targetswitching.plots as tsp
import analysis.variationalbayes.mcvb as mcvb



regressortypes = [ \
        tsm.RegressorType.Velocity1D, 
        tsm.RegressorType.Acceleration, 
        tsm.RegressorType.OptimalTarget, 
        tsm.RegressorType.OptimalTrajectory,
        tsm.RegressorType.Const,
        ]

def load_trials(dbfilter):
    
    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    # Select baseline trials
    dbtrials = dbfilter(di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject)) \
            .all()
    trials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    validtrials = [t for t in trials if t.isvalid]
    n = len(trials)
    nvalid = len(validtrials)
    print("Valid trials: {} of {}, {}%".format(nvalid, n, 100*nvalid/n))
    return validtrials


def trials_to_unified_data(trials):
    res = []
    for trial in trials:
        res.append([trial.motiontrajectoryinterpolated[trial.a:trial.b], 
                    trial.ptstart, trial.pttarget,
                    trial.ptstartcorrected, trial.pttargetcorrected])
    return res


class SamplerInterface(object):
    def __init__(self, env):
        self.env = env

    def param(self, name, initval):
        return self.env.param(name, initval)

    def sample(self, name, distr, obs=None):
        return self.env.sample(name, distr, obs)

    def observe(self, name, obs):
        return self.env.observe(name, obs)


class MaskedTargetSwitchingModel(tsm.TargetSwitchingModel):

    def model_only_straight(self, env):

        class MaskDisturbedSampler(SamplerInterface):
            def __init__(self, modelobj, env):
                super(MaskDisturbedSampler, self).__init__(env)
                self.modelobj = modelobj

            def sample(self, name, distr, obs=None):
                if name == "xtp1_control":
                    disturbed = self.modelobj["disturbed"]
                    straight = 1 - disturbed
                    distr = distr.mask(straight[:, None, None])
                return self.env.sample(name, distr, obs)

        self.model_fast(MaskDisturbedSampler(self, env))

    def proposal_only_straight(self, env):

        class MaskDisturbedSampler(SamplerInterface):
            def __init__(self, modelobj, env):
                super(MaskDisturbedSampler, self).__init__(env)
                self.modelobj = modelobj

            def param(self, name, initval):
                if name.startswith("sigm_") or name.startswith("log_sigm_"):
                    return self.env.const(name, initval)
                else:
                    return self.env.param(name, initval)
                    

        self.proposal(MaskDisturbedSampler(self, env))

    def model_only_switching_params(self, env):

        class MaskAllParamsSampler(SamplerInterface):
            def __init__(self, modelobj, env):
                super(MaskAllParamsSampler, self).__init__(env)
                self.modelobj = modelobj

            def param(self, name, initval):
                return self.env.observe(name, initval)

        self.model_fast(MaskAllParamsSampler(self, env))

    def proposal_only_switching_params(self, env):

        class OnlySwitchingParamsSampler(SamplerInterface):
            def __init__(self, modelobj, env):
                super(OnlySwitchingParamsSampler, self).__init__(env)
                self.modelobj = modelobj

            def param(self, name, initval):
                if name.startswith("sigm_"):
                    return self.env.param(name, initval)
                else:
                    return self.env.const(name, initval)

        self.proposal(OnlySwitchingParamsSampler(self, env))


            
    

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

    # Manual seed
    np.random.seed(0)
    torch.manual_seed(0)

    dirpath = "./subjects/"
    
    #subjects = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    subjects = ["H"]

    for subject in subjects:
        subjpath = os.path.join(dirpath, subject)
        paramsfilename = os.path.join(subjpath, "params.json")
        if not os.path.exists(subjpath):
            os.makedirs(subjpath)
        
        datafilter = lambda query: query \
                .filter(db.Subject.name == subject) \
                .filter(db.Trial.disturbance_mode == su.DisturbanceMode.NoDisturbance or \
                    db.Trial.disturbance_mode == su.DisturbanceMode.Translation) \
                .filter(db.Trial.feedback_delay == -0.0)
        trials = load_trials(datafilter)
        straighttrials = [trial for trial in trials if not trial.disturbed]
        disturbedtrials = [trial for trial in trials if trial.disturbed]
        datacombined = trials_to_unified_data(trials)
        datastraight = trials_to_unified_data(straighttrials)
        datadisturbed = trials_to_unified_data(disturbedtrials)
        print("Straight trials:", len(datastraight))
        print("Disturbed trials:", len(datadisturbed))
        print("Total trials:", len(datacombined))
        
        data = datastraight + datadisturbed
        initvals = mcvb.TensorContainer()
        targetswitching = MaskedTargetSwitchingModel(data, regressortypes=regressortypes, 
                memsize=1, delaysize=0,
                policyactivation=False)
        targetswitching.to_device(device, dtype)
        vb = mcvb.VB()
        vb.to_device(device, dtype)

        if not args.recompute and os.path.exists(paramsfilename):
            with open(paramsfilename, "r") as f:
                initvals.json_load(f)
                initvals.to_device(device, dtype)
            print("Parameters loaded successfully!")
        else:                
            pl = tsm.Plotter()
            nparticles = 100
            # MCVB learning
            for maxiter, lr, model, prop in [
                    # Only straight trials
                    #(10, 0.01, targetswitching.model_only_straight, targetswitching.proposal_only_straight), 
                    #(100, 0.01, targetswitching.model_only_straight, targetswitching.proposal_only_straight), 
                    #(100, 0.001, targetswitching.model_only_straight, targetswitching.proposal_only_straight),
                    #(1000, 0.01, targetswitching.model_only_straight, targetswitching.proposal_only_straight), 
                    #(1000, 0.001, targetswitching.model_only_straight, targetswitching.proposal_only_straight),
                    
                    # Only sigmoid
                    #(10, 0.01, targetswitching.model_only_switching_params, targetswitching.proposal_only_switching_params), 
                    #(100, 0.01, targetswitching.model_only_switching_params, targetswitching.proposal_only_switching_params), 
                    #(100, 0.0001, targetswitching.model_only_switching_params, targetswitching.proposal_only_switching_params)
                    #(1000, 0.01, targetswitching.model_only_switching_params, targetswitching.proposal_only_switching_params), 
                    #(1000, 0.001, targetswitching.model_only_switching_params, targetswitching.proposal_only_switching_params),
                    
                    # Full model
                    #(10, 0.01, targetswitching.model_fast, targetswitching.proposal), 
                    #(100, 0.01, targetswitching.model_fast, targetswitching.proposal), 
                    #(100, 0.001, targetswitching.model_fast, targetswitching.proposal),
                    #(100, 0.1, targetswitching.model_fast, targetswitching.proposal), 
                    (3000, 0.01, targetswitching.model_fast, targetswitching.proposal), 
                    (1000, 0.001, targetswitching.model_fast, targetswitching.proposal)
                    ]:
                model_sampler, proposal_sampler = vb.infer(
                        model, prop,
                        initvals=initvals,
                        nparticles=nparticles, lr=lr, maxiter=maxiter, callback=pl.iter_callback)
                
                initvals.update(model_sampler._param)
                initvals.update(proposal_sampler._param)

                policy = initvals["policy.loc"].to("cpu").data.numpy()
                tsm.pprint_policy(policy, regressortypes)
           
            with open(paramsfilename, "w") as f:
                initvals.json_dump(f, indent=2)
            print("Parameters saved successfully!")

            # MAP optimization
            optimize_MAP = False
            if optimize_MAP:
                maxiter = 5000
                model_sampler, proposal_sampler, log_prob = vb.optimize(targetswitching.model_fast,
                        initvals=initvals,
                        maxiter=maxiter, callback=pl.iter_callback)
                initvals.update(model_sampler._param)
                initvals.update(proposal_sampler._param)

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
                    c=sigmoid[i][:len(trajectory)], 
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
            
            plt.savefig(os.path.join(subjpath, "switching-fitted-({}).pdf".format(i)))
            plt.close()