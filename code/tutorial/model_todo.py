"""

  Author: Dmytro Velychko, Philipps University of Marburg
  velychko@staff.uni-marburg.de
"""

import numpy as np
import torch
import visdom
import mcvariationalbayes.mcvb as mcvb



def vec_pts_to_line(pts, ptline, vline):
    """ Normal vector from a point to a line.

        Arguments:
            - pts - [N, D] - N D-dimensional points
            - ptline - [D] - point on a line
            - vline - [D] - vector of the line, not necessarily normalized
        
        Returns:
            - [N, D] - shortest vectors from the N points to the line (unnormalized)
    """
    a = np.dot(vline, (pts - ptline).T) / vline.dot(vline)
    res = a[:, None] * vline + ptline - pts
    return res

def rot_90(x):
    if x.dim() == 1:
        return x[[1, 0]] * torch.tensor([-1.0, 1.0]).to(x)
    elif x.dim() > 1:
        return x[..., [1, 0]] * torch.tensor([-1.0, 1.0]).to(x)


def vec_pts_to_line_torch(pts, ptlines, vlines):
    """ Normal vector from a point to a line.

        Arguments:
            - pts - [N, D] - N D-dimensional points
            - ptlines - [N, D] - points on the lines
            - vlines - [N, D] - vectors of the lines, not necessarily normalized
        
        Returns:
            - [N, D] - shortest vectors from the N points to the lines (unnormalized)
    """
    a = torch.sum(vlines * (pts - ptlines), dim=-1) / torch.sum(vlines * vlines, dim=-1)
    res = a[:, None] * vlines + ptlines - pts
    return res


def vec_pts_to_lines_torch(pts, ptlines, vlines):
    """ Normal vector from N points to N lines, elementwise.

        Arguments:
            - pts - [..., D] - many D-dimensional points
            - ptlines - [..., D] - points on many lines
            - vlines - [..., D] - vectors of many lines, not necessarily normalized
        
        Returns:
            - [N, D] - shortest vectors from the N points to the lines (unnormalized)
    """
    a = torch.sum(vlines * (pts - ptlines), dim=-1) / torch.sum(vlines * vlines, dim=-1)
    res = a[..., None] * vlines + ptlines - pts
    return res


def stable_normalize(x, etha=1.0e-8):
    """ Numerically stable vector normalization
    """
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if n < etha:
        n = etha
    return x / n


def stable_normalize_torch(x, etha=1.0e-8):
    """ Numerically stable vector normalization
    """
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=etha)


def stable_sigmoid(x):
    """ Numerically stable sigmoid function of one variable
    """
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = exp(x)
        return z / (1 + z)


class RegressorType(object):
    Const = 0
    Position = 1
    Velocity1D = 2
    Acceleration = 3
    Control = 4
    OptimalTarget = 5
    OptimalTrajectory = 6
    count = 7

    names = ("Const",
            "Position",
            "Velocity1D",
            "Acceleration",
            "Control",
            "OptimalTarget",
            "OptimalTrajectory",)
    
    statesize = (1, 2, 1, 2, 2, 2, 2)


def pprint_policy(policy, regressortypes):
    print("Policy:")
    i = 0
    for r in regressortypes:
        s = RegressorType.statesize[r]
        p = policy[:, i:i+s]
        i += s
        print("{}: max(abs)={} \n{}".format(RegressorType.names[r], np.max(np.abs(p)), p))
        

def signal_dependent_noise_covar_torch(control, scale=torch.tensor([2.0, 1.0]), uniform=0.0, etha=1.0e-6):
    """ Signal-dependent noise covariance matrix.
        control: [2] - control vector
        scale: [2] - covariance scale in the [control, normal] directions
        uniform: scalar - additional diagonal noise
        etha: scalar - diagonal boost
    """
    control_n = rot_90(control)
    m_control_global = torch.stack([control, control_n], dim=-1)
    m_control_globalscaled = m_control_global * scale[None, :]    
    vuvt = torch.einsum("...ik,...jk->...ij", m_control_globalscaled, m_control_globalscaled)
    m_covar = vuvt + torch.eye(2).to(vuvt) * (uniform + etha)
    return m_covar, m_control_globalscaled


def signal_dependent_noise_covar_xaligned_torch(control, scale=torch.tensor([2.0, 1.0]), uniform=0.0, etha=1.0e-6):
    """ Signal-dependent uncorrelated noise covariances for x-aligned control vector.
        control: [2] - x-aligned control vector; control[-1] == 0
        scale: [2] - covariance scale in the [control, normal] directions
        uniform: scalar - additional diagonal noise
        etha: scalar - diagonal boost
    """
    control_n = rot_90(control)
    m_control_global = torch.stack([control, control_n], dim=-1)
    m_control_globalscaled = m_control_global * scale[None, :]
    diag_covar = (torch.norm(control, dim=-1, keepdim=True) * scale)**2 + (uniform + etha)
    return diag_covar, m_control_globalscaled


class LinearPolicyModel(mcvb.TensorContainer):
    def __init__(self, 
            data,
            regressortypes=None):
        """
            data : list of tuples (trajectory, ptstart, ptend)
        """
        super(LinearPolicyModel, self).__init__()
        self.data = data
        self.regressortypes = regressortypes
        
        statesize = 0
        for r in regressortypes:
            statesize += RegressorType.statesize[r] 
        self.initialpolicy = np.zeros([2, statesize])

        # Construct extended data tensors with maxlength
        # to unify the model and a binary mask for the missing data
        N = len(self.data)  # number of trials
        lengths = np.array([len(x) for x, _, _ in self.data])  # [N]
        self.lengths = lengths
        maxlength = np.max(lengths)  # max trial length
        x = [np.concatenate([tr, 0.0 * np.zeros([maxlength-len(tr), 2])], axis=0) for tr, _, _, in self.data]   # [N, Xmaxlength, D]
        self.x = np.stack(x, axis=0)  # [N, T, D]
        m = [np.concatenate([np.ones(length), np.zeros(maxlength-length)])  for length in lengths]
        self.mask = np.stack(m, axis=0)  # [N, T], {0-1} mask
        self.start = np.stack(np.array([s for _, s, _ in self.data], dtype=float), axis=0)  # [N, D]
        self.target = np.stack(np.array([t for _, _, t, in self.data], dtype=float), axis=0)  # [N, D]
        
        self["x"] = torch.from_numpy(self.x)
        self["mask"] = torch.tensor(self.mask.astype(int))
        self["start"] = torch.from_numpy(self.start)
        self["target"] = torch.from_numpy(self.target)
        

    def model(self, env):
        xdata = self["x"] 
        mask = self["mask"]
        start = self["start"] 
        target = self["target"]
        
        # Linear policy
        log_cn_covar = env.param("log_cn_covar", torch.tensor([-2.0, -2.0]))  # control execution output noise
        cn_covar = torch.exp(log_cn_covar)
        
        log_pn_covar = env.param("log_pn_covar", torch.tensor(-2.0))  # policy noise covar
        pn_covar = torch.exp(log_pn_covar)
        
        policy = env.sample("policy", mcvb.Normal(loc=0.0, scale=10.0).expand_right(self.initialpolicy.shape))  # linear policy matrix
        
        N, T, D = xdata.shape

        # Process all N trials in parallel
        
        x = []  # list of sampled trajectories
        
        for t in range(3):
            x.append(env.sample("x_{}".format(t), 
                    mcvb.Normal(loc=0.0, scale=1.0).expand_right([N, D]).mask(mask[..., t][..., None]), obs=xdata[..., t, :]))
           
        for t in range(2, T-1):
            # t - current time step
            # t+1 - next time step
            v = x[t] - x[t-1]  # current velocity vector
            vnorm = stable_normalize_torch(v)
            nv_vnorm = rot_90(vnorm)
            vbasis = torch.stack([vnorm, nv_vnorm], dim=-1)  # from local to global
            vbasisinv = vbasis.transpose(-2, -1)  # from global to local. Because basis matrix is ortonormal, inverse is transpose
            v_local = torch.einsum("...ij,...j->...i", vbasisinv, v)
            
            vprev = x[t-1] - x[t-2]  # previous velocity vector
            aprev = v - vprev  # previous acceleration
            vprevnorm = stable_normalize_torch(vprev)
            nv_vprevnorm = rot_90(vprevnorm)
            vprevbasis = torch.stack([vprevnorm, nv_vprevnorm], dim=-1)  # from local to global
            vprevbasisinv = vprevbasis.transpose(-2, -1)  # from global to local. Because basis matrix is ortonormal, inverse is transpose
            aprev_local = torch.einsum("...ij,...j->...i", vprevbasisinv, aprev)

            #vbasisinv_ex = mcvb.extend_batch(vbasisinv, [target.shape[0]])
            vbasisinv_ex = vbasisinv
            vbasis_ex = mcvb.extend_batch(vbasis, [policy.shape[0]])
            target_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, target - x[t])

            # Construct the current state vector (regressors)
            regressors = []
            for r in self.regressortypes:
                if r == RegressorType.Const:
                    regressors.append(torch.ones(N, 1).to(policy))
                elif r == RegressorType.Position:
                    raise NotImplementedError()
                elif r == RegressorType.Velocity1D:
                    regressors.append(v_local[..., 0][..., None])
                elif r == RegressorType.Acceleration:
                    regressors.append(aprev_local)
                elif r == RegressorType.Control:
                    raise NotImplementedError()
                elif r == RegressorType.OptimalTarget:
                    regressors.append(target_local)
                elif r == RegressorType.OptimalTrajectory:  
                    vec_to_tr = vec_pts_to_lines_torch(x[t], start, target)
                    vec_to_tr_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, vec_to_tr)  # vector to optimal trajectory
                    regressors.append(vec_to_tr_local)
            state = mcvb.expand_cat(regressors, dim=-1)
            
            # policy: [nparticles, ndims, nregressors]
            # state: [nparticles, ntrials, nregressors]
            # prediction: [nparticles, ntrials, ndims] 
            policy_ex = mcvb.insert_batch(policy, [state.shape[-2]], 2)
            state = mcvb.extend_batch(state, [policy_ex.shape[0]])
            # TODO: compute local control
            raise NotImplementedError()
            control_local =   # control in local coordinates
            
            control_global = torch.einsum("...ij,...j->...i", vbasis_ex, control_local)
            
            # Control noise covariance
            covar_next, _ = signal_dependent_noise_covar_torch(control=control_global, scale=cn_covar, uniform=pn_covar)
            xnext_local = v_local + control_local
            loc_next = x[t] + torch.einsum("...ij,...j->...i", vbasis_ex, xnext_local)
            x.append(env.sample("x_{}".format(t+1), 
                    mcvb.MultivariateNormal(loc=loc_next, covariance_matrix=covar_next).mask(mask[..., t+1]), 
                    obs=xdata[..., t+1, :]))            
        

    def proposal(self, env):
        xdata = self["x"] 
        mask = self["mask"]
        start = self["start"] 
        target = self["target"]
        
        policy_loc = env.param("policy.loc", torch.zeros(self.initialpolicy.shape))
        policy_logscale = env.param("policy.logscale", -2.0 * torch.ones(self.initialpolicy.shape))
        policy = env.sample("policy", mcvb.Normal(loc=policy_loc, scale=torch.exp(policy_logscale)))

        

class Plotter(object):
    def __init__(self):
        self.vis = visdom.Visdom()
        self.vis.close(win="Loss")
        self.vis.close(win="Policy")
        self.vis.close(win="Noise")
        self.plot_iter = 0
        self.loss = []

    def iter_callback(self, iter, loss, model_sampler, proposal_sampler):
        print("Iter:", iter, "Loss:", loss)
        self.loss.append(loss)
        
        self.plot_iter += 1
        iter = self.plot_iter
        
        if iter >= 0:
            self.vis.line(X=[iter], Y=[-loss], win="Loss", 
                name=str(proposal_sampler._nparticles), 
                update="append", opts=dict(title="Loss"))

            try:
                y = torch.flatten(proposal_sampler.value("policy.loc"))[None, :]
                x = torch.zeros(y.shape) + iter
                self.vis.line(X=x, Y=y, win="Policy", update="append", opts=dict(title="Policy"))

                y = []
                if "log_cn_covar" in model_sampler._param:
                    y.append(model_sampler._param["log_cn_covar"])
                if "log_pn_covar" in model_sampler._param:
                    y.append(model_sampler._param["log_pn_covar"][None])
                if "policy.logscale" in proposal_sampler._param:
                    y.append(torch.flatten(proposal_sampler._param["policy.logscale"]))
                y = torch.cat(y, dim=-1)[None, :]
                
                x = torch.zeros(y.shape) + iter
                self.vis.line(X=x, Y=y, win="Noise", update="append", opts=dict(title="Noise"))
            except:
                pass


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    
    np.random.seed(1)
    torch.manual_seed(1)
    
    T = 100
    #tr = np.vstack([np.sort(np.random.uniform(0, 20, T)), np.linspace(0, 10, T)]).T  # noisy line
    #tr = np.array([[0, 1, 2, 2, 2, 3, 4, 4, 4], [0, 0, 0, 1, 2, 2, 2, 1, 0]]).T  # snake
    tr = np.array([[0, 0, 1, 2, 3, 3, 2, 1], [0, 1, 2, 2, 1, 0, -1, -1]]).T  # circle
    tr = tr.astype(np.double)

    regressortypes = [ \
            RegressorType.Velocity1D, 
            RegressorType.Acceleration, 
            RegressorType.OptimalTarget, 
            RegressorType.OptimalTrajectory,
            RegressorType.Const,
            ]
                
    devicename = "cpu"
    device = torch.device(devicename)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    trial_0 = (tr[:], tr[0] , tr[-1])
    trial_1 = (tr[:-1], tr[0] , tr[-1])
    trial_2 = (tr[:6], tr[0] , tr[-1])
    trial_min = (tr[:5], tr[0] , tr[-1])
    data = [trial_0, trial_1, trial_2]
    
    dirpath = "./plots/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    pl = Plotter()
    vb = mcvb.VB()
    lpm = LinearPolicyModel(data, regressortypes=regressortypes)
    
    ELBO = {}
    initvals = mcvb.TensorContainer()
    model = lpm.model
    
    lpm.to_device(device, dtype)
    vb.to_device(device, dtype)
    
    maxiter = 1000
    nparticles = 10
    model_sampler, proposal_sampler = vb.infer(model, lpm.proposal,
            initvals=initvals,
            nparticles=nparticles, lr=0.01, maxiter=maxiter, callback=pl.iter_callback)
    initvals.update(model_sampler._param)
    initvals.update(proposal_sampler._param)
    ELBO[devicename] = -pl.loss[-1]
    policy = proposal_sampler._param["policy.loc"].to("cpu", torch.float64).data.numpy()
    pprint_policy(policy, regressortypes)
    
    # Sample from the learned model
    plt.plot(tr[:, 0], tr[:, 1])
    for j in range(10):
        ignore_obs = set(["x_{}".format(t) for t in range(3, len(tr))])
        model_sample, proposal_sample = vb.sample(lpm.model, lpm.proposal,
                initvals=initvals,
                ignore_obs=ignore_obs)
        x = torch.stack([model_sample._randv["x_{}".format(t)] for t in range(3, len(tr))], dim=-2)
        x = x[0].data.numpy()
        plt.plot(x[:, 0], x[:, 1])
    plt.axis("equal")
    plt.savefig("./plots/trajectoy-sampled-{}.pdf".format(k))
    plt.close()
    
    with open("values-mcvb.json", "w") as f:
        initvals.json_dump(f, indent=2)
    
    print("ELBO:", ELBO)    
        

        
    
          
