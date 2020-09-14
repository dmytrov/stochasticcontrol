import numpy as np
import torch
import visdom
import analysis.variationalbayes.mcvb as mcvb



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


def sigmoid_torch(x, a=1, b=0):
    """ Sigmoid function
    """
    #return 1.0 / (1.0 + torch.exp(-a*(x - b)))  # numerically unstable, but directly interpretable
    #return 1.0 / (1.0 + torch.exp(- torch.abs(a)*(x - b)))  # numerically unstable, but directly interpretable
    #return 1.0 / (1.0 + torch.exp(torch.clamp(-a*(x - b), min=-10, max=10)))  # stable, interpretable
    #return 1.0 / (1.0 + torch.exp(-a*x + b))  # numerically most stable, use sigmoid_inflection_pt() to interpret
    return 1.0 / (1.0 + torch.exp(torch.clamp(-a*x + b, min=-10, max=10)))  # numerically most stable, use sigmoid_inflection_pt() to interpret


def sigmoid_inflection_pt(a=1, b=0):
    """ Inflection point (f(x_i)=0.5)
        -a*x+b = 0
        x_i = b / a
    """
    return b / a


def sigmoid_inflection_a_to_b(x_i, a):
    """ x_i = b / a
        b = x_i * a
    """
    return x_i * a

    
if False:
    import matplotlib.pyplot as plt
    x = torch.arange(-3.0, 3, 0.1)
    y = sigmoid_torch(x, a=5, b=-2)
    plt.plot(x.data.numpy(), y.data.numpy())
    plt.show()
    exit()

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


class TargetSwitchingModel(mcvb.TensorContainer):
    def __init__(self, 
            data,
            memsize=1,
            delaysize=0, 
            regressortypes=None,
            initialpolicy=None,
            fitsigmoid=True,
            fittrajectories=False,
            fitpolicy=True,
            fitnoise=True,
            policyactivation=False):
        """
            data : list of tumples (trajectory, ptstart, ptend, ptstart_sw, ptend_sw)
        """
        super(TargetSwitchingModel, self).__init__()
        self.data = data
        
        self.memsize = memsize
        self.delaysize = delaysize

        self.regressortypes = regressortypes
        self.fitsigmoid = fitsigmoid
        self.fittrajectories = fittrajectories
        self.fitpolicy = fitpolicy
        self.fitnoise = fitnoise
        self.policyactivation = policyactivation
        
        statesize = 0
        for r in regressortypes:
            if r == RegressorType.Const:
                statesize += 1
            else:
                statesize += RegressorType.statesize[r] * self.memsize
        if initialpolicy is None:
            initialpolicy = np.zeros([2, statesize])
        self.initialpolicy = initialpolicy

        # Construct extended data tensors with maxlength
        # to unify the model and a binary mask for the missing data
        N = len(self.data)  # number of trials
        lengths = np.array([len(x) for x, _, _, _, _ in self.data])  # [N]
        self.lengths = lengths
        maxlength = np.max(lengths)  # max trial length
        x = [np.concatenate([tr, 0.0 * np.zeros([maxlength-len(tr), 2])], axis=0) for tr, _, _, _, _ in self.data]   # [N, Xmaxlength, D]
        self.x = np.stack(x, axis=0)  # [N, T, D]
        m = [np.concatenate([np.ones(length), np.zeros(maxlength-length)])  for length in lengths]
        self.mask = np.stack(m, axis=0)  # [N, T], {0-1} mask
        self.disturbed = np.array([not np.allclose(b1, b2) for tr, a1, b1, a2, b2 in self.data])  # [N]
        self.start = np.stack(np.array([s for _, s, _, _, _ in self.data], dtype=float), axis=0)  # [N, D]
        self.target = np.stack(np.array([t for _, _, t, _, _ in self.data], dtype=float), axis=0)  # [N, D]
        self.start_dist = np.stack(np.array([s for _, _, _, s, _ in self.data], dtype=float), axis=0)  # [N, D], disturbed
        self.target_dist = np.stack(np.array([t for _, _, _, _, t in self.data], dtype=float), axis=0)  # [N, D], disturbed
        
        self["x"] = torch.from_numpy(self.x)
        self["mask"] = torch.tensor(self.mask.astype(int))
        self["disturbed"] = torch.tensor(self.disturbed.astype(int))
        self["start"] = torch.from_numpy(self.start)
        self["target"] = torch.from_numpy(self.target)
        self["start_dist"] = torch.from_numpy(self.start_dist)
        self["target_dist"] = torch.from_numpy(self.target_dist)


    def model(self, env):
        xdata = self["x"] 
        mask = self["mask"]
        disturbed = self["disturbed"] 
        start = self["start"] 
        target = self["target"]
        start_dist = self["start_dist"]
        target_dist =  self["target_dist"]

        # Linear policy
        log_cn_covar = env.param("log_cn_covar", torch.tensor([-2.0, -2.0]))  # control execution output noise
        cn_covar = torch.exp(log_cn_covar)
        
        log_pn_covar = env.param("log_pn_covar", torch.tensor(-2.0))  # policy noise covar
        pn_covar = torch.exp(log_pn_covar)
        
        policy = env.sample("policy", mcvb.Normal(loc=0.0, scale=10.0).expand_right(self.initialpolicy.shape))  # linear policy matrix
        
        N, T, D = xdata.shape

        # Process all N trials in parallel
        
        # Policy activation sigmoid (movement onset)
        if self.policyactivation:
            policy_act_log_sigm_a = env.param("policy_act_log_sigm_a", torch.ones(N)) 
            policy_act_sigm_a = torch.exp(policy_act_log_sigm_a)
            policy_act_sigm_b = env.param("policy_act_sigm_b", -2.0 * torch.ones(N))

        # Target switching sigmoid
        prior_inflection = 0.5
        prior_a = 10.0
        sigm_a = env.sample("sigm_a", mcvb.LogNormal(loc=np.log(prior_a), scale=5.0).expand_right([N]).mask(disturbed))  # [N], prior over sigmoid parameter a
        sigm_b = env.sample("sigm_b", mcvb.Normal(loc=prior_a * prior_inflection, scale=5.0).expand_right([N]).mask(disturbed))  # [N], prior over sigmoid parameter b
        env.store("sigm_a", sigm_a)
        env.store("sigm_b", sigm_b)

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

            # Sigmoid policy activation
            if self.policyactivation:
                policy_act_sigm = sigmoid_torch(x=t/T, a=policy_act_sigm_a, b=policy_act_sigm_b)
            
            # Sigmoidal switching of start and target
            sigm = sigmoid_torch(x=t/T, a=sigm_a, b=sigm_b)  # sigmoidal weight for every time point
            env.store("sigmoid_{}".format(t+1), sigm)
            sigm = sigm[..., None]  # broadcast to dimensions
            start_sigm = start * (1.0 - sigm) +  start_dist * sigm
            target_sigm = target * (1.0 - sigm) +  target_dist * sigm
            vbasisinv_ex = mcvb.extend_batch(vbasisinv, [target_sigm.shape[0]])
            vbasis_ex = mcvb.extend_batch(vbasis, [target_sigm.shape[0]])
            target_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, target_sigm - x[t])

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
                    vec_to_tr = vec_pts_to_lines_torch(x[t], start_sigm, target_sigm)
                    vec_to_tr_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, vec_to_tr)  # vector to optimal trajectory
                    regressors.append(vec_to_tr_local)
            state = mcvb.expand_cat(regressors, dim=-1)
            
            # policy: [nparticles, ndims, nregressors]
            # state: [nparticles, ntrials, nregressors]
            # prediction: [nparticles, ntrials, ndims] 
            policy_ex = mcvb.insert_batch(policy, [state.shape[-2]], 2)
            control_local = torch.einsum("...ij,...j->...i", policy_ex, state)  # control in local coordinates
            
            if self.policyactivation:
                control_local =  policy_act_sigm[None, :, None] * control_local  # modulate the control
            control_global = torch.einsum("...ij,...j->...i", vbasis_ex, control_local)
            
            # Control noise covariance
            covar_next, _ = signal_dependent_noise_covar_torch(control=control_global, scale=cn_covar, uniform=pn_covar)
            xnext_local = v_local + control_local
            loc_next = x[t] + torch.einsum("...ij,...j->...i", vbasis_ex, xnext_local)
            x.append(env.sample("x_{}".format(t+1), 
                    mcvb.MultivariateNormal(loc=loc_next, covariance_matrix=covar_next).mask(mask[..., t+1]), 
                    obs=xdata[..., t+1, :]))
            
        return x


    def model_fast(self, env):
        """ Fast but doesn't support sampling
        """
        xdata = self["x"] 
        mask = self["mask"]
        disturbed = self["disturbed"] 
        start = self["start"] 
        target = self["target"]
        start_dist = self["start_dist"]
        target_dist =  self["target_dist"]

        # Linear policy
        log_cn_covar = env.param("log_cn_covar", torch.tensor(np.log([1.0, 1.0])))  # control execution output noise
        cn_covar = torch.exp(log_cn_covar)
        
        log_pn_covar = env.param("log_pn_covar", torch.tensor(np.log(1.0)))  # policy noise covar
        pn_covar = torch.exp(log_pn_covar)
        #pn_covar = pn_covar * 0.0
        
        policy = env.sample("policy", mcvb.Normal(loc=0.0, scale=10.0).expand_right(self.initialpolicy.shape))  # linear policy matrix
        
        N, T, D = xdata.shape

        # Process all N trials in parallel
        
        # Policy activation sigmoid (movement onset)
        if self.policyactivation:
            policy_act_log_sigm_a = env.param("policy_act_log_sigm_a", torch.ones(N)) 
            policy_act_sigm_a = torch.exp(policy_act_log_sigm_a)
            policy_act_sigm_b = env.param("policy_act_sigm_b", -2.0 * torch.ones(N))

        # Target switching sigmoid
        prior_inflection = 0.5
        prior_a = 1.0
        sigm_a = env.sample("sigm_a", mcvb.LogNormal(loc=np.log(prior_a), scale=1.0).expand_right([N]).mask(disturbed))  # [N], prior over sigmoid parameter a
        sigm_b = env.sample("sigm_b", mcvb.Normal(loc=prior_inflection * prior_a, scale=1.0).expand_right([N]).mask(disturbed))  # [N], prior over sigmoid parameter b
        
        #x = []  # list of sampled trajectories
        x = env.observe("xdata", xdata)

        for t in range(2 + self.memsize + self.delaysize):
            env.sample("x_{}".format(t),
                    mcvb.Normal(loc=0.0, scale=1.0).expand_right([N, D]).mask(mask[..., t][..., None]), 
                    obs=xdata[..., t, :])
        
        xtm2 = x[:, 0:T-3, :]  # prev, t-1
        xtm1 = x[:, 1:T-2, :]  # prev, t-1
        xt = x[:, 2:T-1, :]  # current, t
        xtp1 = x[:, 3:T, :]  # next, t+1

        # t - current time step
        # t+1 - next time step
        v = xt - xtm1  # current velocity vector
        vnorm = stable_normalize_torch(v)
        nv_vnorm = rot_90(vnorm)
        vbasis = torch.stack([vnorm, nv_vnorm], dim=-1)  # from local to global
        vbasisinv = vbasis.transpose(-2, -1)  # from global to local. Because basis matrix is ortonormal, inverse is transpose
        v_local = torch.einsum("...ij,...j->...i", vbasisinv, v)
            
        vprev = xtm1 - xtm2  # previous velocity vector
        aprev = v - vprev  # previous acceleration
        vprevnorm = stable_normalize_torch(vprev)
        nv_vprevnorm = rot_90(vprevnorm)
        vprevbasis = torch.stack([vprevnorm, nv_vprevnorm], dim=-1)  # from local to global
        vprevbasisinv = vprevbasis.transpose(-2, -1)  # from global to local. Because basis matrix is ortonormal, inverse is transpose
        aprev_local = torch.einsum("...ij,...j->...i", vprevbasisinv, aprev)

        t = torch.arange(2, T-1, device=self.device, dtype=self.dtype).to(xdata)

        # Sigmoid policy activation
        if self.policyactivation:
            policy_act_sigm = sigmoid_torch(x=t/T, a=policy_act_sigm_a[..., None], b=policy_act_sigm_b[..., None])
              
        # Sigmoidal switching of start and target
        sigm = sigmoid_torch(x=t/T, a=sigm_a[..., None], b=sigm_b[..., None])  # sigmoidal weight for every time point
        sigm = sigm[..., None]  # broadcast to dimensions
        start_sigm = start[:, None, :] * (1.0 - sigm) +  start_dist[:, None, :] * sigm
        target_sigm = target[:, None, :] * (1.0 - sigm) +  target_dist[:, None, :] * sigm
        vbasisinv_ex = mcvb.extend_batch(vbasisinv, [target_sigm.shape[0]])
        vbasis_ex = mcvb.extend_batch(vbasis, [target_sigm.shape[0]])
        target_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, target_sigm - xt)
        
        # Construct the current state vector (regressors)
        regressors = []
        for r in self.regressortypes:
            if r == RegressorType.Position:
                raise NotImplementedError()
            elif r == RegressorType.Velocity1D:
                regressors.append(v_local[..., 0][..., None])  # add higher order
            elif r == RegressorType.Acceleration:
                regressors.append(aprev_local)  # add higher order
            elif r == RegressorType.Control:
                raise NotImplementedError()
            elif r == RegressorType.OptimalTarget:
                regressors.append(target_local)  # add higher order
            elif r == RegressorType.OptimalTrajectory:  
                vec_to_tr = vec_pts_to_lines_torch(xt[None, ...], start_sigm, target_sigm)
                vec_to_tr_local = torch.einsum("...ij,...j->...i", vbasisinv_ex, vec_to_tr)  # vector to optimal trajectory
                regressors.append(vec_to_tr_local)  # add higher order
        state = mcvb.expand_cat(regressors, dim=-1)

        # Tile the state for higher order regression
        state = torch.cat([state[..., i:i-self.memsize+len(t)+1, :] for i in range(self.memsize)], dim=-1)

        # Add single const regressor
        if r == RegressorType.Const:
            state = mcvb.expand_cat([state, torch.ones(N, state.shape[-2], 1).to(policy)], dim=-1)
        
        # policy: [nparticles, ndims, nregressors]
        # state: [nparticles, ntrials, nt, nregressors]
        # prediction: [nparticles, ntrials, nt, ndims] 
        policy_ex = mcvb.insert_batch(policy, [state.shape[-3], state.shape[-2]], 2)
        if policy_ex.dim() == state.dim() + 2:
            state = mcvb.extend_batch(state, [policy_ex.shape[0]])
        control_local = torch.einsum("...ij,...j->...i", policy_ex, state) # control in local coordinates
        
        if self.policyactivation:
            control_local = policy_act_sigm[None, :, :, None] * control_local
        
        vbasis_ex = vbasis_ex[..., self.memsize-1:, :, :]
        v_local = v_local[:, self.memsize-1:, :]

        control_global = torch.einsum("...ij,...j->...i", vbasis_ex, control_local)
        control_global_norm = stable_normalize_torch(control_global)
        vn_control_global_norm = rot_90(control_global_norm)
        controlbasis = torch.stack([control_global_norm, vn_control_global_norm], dim=-1) 
        controlbasisinv = controlbasis.transpose(-2, -1)

        def global_to_control(a):
            a_control = torch.einsum("...ij,...j->...i", controlbasisinv, a)    
            return a_control

        def local_to_control(a):
            a_global = torch.einsum("...ij,...j->...i", vbasis_ex, a)
            a_control = global_to_control(a_global)
            return a_control
        
        control_control = local_to_control(control_local)
        covar_next, _ = signal_dependent_noise_covar_xaligned_torch(control_control, cn_covar, pn_covar)
        std_next = torch.sqrt(covar_next)

        xnext_local = v_local + control_local
        xnext_control = local_to_control(xnext_local)

        xpt1_xt = (xtp1 - xt)[:, self.memsize-1:, :]
        xtp1_control = global_to_control(xpt1_xt.expand(xnext_local.shape))
        
        tp1 = torch.arange(2+self.memsize+self.delaysize, T, device=self.device)
        s = xtp1_control.shape[-2] - self.delaysize  # valid size for prediction
        env.sample("xtp1_control", 
                mcvb.Normal(loc=xnext_control[..., :s, :], 
                            scale=std_next[..., :s, :]).mask(mask[..., tp1, None]), 
                obs=xtp1_control[..., -s:, :])
        
       
    def proposal(self, env):
        self.proposal_factorized(env)


    def proposal_factorized(self, env):
        xdata = self["x"] 
        mask = self["mask"]
        disturbed = self["disturbed"]
        start = self["start"] 
        target = self["target"]
        start_dist = self["start_dist"]
        target_dist =  self["target_dist"]
        
        N, T, D = xdata.shape
        
        prior_inflection = 0.5
        prior_a = 1.0

        log_sigm_a_loc = env.param("log_sigm_a.loc", np.log(prior_a) * torch.ones(N))
        log_sigm_a_logscale = env.param("log_sigm_a.logscale", np.log(1.0) * torch.ones(N))
        sigm_a = env.sample("sigm_a", mcvb.LogNormal(loc=log_sigm_a_loc, scale=torch.exp(log_sigm_a_logscale)).mask(disturbed))  # [N], prior over sigmoid parameter a
        
        sigm_b_loc = env.param("sigm_b.loc", prior_inflection * prior_a * torch.ones(N))
        sigm_b_logscale = env.param("sigm_b.logscale", np.log(1.0) * torch.ones(N))
        sigm_b = env.sample("sigm_b", mcvb.Normal(loc=sigm_b_loc, scale=torch.exp(sigm_b_logscale)).mask(disturbed))  # [N], prior over sigmoid parameter b

        policy_loc = env.param("policy.loc", torch.zeros(self.initialpolicy.shape))
        policy_logscale = env.param("policy.logscale", np.log(1.0) * torch.ones(self.initialpolicy.shape))
        policy = env.sample("policy", mcvb.Normal(loc=policy_loc, scale=torch.exp(policy_logscale)))

        

def run_policy(policy, start, target, n=5, regressortypes=None):
    
    def rot_90(x):
        return x[[1, 0]] * np.array([-1.0, 1.0])

    x = np.zeros([n, 2])
    if start is None:
        x[:3] = np.array([[0.0, 0.0], [0.01, 0.01], [0.02, 0.02]])
    else:
        x[:3] = start

    if regressortypes is None:
        regressortypes = [RegressorType.Velocity, RegressorType.Acceleration, RegressorType.OptimalTarget, RegressorType.Const]
        regressortypes.append(RegressorType.OptimalTrajectory)
    
    for i in range(2, n-1):
        # i - current time step
        # i+1 - next time step
        v = x[i] - x[i-1]  # current velocity vector
        vnorm = stable_normalize(v)
        nv_vnorm = rot_90(vnorm)
        vbasis = np.vstack([vnorm, nv_vnorm]).T  # from local to global
        vbasisinv = vbasis.T  # from global to local. Because basis matrix is ortonormal, inverse is transpose
        v_local = np.dot(vbasisinv, v)

        vprev = x[i-1] - x[i-2]  # previous velocity vector
        aprev = v - vprev  # previous acceleration
        vprevnorm = stable_normalize(vprev)
        nv_vprevnorm = rot_90(vprevnorm)
        vprevbasis = np.vstack([vprevnorm, nv_vprevnorm]).T  # from local to global
        vprevbasisinv = vprevbasis.T  # from global to local. Because basis matrix is ortonormal, inverse is transpose
        aprev_local = np.dot(vprevbasisinv, aprev)

        target_local = np.dot(vbasisinv, target - x[i])

        vec_to_tr = vec_pts_to_line([x[i]], start[0], target)[0]
        vec_to_tr_local = np.dot(vbasisinv, vec_to_tr)  # vector to optimal traajectory
        
        regressors = []
        for r in regressortypes:
            if r == RegressorType.Const:
                regressors.append([1])
            elif r == RegressorType.Position:
                raise NotImplementedError()
            elif r == RegressorType.Velocity1D:
                regressors.append(v_local[0])
            elif r == RegressorType.Acceleration:
                regressors.append(aprev_local)
            elif r == RegressorType.Control:
                raise NotImplementedError()
            elif r == RegressorType.OptimalTarget:
                regressors.append(target_local)
            elif r == RegressorType.OptimalTrajectory:  
                regressors.append(vec_to_tr_local)
        state = np.hstack(regressors) 
        control = policy.dot(state)
        xnext_local = v_local + control
        x[i+1] = x[i] + vbasis.dot(xnext_local)
       
    return x


class Plotter(object):
    def __init__(self):
        self.vis = visdom.Visdom()
        self.vis.close(win="Loss")
        self.vis.close(win="Policy")
        self.vis.close(win="Noise")
        self.vis.close(win="Sigm_a")
        self.vis.close(win="Sigm_b")
        self.vis.close(win="Sigm_a.scale")
        self.vis.close(win="Sigm_b.scale")
        self.vis.close(win="Sigm_inflection")
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

                sigm_a = torch.flatten(torch.exp(proposal_sampler.value("log_sigm_a.loc")))[None, :]
                x = torch.zeros(sigm_a.shape) + iter
                self.vis.line(X=x, Y=sigm_a, win="Sigm_a", update="append", opts=dict(title="sigm_a"))

                y = torch.flatten(torch.exp(proposal_sampler.value("log_sigm_a.logscale")))[None, :]
                x = torch.zeros(y.shape) + iter
                self.vis.line(X=x, Y=y, win="Sigm_a.scale", update="append", opts=dict(title="sigm_a.scale"))

                sigm_b = torch.flatten(proposal_sampler.value("sigm_b.loc"))[None, :]
                x = torch.zeros(sigm_b.shape) + iter
                self.vis.line(X=x, Y=sigm_b, win="Sigm_b", update="append", opts=dict(title="sigm_b"))

                y = torch.flatten(torch.exp(proposal_sampler.value("sigm_b.logscale")))[None, :]
                x = torch.zeros(y.shape) + iter
                self.vis.line(X=x, Y=y, win="Sigm_b.scale", update="append", opts=dict(title="sigm_b.scale"))

                sigm_inflection = sigmoid_inflection_pt(a=sigm_a, b=sigm_b)
                x = torch.zeros(sigm_inflection.shape) + iter
                self.vis.line(X=x, Y=sigm_inflection, win="Sigm_inflection", update="append", opts=dict(title="sigm_inflection"))


                y = []
                if "log_cn_covar" in model_sampler._param:
                    y.append(model_sampler._param["log_cn_covar"])
                if "log_pn_covar" in model_sampler._param:
                    y.append(model_sampler._param["log_pn_covar"][None])
                if "policy.logscale" in proposal_sampler._param:
                    y.append(torch.flatten(proposal_sampler._param["policy.logscale"]))
                y = torch.exp(torch.cat(y, dim=-1))[None, :]
                
                x = torch.zeros(y.shape) + iter
                self.vis.line(X=x, Y=y, win="Noise", update="append", opts=dict(title="Noise"))
            except:
                pass


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    

    # NAN anomaly detection
    from torch import autograd
    #with autograd.detect_anomaly():
    #torch.set_anomaly_enabled(True)

    np.random.seed(1)
    torch.manual_seed(1)
    
    T = 100
    #tr = np.array([[0, 1, 3, 6, 10], [5, 5.1, 4.9, 5.1, 4.9]]).T
    #tr = np.array([[0, 1, 3, 6, 10], [5, 5, 5, 5, 5]]).T
    #tr = np.array([[0, 1, 3, 6, 10], [0, 1, 3, 6, 10]]).T
    #tr = np.array([[2, 4, 6, 8, 10], [1, 2.01, 3.3, 4.1, 5]]).T
    #tr = np.vstack([np.sort(np.random.uniform(0, 20, T)), np.linspace(0, 10, T)]).T
    #tr = np.array([[0, 1, 2, 2, 2, 3, 4, 4, 4], [0, 0, 0, 1, 2, 2, 2, 1, 0]]).T  # snake
    #tr = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 1], [0, 0, 0, 0, 1, 2, 2, 1, 0, -1, -1]]).T  # circle
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

    trial_0 = (tr[:], tr[0] , tr[-1], tr[0] , tr[-1])
    trial_1 = (tr[:-1], tr[0] , tr[-1], tr[0] , tr[-1])
    trial_2 = (tr[:6], tr[0] , tr[-1], tr[0] , tr[-1])
    trial_min = (tr[:5], tr[0] , tr[-1], tr[0] , tr[-1])
    data = [trial_0, trial_1, trial_2]
    
    dirpath = "./plots/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    pl = Plotter()
    vb = mcvb.VB()
    targetswitching = TargetSwitchingModel(data, regressortypes=regressortypes, 
            fittrajectories=False, memsize=1, delaysize=0,
            policyactivation=False)
    
    ELBO = {}
    initvals = mcvb.TensorContainer()
    #model = targetswitching.model
    model = targetswitching.model_fast
    for k in range(1):
        targetswitching.to_device(device, dtype)
        vb.to_device(device, dtype)
        
        maxiter = 2000
        nparticles = 10
        model_sampler, proposal_sampler = vb.infer(model, targetswitching.proposal,
                initvals=initvals,
                nparticles=nparticles, lr=0.01, maxiter=maxiter, callback=pl.iter_callback)
        initvals.update(model_sampler._param)
        initvals.update(proposal_sampler._param)
        ELBO[devicename] = -pl.loss[-1]
        policy = proposal_sampler._param["policy.loc"].to("cpu", torch.float64).data.numpy()
        
        pprint_policy(policy, regressortypes)
        x = run_policy(policy, tr[:3], tr[-1], n=len(tr), regressortypes=regressortypes)
        
        plt.plot(tr[:, 0], tr[:, 1])
        plt.plot(x[:, 0], x[:, 1])
        plt.axis("equal")
        plt.savefig("./plots/targetswitching-mcvb-{}.pdf".format(k))
        plt.close()

        plt.plot(tr[:, 0], tr[:, 1])
        for j in range(10):
            ignore_obs = set(["x_{}".format(t) for t in range(3, len(tr))])
            # Sampling can be performed only from the full model!!!
            model_sample, proposal_sample = vb.sample(targetswitching.model, targetswitching.proposal,
                    initvals=initvals,
                    ignore_obs=ignore_obs)
            x = torch.stack([model_sample._randv["x_{}".format(t)] for t in range(3, len(tr))], dim=-2)
            x = x[0].data.numpy()
            plt.plot(x[:, 0], x[:, 1])
        plt.axis("equal")
        plt.savefig("./plots/targetswitching-sampled-{}.pdf".format(k))
        plt.close()
    
    with open("values-mcvb.json", "w") as f:
        initvals.json_dump(f, indent=2)
    
    log_p_data = None
    for k in range(1):
        maxiter = 5000
        model_sampler, proposal_sampler, log_prob = vb.optimize(model,
                initvals=initvals,
                maxiter=maxiter, callback=pl.iter_callback)
        initvals.update(model_sampler._param)
        initvals.update(proposal_sampler._param)
        policy = proposal_sampler._param["policy.loc"].data.numpy()
        
        pprint_policy(policy, regressortypes)
        x = run_policy(policy, tr[:3], tr[-1], n=len(tr), regressortypes=regressortypes)
        
        plt.plot(tr[:, 0], tr[:, 1])
        plt.plot(x[:, 0], x[:, 1])
        plt.axis("equal")
        plt.savefig("./plots/targetswitching-bfgs-{}.pdf".format(k))
        plt.close()

        log_p_data = mcvb.marginal_log_p_data_Laplace(proposal_sampler, model_sampler)
    
    with open("values-bfgs.json", "w") as f:
        initvals.json_dump(f, indent=2)
    
    print("ELBO:", ELBO)    
    print("Laplace log_p(D):", log_p_data)
        

    for name, mdl in [("normal", targetswitching.model), ("fast", targetswitching.model_fast)]:
        model_sampler, proposal_sampler, full_log_prob = vb.optimize(mdl,
                initvals=initvals,
                maxiter=0)

        print(name, "full_log_prob:", full_log_prob)
        
    
          
