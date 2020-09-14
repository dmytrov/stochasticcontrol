import json
import numpy as np
import torch
import torch.distributions as distr
from torch.distributions.utils import broadcast_all
from torch.distributions import constraints
from numbers import Number
import scipy.optimize as opt
import analysis.variationalbayes.modelcomparison as modelcomparison
import analysis.variationalbayes.torchjson as torchjson

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
            maxls=100, 
            #factr=0, 
            #pgtol=1.0e-8, 
            callback=callback)
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


def add_dims_right(tensor, ndims, right_indent=0):
    """ Add empty dimensions to the right of tensor shape
    """
    assert right_indent >= 0
    for i in range(ndims):
        tensor = torch.unsqueeze(tensor, -1-right_indent)
    return tensor


def insert_batch(tensor, insert_shape, right_indent=0):
    """ Insert batch dimensions from the right
    """
    assert right_indent >= 0
    insert_shape = torch.Size(insert_shape)
    N = len(tensor.shape)
    new_shape = tensor.shape[:N-right_indent] + insert_shape + tensor.shape[N-right_indent:]
    new_tensor = add_dims_right(tensor, len(insert_shape), right_indent).expand(new_shape)
    return new_tensor


def extend_batch(tensor, extend_shape):
    """ Extend tensor to the left
    """
    extend_shape = torch.Size(extend_shape)
    return tensor.expand(extend_shape + tensor.shape)


def expand_cat(ts, dim=-1):
    """ Expand all dims except dim. Then concatenate along dim
    """
    ms = max([len(t.shape) for t in ts])
    es = np.array([torch.Size([1]*(ms-len(t.shape))) + t.shape for t in ts])
    mx = np.max(es, axis=0)
    ss = np.tile(mx, [len(ts), 1])
    ss[:, dim] = es[:, dim]
    ts = [t.expand(torch.Size(s)) for t, s in zip(ts, ss)]
    return torch.cat(ts, dim)


class DistrMixin(object):
    def mask(self, mask):
        return MaskedDistribution(self, mask)

    def to_device(self, device, dtype=torch.float64):
        """ Not sure whether this method is useful at all.
            Probably - not.
        """
        pass


class MaskedDistribution(distr.Distribution, DistrMixin):
    has_rsample = True

    def __init__(self, inner, mask):
        assert isinstance(inner, distr.Distribution)
        self.inner = inner
        self._mask = mask.byte()
        
    def to_device(self, device, dtype=torch.float64):
        self._mask = self._mask.to(device)
        self.inner.to_device(device, dtype)

    def rsample(self, sample_shape=torch.Size()):
        return self.inner.rsample(sample_shape)

    def log_prob(self, value):
        lp = self.inner.log_prob(value)
        lp, mask = broadcast_all(lp, self._mask)
        lp = lp.masked_fill(~mask, 0.0)
        return lp
        
    @property
    def batch_shape(self):
        return self.inner._batch_shape

    @property
    def event_shape(self):
        return self.inner._event_shape

    def __repr__(self):
        return "Masked({}, mask:{})".format(self.inner, self._mask.shape)





class Normal(distr.Normal, DistrMixin):
    def __init__(self, loc, scale, validate_args=None):
        super(Normal, self).__init__(loc, scale, validate_args)
        
    def to_device(self, device, dtype=torch.float64):
        self.loc = self.loc.to(device, dtype)
        self.scale = self.scale.to(device, dtype)

    @property
    def event_shape(self):
        return torch.Size([])
    
    def expand_right(self, batch_shape, _instance=None):
        """ Expand batch dimensions right
        """
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = insert_batch(self.loc, batch_shape)
        new.scale = insert_batch(self.scale, batch_shape)
        super(distr.Normal, new).__init__(new.loc.shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    
class MultivariateNormal(distr.MultivariateNormal, DistrMixin):
    def expand_right(self, batch_shape, _instance=None):
        """ Expand batch dimensions right
        """
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new_batch_shape = torch.Size(self.batch_shape + batch_shape)
        es = self.event_shape
        new.loc = insert_batch(self.loc, batch_shape, len(es))
        new._unbroadcasted_scale_tril = insert_batch(self._unbroadcasted_scale_tril, batch_shape, len(es + es))
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix =  insert_batch(self.covariance_matrix, batch_shape, len(es + es))
        if 'scale_tril' in self.__dict__:
            new.scale_tril = insert_batch(self.scale_tril, batch_shape, len(es + es))
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = insert_batch(self.precision_matrix, batch_shape, len(es + es))
        super(distr.MultivariateNormal, new).__init__(new_batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new

    def to_device(self, device, dtype=torch.float64):
        if 'covariance_matrix' in self.__dict__:
            self.covariance_matrix = self.covariance_matrix.to(device, dtype)
        if 'scale_tril' in self.__dict__:
            self.scale_tril = self.scale_tril.to(device, dtype)
        if 'precision_matrix' in self.__dict__:
            self.precision_matrix = self.precision_matrix.to(device, dtype)
    


class LogNormal(distr.LogNormal, DistrMixin):
    def to_device(self, device, dtype=torch.float64):
        self.base_dist.loc = self.base_dist.loc.to(device, dtype)
        self.base_dist.scale = self.base_dist.scale.to(device, dtype)

    def expand_right(self, batch_shape, _instance=None):
        """ Expand batch dimensions right
        """
        loc = insert_batch(self.base_dist.loc, batch_shape)
        scale = insert_batch(self.base_dist.scale, batch_shape)
        return LogNormal(loc, scale)


class Gamma(distr.Gamma, DistrMixin):
    def to_device(self, device, dtype=torch.float64):
        self.concentration = self.concentration.to(device, dtype)
        self.rate = self.rate.to(device, dtype)



def tensor_list_to(tlist, device=None, dtype=torch.float64):
    return [t.to(device, dtype if t.is_floating_point() else None) for t in tlist]

def tensor_dict_to(tdict, device=None, dtype=torch.float64):
    return {name: t.to(device, dtype if t.is_floating_point() else None) for name, t in tdict.items()}

def distr_dict_to(ddict, device=None, dtype=torch.float64):
    return {name: d.to(device, dtype) for name, d in ddict.items()}


class TensorContainer(object):
    """ Allows easy transfer of tensor buffers
    """
    def __init__(self, device=None, dtype=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        self._named_buffers = {}
        
    def to_device(self, device, dtype=torch.float64):
        if device != self.device or self.dtype != dtype:
            self.device = device
            self.dtype = dtype
            self._named_buffers = tensor_dict_to(self._named_buffers, device, dtype)            

    def items(self):
        return self._named_buffers.items()
    
    def update(self, other):
        self._named_buffers.update(other)

    def pop(self, key):
        return self._named_buffers.pop(key)

    def __getitem__(self, name):
        return self._named_buffers[name]

    def __setitem__(self, key, value):
        self._named_buffers[key] = value

    def __contains__(self, key):
        return key in self._named_buffers

    def __iter__(self):
        return iter(self._named_buffers)

    def __len__(self):
        return len(self._named_buffers)

    def __keytransform__(self, key):
        return key

    def json_dumps(self, *args, **kwargs):
        return torchjson.dumps(self._named_buffers, *args, **kwargs)

    def json_dump(self, *args, **kwargs):
        return torchjson.dump(self._named_buffers, *args, **kwargs)

    def json_loads(self, *args, **kwargs):
        self._named_buffers = torchjson.loads(*args, **kwargs)
        return self
    
    def json_load(self, *args, **kwargs):
        self._named_buffers = torchjson.load(*args, **kwargs)
        return self

    def __repr__(self):
        sp = "\n".join(["  device: {}/{}".format(self.device, self.dtype)] +
                       ["  '{}': {} = {}".format(name, value.shape, value) for name, value in self._named_buffers.items()])
        return sp    

class Sampler(TensorContainer):
    def __init__(self, param=None, const=None, randv=None, obser=None, nparticles=1, 
            device=None, dtype=torch.float64):
        super(Sampler, self).__init__(device, dtype)
        
        self._param = {}  # {name: value}           optimized parameters
        self._const = {}  # {name: value}           constant parameters
        self._distr = {}  # {name: distr}           distributions
        self._randv = {}  # {name: random_value}    random values sampled from the distributions
        self._obser = {}  # {name: observed_value}  observed values
        self._store = {}  # {name: value}           miscelaneous logging, debug, temporary inner variables
        if param is not None:
            self._param = param
        if const is not None:
            self._const = const
        if randv is not None:
            self._randv = randv
        if obser is not None:
            self._obser = obser
        self.to_device(self.device, self.dtype)
        self._nparticles = nparticles
        # Set of free vars that can't be observed. Used for pure generative mode
        # None, all, or set of names
        self.ignore_obs = None
        self.verbose = False
        
    
    def to_device(self, device, dtype=torch.float64):
        super(Sampler, self).to_device(device, dtype)
        self._param = tensor_dict_to(self._param, device=self.device, dtype=self.dtype)
        self._const = tensor_dict_to(self._const, device=self.device, dtype=self.dtype)
        self._randv = tensor_dict_to(self._randv, device=self.device, dtype=self.dtype)
        self._obser = tensor_dict_to(self._obser, device=self.device, dtype=self.dtype)
        self._distr = distr_dict_to(self._distr, device=self.device, dtype=self.dtype)


    def param(self, name, initval, replace=False):
        """ Create free parameter
        """
        if replace or name not in self._param:
            if self.verbose:
                print("Creating parameter {} at {}".format(name, self.device.type))
            if not isinstance(initval, torch.Tensor):
                initval = torch.tensor(initval)
            self._param[name] = initval.to(device=self.device, dtype=self.dtype).detach().requires_grad_(True)
        return self._param[name]


    def const(self, name, initval):
        """ Create const parameter
        """
        if name not in self._const:
            if self.verbose:
                print("Creating const {} at {}".format(name, self.device.type))
            self._const[name] = torch.tensor(initval, requires_grad=False,
                    device=self.device, dtype=self.dtype)
        return self._const[name]


    def sample(self, name, distr, obs=None):
        """ Sample a random variable from a distribution
        """
        if name not in self._distr:
            distr.to_device(self.device, self.dtype)
            self._distr[name] = distr
        if obs is None:
            if name not in self._randv:
                self._randv[name] = distr.rsample(sample_shape=[self._nparticles])  # here goes sampling expansion
            return self._randv[name]
        else:
            if self.ignore_obs is all or (isinstance(self.ignore_obs, set) and name in self.ignore_obs):
                # Force pure generative mode
                sample = distr.rsample()
                if len(sample.shape) == len(obs.shape) + 1:
                    sample = sample[0]  # here goes sampling contraction
                self._randv[name] = sample
                return self._randv[name]
            else:
                # Store observed data
                self._obser[name] = obs
                return self._obser[name]


    def observe(self, name, obs):
        """ Add a disconnected observed variable.
            Disconnected observed variable can be transformed and 
            connected to the model in Sampler.sample().
            Such model can't be run in generative mode!!!
        """
        if name not in self._obser:
            self._obser[name] = obs.to(device=self.device)
        return self._obser[name]

    
    def store(self, name, value):
        """ Store additional value for debugging or plotting
        """
        self._store[name] = value


    def get_stored(self, name):
        """ Return the sored value
        """
        return self._store[name]

    
    def update_values(self, initvalues):
        if initvalues is None:
            return
        for name, value in initvalues.items():
            if name in self._param:
                self.param(name, value, replace=True)
            if name in self._const:
                self._const[name] = value
            if name in self._obser:
                self._obser[name] = value


    def copy_values(self, initvalues):
        if initvalues is None:
            return
        self.update_values(initvalues)
        for name, value in initvalues.items():
            self._param[name] = value
            
    
    def value(self, name):
        if name in self._param:
            return self._param[name]
        if name in self._const:
            return self._const[name]
        if name in self._obser:
            return self._obser[name]
        if name in self._randv:
            return self._randv[name]
        return None


    def log_prob(self):
        """ Log-probability of the samples
        """
        s = 0.0
        for name, value in {**self._randv, **self._obser}.items():
            if name in self._distr:
                lp = self._distr[name].log_prob(value)
                if self.verbose:
                    print(name, value, self._distr[name], lp)
                s += torch.sum(lp)
            else:
                #print("WARNING: observed {} has no distribution!".format(name))
                pass
        return s
        

    def parameters(self):
        """ Return list of parameters
        """
        for name, value in self._param.items():
            if not value.is_leaf:
                print("WARNING: parameter tensor \"{}\" is not a leaf!".format(name))
        return list(self._param.values())
    

    def named_parameters(self):
        """ Return dict of {name: parameter}
        """
        return self._param


    def __repr__(self):
        sp = "\n".join(["  param {}: {} = {}".format(name, value.shape, value) for name, value in self._param.items()] +
                       ["  const {}: {} = {}".format(name, value.shape, value) for name, value in self._const.items()] + 
                       ["  randv {}: {}".format(name, value.shape) for name, value in self._randv.items()] + 
                       ["  distr {}: {} {}".format(name, distr.batch_shape, distr) for name, distr in self._distr.items()] +
                       ["  obser {}: {}".format(name, value.shape) for name, value in self._obser.items()])
        return sp


def update_if_exists(dest, src):
    for name, value in src.items():
        if name in dest:
            dest[name] = value


class VB(TensorContainer):

    class SmallChangeStopCriterion(object):
        def __init__(self, max_change=0.01, lr=0.01):
            self.max_change = max_change
            self.lr = lr
            self.old_params = None

        def __call__(self, params):
            def to_1d(x):
                x = x.flatten()
                if x.dim() == 0:
                    x = x.unsqueeze(0)
                return x

            new_params = torch.cat([to_1d(param) for param in params])
            if self.old_params is None:
                res = False
            else:
                res = torch.all(torch.abs((new_params - self.old_params)/self.old_params) 
                        < self.lr * self.max_change)
            self.old_params = new_params
            return res

    class GradStatsCollector(object):
        def __init__(self, callback=None):
            self.n = 5  # collect n grads
            self.grads = []
            self.callback = None
        
        def __call__(self, params):
            """ Return: 
                 - True to accept the gradient and proceed,
                 - False to re-estimate the gradient 
            """
            if self.callback is None:
                return True
            self.grads.append(grad_to_vector(params))
            if len(self.grads) >= self.n:
                self.callback(self.grads)
                self.grads = []
                return True
            return False

    def print_grad_variance(grads):
        grads = np.stack(grads, axis=0)
        gnorm = np.linalg.norm(np.mean(grads, axis=0))
        s, v, d = np.linalg.svd(grads)
        print("|grad| = {} (STD = {})".format(gnorm, np.sqrt(v[0])))


    def __init__(self, device=None, dtype=torch.float64):
        super(VB, self).__init__(device, dtype)
        self.accept_grad = VB.GradStatsCollector()  # callable
        self.stop_criterion = VB.SmallChangeStopCriterion()  # callable 


    def sample(self, model, proposal, 
            prev_model_sampler=None, 
            prev_proposal_sampler=None, 
            initvals=None,
            nparticles=1, ignore_obs=all):
        """ Draw nparticles samples from the (proposal)->(model)
            Ignores provided observed data
        """
        assert nparticles == 1
        # Sample many particles from the proposal distribution
        proposal_sampler = Sampler(nparticles=nparticles, 
                device=self.device, dtype=self.dtype)
        if prev_proposal_sampler is not None:
            proposal_sampler._param = prev_proposal_sampler._param
        if initvals is not None:
            proposal_sampler.copy_values(initvals)
        proposal(proposal_sampler)

        # Now we have proposal parameters and posterior samples.
        # Use them to draw samples from the model
        model_sampler = Sampler(randv=proposal_sampler._randv, 
                nparticles=nparticles, 
                device=self.device, dtype=self.dtype)
        model_sampler.ignore_obs = ignore_obs  # run in generative mode
        if prev_model_sampler is not None:
            model_sampler._param = prev_model_sampler._param
        if initvals is not None:
            model_sampler.copy_values(initvals)
        
        # Sample new data
        model(model_sampler)

        return model_sampler, proposal_sampler


    def infer(self, model, proposal, 
            prev_model_sampler=None, 
            prev_proposal_sampler=None, 
            initvals=None,
            paramfilter=None,
            nparticles=1, lr=0.01, maxiter=1000, callback=None):
        """ Monte Carlo Variational Bayes
            paramfilter: name->bool - True to optimize
        """
        if paramfilter is None:
            paramfilter = lambda name: True
        self.stop_criterion = VB.SmallChangeStopCriterion()  # callable
        self.stop_criterion.lr = lr
        # Sample many particles from the proposal distribution
        proposal_sampler = Sampler(nparticles=nparticles, 
                device=self.device, dtype=self.dtype)
        if prev_proposal_sampler is not None:
            proposal_sampler._param = prev_proposal_sampler._param
            proposal_sampler._const = prev_proposal_sampler._const
        proposal(proposal_sampler)

        # Now we have proposal parameters and posterior samples.
        # Use them to estimate ELBO and gradient with MC integration
        model_sampler = Sampler(randv=proposal_sampler._randv, 
                nparticles=nparticles, 
                device=self.device, dtype=self.dtype)
        if prev_model_sampler is not None:
            model_sampler._param = prev_model_sampler._param
            model_sampler._const = prev_model_sampler._const
        model(model_sampler)

        # Set initial values if provided
        proposal_sampler.update_values(initvals)
        model_sampler.update_values(initvals)
        
        # Optimizer
        all_params = {**proposal_sampler.named_parameters(), **model_sampler.named_parameters()}
        parameters = [val for name, val in all_params.items() if paramfilter(name)]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for t in range(maxiter):
            while True:
                optimizer.zero_grad()

                # Run particles
                proposal_sampler = Sampler(param=proposal_sampler._param, 
                                           const=proposal_sampler._const, 
                                           nparticles=nparticles, 
                                           device=self.device, dtype=self.dtype)
                proposal(proposal_sampler)
                model_sampler = Sampler(randv=proposal_sampler._randv, 
                                        param=model_sampler._param, 
                                        const=model_sampler._const, 
                                        obser=model_sampler._obser,
                                        nparticles=nparticles, 
                                        device=self.device, dtype=self.dtype)
                model(model_sampler)
                
                # ELBO estimation
                ELBO = (model_sampler.log_prob() - proposal_sampler.log_prob()) / nparticles
                loss = -1.0 * ELBO
                loss.backward(retain_graph=False)
                if self.accept_grad is None or \
                        self.accept_grad == all or \
                        self.accept_grad(parameters):
                    break
            optimizer.step()
            
            if callback is not None:
                callback(t, loss.to("cpu").data.numpy(), model_sampler, proposal_sampler)
            if self.stop_criterion(parameters):
                print("STOP criterion fulfilled")
                break

        return model_sampler, proposal_sampler


    def optimize(self, model, 
            prev_model_sampler=None, prev_proposal_sampler=None, 
            initvals=None, maxiter=1000, callback=None):
        """ MAP estimation
        """
        assert self.device.type == "cpu"
        # Construct the MAP proposal
        proposal = MAPProposal(model, device=self.device, dtype=self.dtype)
        proposal_sampler = Sampler(nparticles=1, device=self.device, dtype=self.dtype)
        if prev_proposal_sampler is not None:
            proposal_sampler._param = prev_proposal_sampler._param
        proposal(proposal_sampler)
        
        # Construct the model sampler
        model_sampler = Sampler(randv=proposal_sampler._randv, nparticles=1, device=self.device, dtype=self.dtype)
        if prev_model_sampler is not None:
            model_sampler._param = prev_model_sampler._param
        model(model_sampler)

        # Set initial values if provided
        proposal_sampler.update_values(initvals)
        model_sampler.update_values(initvals)
        
        parameters = proposal_sampler.parameters() + model_sampler.parameters()
        
        t = 0
        loss = 0.0
        def f_log_prob():
            nonlocal proposal_sampler
            nonlocal model_sampler
            
            proposal_sampler = Sampler(param=proposal_sampler._param, 
                                    nparticles=1, 
                                    device=self.device, dtype=self.dtype)
            proposal(proposal_sampler)
            model_sampler = Sampler(randv=proposal_sampler._randv, 
                                    param=model_sampler._param, 
                                    obser=model_sampler._obser,
                                    nparticles=1, 
                                    device=self.device, dtype=self.dtype)
            model(model_sampler)
            
            # MAP
            log_prob = model_sampler.log_prob() + proposal_sampler.log_prob()
            return log_prob

        def f_loss():
            nonlocal loss
            loss = -f_log_prob()
            return loss
            
        def cb_proxy(xk):
            nonlocal t
            nonlocal loss
            t += 1
            if callback is not None:
                callback(t, loss.data.numpy(), model_sampler, proposal_sampler)

        # Optimizer
        if maxiter > 0:
            xopt, fopt, d = optimize_l_bfgs_scipy(f_loss, parameters, maxiter=maxiter, callback=cb_proxy)
        
        final_log_prob = f_log_prob()  # re-create computational graph!!!
        return model_sampler, proposal_sampler, final_log_prob


    def log_prob(self, model, initvals=None):
        """ Join log-probability of the model
        """
        return self.optimize(model, initvals=initvals, maxiter=0)


    def log_prob_simple(self, model, initvals):
        """ Join log-probability of the model
        """
        proposal = MAPProposal(model, device=self.device, dtype=self.dtype)
        proposal_sampler = Sampler(nparticles=1, device=self.device, dtype=self.dtype)
        proposal(proposal_sampler)

        # Use initvals
        proposal_sampler.update_values(initvals)
        proposal_sampler._distr = {}
        proposal_sampler._randv = {}
        proposal(proposal_sampler)

        model_sampler = Sampler(randv=proposal_sampler._randv, nparticles=1, device=self.device, dtype=self.dtype)
        model(model_sampler)
        model_sampler.update_values(initvals)
        model_sampler._distr = {}
        model_sampler._obser = {} 
        model(model_sampler)
        return model_sampler.log_prob()


class DeltaDistr(torch.distributions.Distribution, DistrMixin):
    arg_constraints = {'value': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, value):
        self.value = value
        if isinstance(value, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.value.size()
        super(DeltaDistr, self).__init__(batch_shape)
        

    def rsample(self, sample_shape=torch.Size([])):
        return self.value

    def log_prob(self, value):
        if torch.all(value == self.value):
            return torch.tensor(0.0)
        else:
            return torch.tensor(-1.0e-6)


class MAPProposal(object):
    """ Universal deterministic proposal for MAP estimation
    """
    def __init__(self, model, device=None, dtype=torch.float64):
        """ 
            model: stochastic process, callable
        """
        self.device = device
        self.dtype = dtype
        self._params = {}
        self._distr = {}
        
        # Sampling environment
        sampler = Sampler(nparticles=1, device=self.device, dtype=self.dtype)
        
        # Collect distributions and random values created by the model
        model(sampler)
        for name, rv in sampler._randv.items():
            if name not in sampler._obser:
                self._params[name] = sampler._randv[name]
                self._distr[name] = sampler._distr[name]

    def __call__(self, sampler):
        def get_innder_distribution(distr):
            if isinstance(distr, MaskedDistribution):
                return get_innder_distribution(distr.inner)
            else:
                return distr

        for name, value in self._params.items():
            distr = get_innder_distribution(self._distr[name])
            if isinstance(distr, Normal):
                param = sampler.param(name+".loc", value[0])
                particle_param = param[None, ...]
                sampler.sample(name, DeltaDistr(particle_param))
            elif isinstance(distr, LogNormal):
                param = sampler.param("log_"+name+".loc", value[0])
                particle_param = torch.exp(param[None, ...])
                sampler.sample(name, DeltaDistr(particle_param))
            else:
                raise NotImplementedError()

    def __repr__(self):
        sp = "\n".join(["  device: {}/{}".format(self.device, self.dtype)] +
                       ["  '{}': {} = {}".format(name, value.shape, value) for name, value in self._params.items()])
        return sp


def hessian(y, xs, allow_unused=False):
    """ Hessian of y w.r.t. xs
    """
    dys = torch.autograd.grad(y, xs, create_graph=True, allow_unused=allow_unused)
    flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
    H = []
    for dyi in flat_dy:
        Hi = torch.cat([Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, xs, 
                retain_graph=True, allow_unused=allow_unused)])
        H.append(Hi)
    H = torch.stack(H)
    return H
        

def marginal_log_p_data_Laplace(model_sampler, proposal_sampler):
    params = proposal_sampler.parameters() + model_sampler.parameters()
    log_prob = model_sampler.log_prob() + proposal_sampler.log_prob()
    h = hessian(log_prob, params)
    log_p_data = modelcomparison.marginal_log_p_data_Laplace(
            log_p_model_MAP=log_prob.data.numpy(), 
            hessian_log_MAP=h.data.numpy())
    return log_p_data


if __name__ == "__main__":
    import visdom
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(1)
    vis = visdom.Visdom()
    vis.close(win="Loss")
    
    class GaussianData(TensorContainer):
        def __init__(self):
            super(GaussianData, self).__init__()
            self.loc = 5.0
            self.scale = 2.0
            y = distr.Normal(loc=0.0, scale=1.0).sample(sample_shape=[200])
            y = y - torch.mean(y)
            y = y / torch.sqrt((torch.sum(y*y)/len(y)))
            self["y"] = self.scale * y + self.loc
            
        def model(self, sampler):
            # Priors
            loc = sampler.sample("loc", Normal(0, 10))  
            log_scale = sampler.sample("log_scale", Normal(0, 10))
            # Observed
            d = Normal(loc, torch.exp(log_scale)).expand_right([len(self["y"])]).mask(self["y"] > self.loc)
            sampler.sample("y", d, obs=self["y"])

        def model_multivar(self, sampler):
            # Priors
            loc = sampler.sample("loc", Normal(0, 10))  
            log_scale = sampler.sample("log_scale", Normal(0, 10))
            # Observed
            d = MultivariateNormal(loc[..., None], torch.exp(log_scale)[..., None, None]*torch.eye(2))
            d = d.expand_right([len(self["y"])]).mask(self["y"] > self.loc)
            sampler.sample("y", d, obs=self["y"][:, None])
            
        def proposal(self, sampler):
            # Proposed posterior for location
            loc_loc = sampler.param("loc.loc", 3.0)
            loc_log_scale = sampler.param("loc.log_scale", -1.0)
            loc = sampler.sample("loc", Normal(loc_loc, torch.exp(loc_log_scale)))
            # Proposed posterior for scale
            log_scale_loc = sampler.param("log_scale.loc", 0.0)
            log_scale_log_scale = sampler.param("log_scale.log_scale", -1.0)
            log_scale = sampler.sample("log_scale", Normal(log_scale_loc, torch.exp(log_scale_log_scale)))

    
    class OptimizationLog(object):
        def __init__(self):
            self.loss = []

        def iter_callback(self, iter, loss, model_sampler, proposal_sampler):
            self.loss.append(loss)
            
            print("Iter:", iter, "Loss:", loss)
            #print("~ Proposal:")
            #print(proposal_sampler)
            #print("~ Model:")
            #print(model_sampler)

            if iter > 20:
                vis.line(X=[iter], Y=[-loss], win="Loss", name=str(proposal_sampler._nparticles), update="append")
    
    
    gd = GaussianData()
    vb = VB()
    vb.accept_grad.callback = VB.print_grad_variance
    ELBO = {}
    if True:
        # MCVB estimation
        for devicename, dtype in (("cuda", torch.float32), ("cpu", torch.float64)):
            try: 
                device = torch.device(devicename)
                print("Using {}".format(device))
                gd.to_device(device, dtype)
                vb.to_device(device, dtype)
                ol = OptimizationLog()
                model_sampler, proposal_sampler = vb.infer(gd.model, gd.proposal, 
                        nparticles=10, lr=0.1, maxiter=2000, callback=ol.iter_callback)
                print("loc: ", proposal_sampler._param["loc.loc"])
                print("scale: ", torch.exp(proposal_sampler._param["log_scale.loc"]))
                print("~ Proposal:")
                print(proposal_sampler)
                print("~ Model:")
                print(model_sampler)
                ELBO[devicename] = -ol.loss[-1]
            except:
                pass
        
    log_p_data = None
    if True:
        # MAP estimation
        device = torch.device("cpu")
        dtype = torch.float64
        print("Using {}".format(device))
        gd.to_device(device, dtype)
        vb.to_device(device, dtype)
        ol = OptimizationLog()
        model_sampler, proposal_sampler, log_prob = vb.optimize(gd.model, maxiter=200, callback=ol.iter_callback)
        print("~ Proposal:")
        print(proposal_sampler)
        print("~ Model:")
        print(model_sampler)
        
        log_p_data = marginal_log_p_data_Laplace(proposal_sampler, model_sampler)

        
    print("ELBO:", ELBO)    
    print("Laplace log_p(D):", log_p_data)
    

