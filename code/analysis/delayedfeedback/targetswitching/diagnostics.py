"""  A collection of different diagnostic routines
"""
import numpy as np
import torch
import analysis.variationalbayes.mcvb as mcvb
import analysis.delayedfeedback.targetswitching.model as tsm



def sigmoid_logprob_profile(model, initvals, i, sigma_as=None, x_is=None):
    """ Full model joint distribution w.r.t. varying sigmoid parameters.
        sigma_as: list of sigmoid slopes
        x_is:  list of intercept points
    """
    if sigma_as is None:
        sigma_as = np.logspace(0.1, 2.0, 20)
    if x_is is None:
        x_is = np.linspace(0.0, 1.0, 50)
  
    a_list = []
    a_original = initvals["log_sigm_a.loc"].clone()
    for sigma_a_i in sigma_as:
        initvals["log_sigm_a.loc"].data[i] = np.log(sigma_a_i)
        
        b_list = []
        b_original = initvals["sigm_b.loc"].clone()
        for x_i in x_is:
            initvals["sigm_b.loc"].data[i] = tsm.sigmoid_inflection_a_to_b(x_i, sigma_a_i)
            log_prob = mcvb.VB(dtype=initvals.dtype).log_prob_simple(model, initvals=initvals).detach()
            b_list.append(log_prob)
            print("sigma_a_i, x_i", sigma_a_i, x_i, log_prob)
            initvals["sigm_b.loc"] = b_original.clone()
            
        a_list.append(torch.stack(b_list))
        initvals["log_sigm_a.loc"] = a_original.clone()
        
    return torch.stack(a_list).detach().numpy()
        
        
def sigmoid_posterior_density(proposal, initvals, i, sigma_as=None, x_is=None):
    """ Posterior distribution of sigmoid parameters
        sigma_as: list of sigmoid slopes
        x_is:  list of intercept points
    """ 
    if sigma_as is None:
        sigma_as = np.logspace(0.1, 2.0, 20)
    if x_is is None:
        x_is = np.linspace(0.0, 1.0, 50)

    proposal_sampler = mcvb.Sampler(device=initvals.device, dtype=initvals.dtype)
    proposal_sampler.copy_values(initvals)
    proposal(proposal_sampler)
    
    a_list = []
    a_original = proposal_sampler._randv["sigm_a"].clone()
    for sigma_a_i in sigma_as:
        proposal_sampler._randv["sigm_a"].data[0, i] = sigma_a_i
        
        b_list = []
        b_original = proposal_sampler._randv["sigm_b"].clone()
        for x_i in x_is:
            sigm_b_i = tsm.sigmoid_inflection_a_to_b(x_i, sigma_a_i)
            proposal_sampler._randv["sigm_b"].data[0, i] = sigm_b_i
            log_prob = proposal_sampler.log_prob()
            b_list.append(log_prob)
            print("sigma_a_i, x_i", sigma_a_i, x_i, log_prob)
            proposal_sampler._randv["sigm_b"] = b_original.clone()
            
        a_list.append(torch.stack(b_list))
        proposal_sampler._randv["sigm_a"] = a_original.clone()
        
    return torch.stack(a_list).detach().numpy()


def _sigmoid_posterior_density(proposal, initvals, i, sigma_as=None, x_is=None):
    """ Posterior distribution of sigmoid parameters
        sigma_as: list of sigmoid slopes
        x_is:  list of intercept points
    """ 
    import matplotlib.pyplot as plt
    import numpy as np    
    
    sigma_as = np.logspace(0.1, 2.0, 200)
    x_is = np.linspace(0.0, 1.0, 50)

    proposal_sampler = mcvb.Sampler(device=initvals.device, dtype=initvals.dtype)
    proposal_sampler.copy_values(initvals)
    proposal(proposal_sampler)
    
    a_list = []
    a_original = proposal_sampler._randv["sigm_a"].clone()
    for sigma_a_i in sigma_as:
        proposal_sampler._randv["sigm_a"].data[0, i] = sigma_a_i
        log_prob = proposal_sampler.log_prob()
        a_list.append(log_prob)
        proposal_sampler._randv["sigm_a"] = a_original.clone()
    a_list = torch.stack(a_list).detach().numpy()
    print(a_list)
    plt.close()
    plt.figure()
    plt.plot(sigma_as, np.exp(a_list))
    plt.savefig("out/sigma_a.pdf")

    b_list = []
    b_original = proposal_sampler._randv["sigm_b"].clone()
    sigma_bs = np.linspace(0.0, 20.0, 200)
    for sigma_b_i in sigma_bs:
        proposal_sampler._randv["sigm_b"].data[0, i] = sigma_b_i
        log_prob = proposal_sampler.log_prob()
        b_list.append(log_prob)
        proposal_sampler._randv["sigm_b"] = b_original.clone()
    b_list = torch.stack(b_list).detach().numpy()
    print(b_list)
    plt.close()
    plt.figure()
    plt.plot(sigma_bs, np.exp(b_list))
    plt.savefig("out/sigma_b.pdf")


  