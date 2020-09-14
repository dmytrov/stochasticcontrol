"""

  Author: Dmytro Velychko, Philipps University of Marburg
  velychko@staff.uni-marburg.de
"""

from mcvariationalbayes.mcvb import *

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
            d = Normal(loc, torch.exp(log_scale)).expand_right([len(self["y"])])
            sampler.sample("y", d, obs=self["y"])
            
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
                        nparticles=100, lr=0.1, maxiter=2000, callback=ol.iter_callback)
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
    

