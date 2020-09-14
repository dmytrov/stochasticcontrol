import numpy as np

def Gaussian_normalizing_z(sigma):
    """ Normalizing coefficient for Gaussian distribution
        sigma is convariance matrix
    """
    d = len(sigma)
    det = np.linalg.det(sigma)
    return np.sqrt((2.0 * np.pi)**d * det)


def Gaussian_log_normalizing_z(sigma):
    """ Log-normalizing coefficient for Gaussian distribution
        sigma is convariance matrix
    """
    d = len(sigma)
    sig, logdet = np.linalg.slogdet(sigma)
    return 0.5 * (np.log(2.0 * np.pi) * d + logdet)


def remove_zero_diagonal(m, etha = 1.0e-8):
    """ Remove every i-th row and column if m[i, i] == 0 
    """
    v = np.abs(np.diag(m)) > etha
    return m[v, :][:, v]


def marginal_p_data_Laplace(p_model_MAP, hessian_MAP):
    """ Compute p(D) via laplace approximation
    """
    hessian_MAP = remove_zero_diagonal(hessian_MAP)
    sigma = - p_model_MAP * np.linalg.inv(hessian_MAP)
    return p_model_MAP * Gaussian_normalizing_z(sigma)


def marginal_log_p_data_Laplace(log_p_model_MAP, hessian_log_MAP):
    """ Compute log(p(D)) via laplace approximation
    """
    hessian_log_MAP = remove_zero_diagonal(hessian_log_MAP)
    sigma = - np.linalg.inv(hessian_log_MAP)
    return log_p_model_MAP + Gaussian_log_normalizing_z(sigma)


if __name__ == "__main__":
    m = np.array([[1, 0, 2],
                    [0, 0, 0],
                    [2, 0, 3]])
    m = remove_zero_diagonal(m)
    assert np.allclose(m, [[1, 2], [2, 3]])
    
    import torch
    torch.set_default_tensor_type(torch.DoubleTensor)

    loc = torch.tensor([0.1, 1.0])
    covar = torch.tensor([[2.1, 1], [1, 5.1]])
    mn = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covar)
    
    scale = 3.4
    p_model_MAP = scale * torch.exp(mn.log_prob(loc))
    hessian_MAP = - p_model_MAP * torch.inverse(covar)
    p_D = marginal_p_data_Laplace(p_model_MAP.data.numpy(), hessian_MAP.data.numpy())
    assert np.allclose(p_D, scale)
    
    log_p_model_MAP = torch.log(p_model_MAP)
    hessian_log_MAP = - torch.inverse(covar)
    log_p_D = marginal_log_p_data_Laplace(log_p_model_MAP.data.numpy(), hessian_log_MAP.data.numpy())
    p_D = np.exp(log_p_D)
    assert np.allclose(p_D, scale)
    
    