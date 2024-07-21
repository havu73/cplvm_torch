import torch
import numpy as np
from torch.distributions import Gamma, LogNormal
from torch.nn import functional as F

def _poisson_loss(X_hat, X):
    """
    Poisson loss function: negative log likelihood of X ~ Poisson(X_hat).
    Parameters:
    - x: tensor with the actual counts (X).
    - x_hat: tensor with the predicted rates (lambda) for the Poisson distribution.
    Returns:
     negative log likelihood of the Poisson distribution.
    """
    loss = -torch.sum(X * torch.log(X_hat) - X_hat)  # we actually ignore the constant term (log(x!)) here cuz it doesn't affect the optimization
    return loss

def convert_to_tensor(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data

def var_total_count_per_cell(X):
    """
    Calculate the total count per cell. Then, calculate the mean and variance of the log(total count/cell), across all the cells.
    :param X: torch tensor of shape (n_cells, n_genes).
    :return: mean, std of the log(total count/cell) (acosss all the cells).
    """
    sum_x = X.sum(dim=1)  # (n_cells,)
    logsum_x = torch.log(sum_x)  # (n_cells,)
    # now get the mean and the standarad deviation of the logsum_x
    return torch.mean(logsum_x), torch.std(logsum_x)

def reparametrize_log_normal(mu, sigma):
    """
    Reparametrize the log-normal distribution.
    :param mu: Mean of the log-normal distribution.
    :param sigma: Standard deviation of the log-normal distribution. Can be just a scalar or the same shape as mu.
    :return: A sample from the log-normal distribution.
    """
    # check if sigma is a tensor of the same shape as mu or a scalar tensor
    if sigma.shape == mu.shape:
        pass
    elif sigma.shape == torch.Size([]):
        pass
    elif sigma.shape == torch.Size([1]):
        pass
    else:
        raise ValueError('sigma must be either a scalar or a tensor of the same shape as mu')
    epsilon = torch.randn(mu.shape)  # from standard normal distribution
    return torch.exp(mu + F.softplus(sigma) * epsilon)

def reparametrize_normal(mu, sigma):
    """
    Reparametrize the normal distribution.
    :param mu: Mean of the normal distribution.
    :param sigma: Standard deviation of the normal distribution. Can be just a scalar or the same shape as mu.
    :return: A sample from the normal distribution.
    """
    # check if sigma is a tensor of the same shape as mu or a scalar tensor
    if sigma.shape == mu.shape:
        pass
    elif sigma.shape == torch.Size([]):
        pass
    elif sigma.shape == torch.Size([1]):
        pass
    else:
        raise ValueError('sigma must be either a scalar or a tensor of the same shape as mu')
    epsilon = torch.randn(mu.shape)  # from standard normal distribution
    return mu + F.softplus(sigma) * epsilon

def _gammaPDF_logNormalPDF(theta, mu, sigma, alpha=1, beta=1, theta_name='theta'):
    '''
    It is not easy to calculate the exact KL divergence between the gamma and log normal distribution. So, we will approximate it by just calculating the log(q(theta)/p(theta)) where q=LogNormal(mu, sigma) and p=Gamma(alpha, beta), given only one value of theta
    :param mu1:
    :param mu2:
    :param sigma1:
    :param sigma2:
    :return:
    '''
    q = LogNormal(mu, F.softplus(sigma))
    p = Gamma(alpha, beta)
    # assert that every values in theta is positive
    assert (theta > 0).all(), 'All the values in {} must be positive'.format(theta_name)
    log_qOverP = (q.log_prob(theta)) - (p.log_prob(theta))  # same shape as theta
    # return the average of the log(q/p) over all the elements in theta
    return log_qOverP.mean()

def kl_divergence_lognormal(mu1, sigma1, mu2, sigma2):
    """
    Compute the Kullback-Leibler divergence between two LogNormal distributions.
    :param mu1: Mean of the first distribution.
    :param sigma1: Standard deviation of the first distribution.
    :param mu2: Mean of the second distribution.
    :param sigma2: Standard deviation of the second distribution.
    :return: The KL divergence between the two distributions.
    """
    sigma1 = F.softplus(sigma1)
    sigma2 = F.softplus(sigma2)
    assert sigma1.shape == sigma2.shape, 'sigma1 and sigma2 must have the same shape'
    assert mu1.shape == mu2.shape, 'mu1 and mu2 must have the same shape'
    kl_div = torch.log(sigma2 / sigma1) + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2 * sigma2.pow(2)) - 0.5
    return kl_div.mean()

def kl_divergence_normal(mu1, sigma1, mu2=0, sigma2=1):
    """
    Compute the Kullback-Leibler divergence between two normal distributions with diagonal covariance matrices.
    :param mu1: Mean of the first distribution.
    :param mu2: Mean of the second distribution.
    :param sigma1: Diagonal entries of the covariance matrix of the first distribution.
    :param sigma2: Diagonal entries of the covariance matrix of the second distribution.
    :return: The KL divergence between the two distributions.
    """
    sigma1 = F.softplus(sigma1)
    sigma2 = F.softplus(sigma2)
    assert sigma1.shape == sigma2.shape, 'sigma1 and sigma2 must have the same shape'
    assert mu1.shape == mu2.shape, 'mu1 and mu2 must have the same shape'
    kl_div = 0.5 * ((mu2 - mu1).pow(2) / sigma1 + sigma1 / sigma2 - 1 - torch.log(sigma1 / sigma2))
    return kl_div.sum()
