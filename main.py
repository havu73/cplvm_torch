import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import torch.distributions as dist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Multinomial

np.random.seed(0)
def generate_data(num_samples, pi, ground_truth_mu_list, sigma, sigma_0, K):
    """
    Generate dataset based on the specified model.

    :param num_samples: Number of samples to generate.
    :param pi: Vector of size K with the probabilities for the multinomial distribution.
    :param ground_truth_mu_list: ground-truth means for each group
    :param sigma: Standard deviation for the normal distribution of observed data.
    :param sigma_0: Standard deviation for the prior normal distribution of each group.
    :param K: Number of groups.
    :return: Generated dataset.
    """
    # Generate group memberships Z_i
    Z = np.random.choice(K, size=num_samples, p=pi)
    # Generate observed data X_i
    X = np.array([np.random.multivariate_normal(ground_truth_mu_list[Z[i]], np.diag([sigma**2, sigma**2])) for i in range(num_samples)])
    return X, Z

# Example usage
num_samples = 300
K = 3  # Number of groups
input_dim = 2  # Dimension of the observed data
pi = [0.3, 0.4, 0.3]  # Probabilities for each group
nu_k = np.array([[0,0], [3, -3], [-3,3]])  # Prior means for each group (for each dimension)
nu_k = np.array(nu_k)
ground_truth_mu_list = [(0.5,0.5), (2.5,-1.5), (-2.5,1.5)]
sigma = 1  # Standard deviation for observed data
sigma_0 = 1  # Standard deviation for group means

X, Z = generate_data(num_samples, pi, ground_truth_mu_list, sigma, sigma_0, K)


# Variational parameters (to be optimized)
mu_loc = torch.randn(nu_k.shape, requires_grad=True)
mu_scale = torch.ones(input_dim, requires_grad=False)
log_pi = torch.zeros(K, requires_grad=True)  # Log probabilities for stability

# change some parameters to tensor so that it;s easier to work with
nu_k = torch.tensor(nu_k, dtype=torch.float32)
tensor_X = torch.tensor(X, dtype=torch.float32)
# Define the optimization
optimizer = torch.optim.Adam([mu_loc, mu_scale, log_pi], lr=0.01)
num_steps = 1000

for step in range(num_steps):
    optimizer.zero_grad()
    # ELBO calculation
    # 1. Log likelihood under the model
    mixture_dist = dist.Categorical(logits=log_pi)
    component_dists = MultivariateNormal(mu_loc, scale_tril=torch.diag(mu_scale))
    likelihood = dist.MixtureSameFamily(mixture_dist, component_dists)
    log_likelihood = likelihood.log_prob(tensor_X).sum()
    # 2. KL divergence between variational distribution and prior
    covariance_matrices = torch.diag_embed(torch.full((len(nu_k), input_dim), sigma_0, dtype=torch.float32))
    prior = MultivariateNormal(nu_k, scale_tril=covariance_matrices)
    variational_dist = MultivariateNormal(mu_loc, scale_tril=torch.diag(mu_scale))
    kl_divergence = dist.kl_divergence(variational_dist, prior).sum() + dist.kl_divergence(mixture_dist, dist.Categorical(probs=torch.tensor(pi, dtype=float))).sum()
    # 3. Negative ELBO (since we're minimizing)
    elbo = log_likelihood - kl_divergence
    loss = -elbo
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item()}")

print("Variational means:", mu_loc)
print("Variational standard deviations:", mu_scale)
print("Variational log pi:", log_pi)
print('Pi:', torch.softmax(log_pi, dim=0))

mixture_dist = dist.Categorical(logits=log_pi)  # size #(K)
component_dists = MultivariateNormal(mu_loc, scale_tril=torch.diag(mu_scale))  # size #(K, input_dim)
X_expanded = tensor_X.unsqueeze(1).repeat(1,K,1)  # size #(num_sample, K, input_dim), where each K's X_i is the same
log_prob_components = component_dists.log_prob(X_expanded)  # size #(n_sample, K)
log_prob_components = log_prob_components + mixture_dist.logits  # size #(n_sample, K)
post_Z = log_prob_components.softmax(dim=1)  # size #(n_sample, K)
# log_prob_components = log_prob_components +