import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
import tqdm
import utils_dist as utils

class CPLVM_general(nn.Module, ABC):
    def __init__(self, p, Ks, Kf):
        super(CPLVM_general, self).__init__()
        self.p = p
        self.Ks = Ks
        self.Kf = Kf

    def _init_params(self, X, Y, nf, nb, p):
        '''
        This function initializes the parameters of the model
        :param X: (cells, genes), foreground. Needed to calculate the initial values for alpha
        :param Y: (cells, genes), background. Needed to calculate the initial values for alpha
        :param nf:
        :param nb:
        :param p:
        :param Ks:
        :param Kf:
        :return:
        '''
        # note that sigma is the standard deviation, not the variance of the distribution. Torch.rand: unif(0,1). Torch.randn: normal(0,1)
        self.qzb_mu = nn.Parameter(torch.rand(nb, self.Ks, requires_grad=True))
        self.qzb_sigma = nn.Parameter(torch.tensor(1e-4))
        self.qzf_mu = nn.Parameter(torch.rand((nf, self.Ks), requires_grad=True))
        self.qzf_sigma = nn.Parameter(torch.tensor(1e-4))
        self.qt_mu = nn.Parameter(torch.rand((nf, self.Kf), requires_grad=True))
        self.qt_sigma = nn.Parameter(torch.tensor(1e-4))
        self.qs_mu = nn.Parameter(torch.rand((self.Ks, p), requires_grad=True))
        self.qs_sigma = nn.Parameter(torch.tensor(1e-4))
        self.qw_mu = nn.Parameter(torch.rand((self.Kf, p), requires_grad=True))
        self.qw_sigma = nn.Parameter(torch.tensor(1e-4))
        # alpha: inital values of alpha is the emprical mean(logsum(X)) --> alpha for each cell is initialized to be the mean of the log of the total count for that cell.
        _, self.std_logsum_x = utils.var_total_count_per_cell(X)  # foreground cells alpha
        self.palphaf_mu = torch.log(X.float().sum(dim=1))  # (nf)
        self.qalphaf_mu = nn.Parameter(self.palphaf_mu)  # (nf)
        self.qalphaf_sigma = nn.Parameter(torch.tensor(1e-4)) # scalar
        _, self.std_logsum_y = utils.var_total_count_per_cell(Y)  # background cells alpha
        self.palphab_mu = torch.log(Y.float().sum(dim=1))  # (nb)
        self.qalphab_mu = nn.Parameter(self.palphab_mu)  # (nb)
        self.qalphab_sigma = nn.Parameter(torch.tensor(1e-4)) # scalar
        return

    @abstractmethod
    def forward(self, X, Y):
        '''
        This function takes in data of background and foreground cells, and return the parameters for the Poisson distribution
        :param X: (cells, genes), foreground
        :param Y: (cells, genes), background
        :return: parameters fro the Poisson distribution X_hat (cells, genes) , Y_hat (cells, genes)
        '''
        pass

    @abstractmethod
    def loss_function(self, X_hat, X, Y_hat, Y, theta):
        pass

    def qc_input(self, X, Y):
        '''
        This function does quality control on the input data
        :param X: (cells, genes), foreground
        :param Y: (cells, genes), background
        :return:
        '''
        assert X.shape[1] == self.p, 'Number of input genes does not match the model dimension in X'
        assert Y.shape[1] == self.p, 'Number of input genes does not match the model dimension in Y'
        # maek sure that the sum of counts in all the cells are positive, because we assume that all the cells
        # that have zero counts are removed in the preprocessing step
        assert (X.sum(dim=1) > 0).all(), 'There are cells with zero counts in X. Please remove them before fitting the model.'
        assert (Y.sum(dim=1) > 0).all(), 'There are cells with zero counts in Y. Please remove them before fitting the model.'
        return

    def _fit_VI(self, X, Y, num_epochs, learning_rate):
        '''
        This function fits the model using variational inference
        :param X: (cells, genes), foreground
        :param Y: (cells, genes), background
        :param num_epochs:
        :param batch_size:
        :param learning_rate:
        :return:
        '''
        # first we need to do quality control on the input data
        self.qc_input(X, Y)
        # initialize the parameters
        self._init_params(X, Y, X.shape[0], Y.shape[0], X.shape[1])
        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        negLogP_history = []
        kl_qp_history = []
        # we do not use the data loader here because this is variational inference (need latent variable for all samples)
        # start the training loop
        for epoch in tqdm.tqdm(range(num_epochs)):
            X_hat, Y_hat, theta = self.forward(X, Y)
            # Compute loss
            negLogP, kl_qp = self.loss_function(X_hat, X, Y_hat, Y, theta)
            negLogP_history.append(negLogP.item())
            kl_qp_history.append(kl_qp.item())
            loss = negLogP + kl_qp
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if epoch % 100 == 0:
            #     print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
        return negLogP_history, kl_qp_history


class CPLVM(CPLVM_general):
    def __init__(self, p, Ks, Kf):
        super(CPLVM, self).__init__(p, Ks, Kf)

    def _init_params(self, X, Y, nf, nb, p):
        super(CPLVM, self)._init_params(X, Y, nf, nb, p)
        # then add our CPLVM-specific parameters
        self.qdelta_mu = nn.Parameter(torch.rand((p,1), requires_grad=True)) # parameters associated with each of the genes to account for how the genes are differentially expressed in the foreground and background cells
        self.qdelta_sigma = nn.Parameter(torch.tensor(1e-4)) # scalar
        return


    def forward(self, X, Y):
        '''
        This function takes in data of background and foreground cells, and return the parameters for the Poisson distribution
        :param X: (cells, genes), foreground
        :param Y: (cells, genes), background
        :return: parameters fro the Poisson distribution X_hat (cells, genes) , Y_hat (cells, genes)
        '''
        # first, since all the parameters have the variational distribution of log normal, we do reparametrization to obtain ONE instance of the parameters. We will then use these instances to calculate the parameters for the Poisson distribution
        # generate z_b, z_f, t, s, w, alpha through reparametrization.
        # first generate epsilon from standard normal. Then, for each parameters, say Z, Z= exp(mu + sigma * epsilon)
        z_b = utils.reparametrize_log_normal(self.qzb_mu, self.qzb_sigma) # (nb, Ks)
        z_f = utils.reparametrize_log_normal(self.qzf_mu, self.qzf_sigma) # (nf, Ks)
        t = utils.reparametrize_log_normal(self.qt_mu, self.qt_sigma) # (nf, Kf)
        s = utils.reparametrize_log_normal(self.qs_mu, self.qs_sigma)  # (Ks, p)
        w = utils.reparametrize_log_normal(self.qw_mu, self.qw_sigma)  # (Kf, p)
        alpha_f = utils.reparametrize_log_normal(self.qalphaf_mu, self.qalphaf_sigma) # (nf)
        alpha_b = utils.reparametrize_log_normal(self.qalphab_mu, self.qalphab_sigma) # (nb)
        delta = utils.reparametrize_log_normal(self.qdelta_mu, self.qdelta_sigma)  # (p, 1)
        # then calculate the parameters for the Poisson distribution
        Y_hat = (alpha_b * delta).T * (z_b @ s)  # (nb, p), also not that * denotes element-wise multiplication
        X_hat = (alpha_f.view(-1,1)) * ((z_f @ s) + (t @ w))  # (nf, p)
        # calculate the log(q(theta)/p(theta)) for each of the parameters, which will be used to calculate the KL divergence
        theta = {'zb': z_b, 'zf': z_f, 't': t, 's': s, 'w': w, 'alpha_f': alpha_f, 'alpha_b': alpha_b, 'delta': delta}
        return X_hat, Y_hat, theta

    def loss_function(self, X_hat, X, Y_hat, Y, theta):
        # poisson loss: -log(P(X|X_hat))-log(P(Y|Y_hat))
        loss = utils._poisson_loss(X_hat, X) + utils._poisson_loss(Y_hat, Y)
        # and the KL divergence between the variational distribution and the prior
        kl = 0 # between varation distribution (log normal) and the prior (gamma(1,1)) for most components of theta,
        # theta=(W,S,Z_b,Z_f,T,alpha_f,alpha_b,delta)
        # note that right now, we all assume that the prior is gamma(1,1) for all the parameters (except for scale factor alpha)
        kl += utils._gammaPDF_logNormalPDF(theta['zb'], self.qzb_mu, self.qzb_sigma, theta_name='zb')
        kl += utils._gammaPDF_logNormalPDF(theta['zf'], self.qzf_mu, self.qzf_sigma, theta_name='zf')
        kl += utils._gammaPDF_logNormalPDF(theta['t'], self.qt_mu, self.qt_sigma, theta_name='t')
        kl += utils._gammaPDF_logNormalPDF(theta['s'], self.qs_mu, self.qs_sigma, theta_name='s')
        kl += utils._gammaPDF_logNormalPDF(theta['w'], self.qw_mu, self.qw_sigma, theta_name='w')
        kl += utils.kl_divergence_lognormal(mu1=self.qalphaf_mu, sigma1=self.qalphaf_sigma, mu2=self.palphaf_mu, sigma2=self.std_logsum_x)  # kl(q||p)=kl(logNormal(mu1, sigma1)||logNormal(mu2, sigma2))
        kl += utils.kl_divergence_lognormal(mu1=self.qalphab_mu, sigma1=self.qalphab_sigma, mu2=self.palphab_mu, sigma2=self.std_logsum_y)  # kl(q||p)=kl(logNormal(mu1, sigma1)||logNormal(mu2, sigma2))
        kl += utils._gammaPDF_logNormalPDF(theta['delta'], self.qdelta_mu, self.qdelta_sigma, theta_name='delta')
        return loss, kl

    def _fit_VI(self, X, Y, num_epochs, learning_rate):
        negLogP_history, kl_qp_history= super(CPLVM, self)._fit_VI(X, Y, num_epochs, learning_rate)
        return negLogP_history, kl_qp_history


class CGLVM(CPLVM_general):
    def __init__(self, p, Ks, Kf):
        super(CGLVM, self).__init__(p, Ks, Kf)
        self.p = p
        self.Ks = Ks
        self.Kf = Kf

    def _init_params(self, X, Y, nf, nb, p):
        '''
        This function initializes the parameters of the model
        :param X: (cells, genes), foreground. Needed to calculate the initial values for alpha
        :param Y: (cells, genes), background. Needed to calculate the initial values for alpha
        :param nf:
        :param nb:
        :param p:
        :param Ks:
        :param Kf:
        :return:
        '''
        super(CGLVM, self)._init_params(X, Y, nf, nb, p)
        # then add our CGLVM-specific parameters
        self.q_muf_mu = nn.Parameter(torch.randn((nf, p), requires_grad=True))
        self.q_muf_sigma = nn.Parameter(torch.randn(1, requires_grad=True))
        self.q_mub_mu = nn.Parameter(torch.randn((nb, p), requires_grad=True))
        self.q_mub_sigma = nn.Parameter(torch.randn(1, requires_grad=True))
        return

    def forward(self, X, Y):
        '''
        This function takes in data of background and foreground cells, and return the parameters for the Poisson distribution
        :param X: (cells, genes), foreground
        :param Y: (cells, genes), background
        :return: parameters fro the Poisson distribution X_hat (cells, genes) , Y_hat (cells, genes)
        '''
        s = utils.reparametrize_normal(self.qs_mu, self.qs_sigma)  # (Ks, p)
        w = utils.reparametrize_normal(self.qw_mu, self.qw_sigma)  # (Kf, p)
        z_f = utils.reparametrize_normal(self.qzf_mu, self.qzf_sigma)  # (nf, Ks)
        z_b = utils.reparametrize_normal(self.qzb_mu, self.qzb_sigma)  # (nb, Ks)
        t = utils.reparametrize_normal(self.qt_mu, self.qt_sigma)  # (nf, Kf)
        alpha_f = utils.reparametrize_log_normal(self.qalphaf_mu, self.qalphaf_sigma)  # (nf)
        alpha_b = utils.reparametrize_log_normal(self.qalphab_mu, self.qalphab_sigma)  # (nb)
        # TODO: make sure mu are all (p*1), or cells-specific for mu
        mu_f = utils.reparametrize_normal(self.q_muf_mu, self.q_muf_sigma)  # (nf, p)
        mu_b = utils.reparametrize_normal(self.q_mub_mu, self.q_mub_sigma)  # (nb, p)
        # TODO: the following line of code need debugging
        X_hat = ((z_f @ s) + (t @ w)) + mu_f + torch.log(alpha_f)  # (nf, p)
        Y_hat = (z_b @ s) + mu_b + torch.log(alpha_b)  # (nb, p)
        # note that the +1 in the two above lines are needed to avoid log(0) when alpha is zero,
        # transform X and Y as Poisson(exp(X)) and Poisson(exp(Y))
        X_hat = torch.exp(X_hat)
        Y_hat = torch.exp(Y_hat)
        return X_hat, Y_hat

    def loss_function(self, X_hat, X, Y_hat, Y):
        # loss function based on the poisson distribution
        loss = utils._poisson_loss(X_hat, X) + utils._poisson_loss(Y_hat, Y)
        # and the KL divergence between the variational distribution and the prior
        kl = 0
        kl += utils.kl_divergence_normal(mu1=self.q_muf_mu, sigma1=self.q_muf_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.q_mub_mu, sigma1=self.q_mub_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.qs_mu, sigma1=self.qs_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.qw_mu, sigma1=self.qw_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.qzf_mu, sigma1=self.qzf_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.qzb_mu, sigma1=self.qzb_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_normal(mu1=self.qt_mu, sigma1=self.qt_sigma, mu2=0, sigma2=1)
        kl += utils.kl_divergence_lognormal(mu1=self.qalphaf_mu, sigma1=self.qalphaf_sigma, mu2=self.mean_logsum_x, sigma2=self.std_logsum_x)
        kl += utils.kl_divergence_lognormal(mu1=self.qalphab_mu, sigma1=self.qalphab_sigma, mu2=self.mean_logsum_y, sigma2=self.std_logsum_y)
        return loss + kl

    def _fit_VI(self, X, Y, num_epochs, learning_rate):
        super(CPLVM, self)._fit_VI(X, Y, num_epochs, learning_rate)
