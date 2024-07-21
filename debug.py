import os
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # to shut up some stupid messages
import pandas as pd
import numpy as np
import cplvm as code
import utils_dist as utils
# Now define the model
import torch
from torch import nn
from torch.nn import functional as F
import torch
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


Y = pd.read_csv("./data/toy_background.csv", header=None).values
X = pd.read_csv("./data/toy_foreground.csv", header=None).values
nf, nb = X.shape[0], Y.shape[0]
assert X.shape[1] == Y.shape[1]
p = X.shape[1]

Ks= 1
Kf = 1
p = 2

X = utils.convert_to_tensor(X)
Y = utils.convert_to_tensor(Y)
# Now let's test the model
# now apply the model to the sample data
model = code.CPLVM(p, Ks, Kf)
# model.qc_input(X,Y)
# model._init_params(X, Y, X.shape[0], Y.shape[0], X.shape[1])
# model.forward(X, Y)
negLogP_history, kl_qp_history = model._fit_VI(X, Y, 10000, 0.01)
X_hat, Y_hat, theta = model.forward(X, Y)
print('Done training')
