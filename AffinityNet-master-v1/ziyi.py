import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
#import torch.nn as nn
from torch.autograd import Variable
# from AffnityNet_function import *
from utils.test_graph_attention import *


use_gpu = True
if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


signal_dim = 1            # feature size / 2
clu_size = 1000           # sample size
noisy_dim = 20
train_portions = [0.01]   # Participation in training


means = np.array([[0,0],[0,1],[1,0],[1,1]]) * 5
num_clusters = len(means)
means = [i for i in means]    # the mean of 2-dimensional data
sigmas = 1.0*np.ones(len(means))         # the sigmas of 2-dimensional data


x = []
y = []
for i, (mean, sigma) in enumerate(zip(means, sigmas)):
    x.append(np.random.multivariate_normal(mean, sigma*np.eye(len(mean)), size=clu_size))  # 生成多元正态分布矩阵
    y.append(i*np.ones(clu_size))

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)


## Plots of the true signal
# title = 'The distribution for ture signal'
# plt.figure(figsize=(8,8))
# plt.scatter(x[:,0], x[:,1], c=y)
# plt.title(title)
# plt.show()
# plt.close()


z = np.random.multivariate_normal([2.5]*noisy_dim*2, 10*np.eye(noisy_dim*2), size=len(y))
x = np.concatenate([x[:,:signal_dim], z[:,:noisy_dim], x[:,signal_dim:],z[:,noisy_dim:]],axis=1)
in_dim = x.shape[1]      # in_dim: number of features, x.shape=(4000,42)

x_var = Variable(torch.from_numpy(x).float())  # variable 可求导
y_var = Variable(torch.from_numpy(y).long())
num_cls = num_clusters


seed = 0
proportions = [train_portions] * num_cls

x_train, y_train, x_test, y_test, train_idx, test_idx = split_data(
        x_var, y_var, proportions=proportions, seed=seed)

