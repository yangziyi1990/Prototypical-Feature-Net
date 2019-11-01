import numpy as np
import math
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, BatchSampler
from itertools import chain


from pronet_function import *
##-----------------------------------
## Setting initialization parameters
##-----------------------------------

# use_gpu = True
# if torch.cuda.is_available() and use_gpu:
#   device = torch.device('cuda')
# else:
#   device = torch.device('cpu')

seed = 0
signal_dim = 1
noisy_dim = 20
clu_size = 100          # number of sample for each class
num_clusters = 4        # number of class
hidden_dims = [100]     # a hidden layer with 100 hidden units
num_iter = 100          # number of iteration
batch_size = 100        # batch? where use it?
lr = 1e-1
weight_decay = 1e-4
train_portions = 0.7   # the proportion of training samples
proportions = [train_portions] * num_clusters

##-----------------------------------
## Step1: Generating simulation data
##-----------------------------------
means = np.array([[0,0], [1,0], [0,1], [1,1]]) * 5
means = [m*signal_dim for m in means]
sigmas = 1.0*np.ones(len(means))
x = []
y = []

for i, (mean, sigma) in enumerate(zip(means, sigmas)):
    x.append(np.random.multivariate_normal(mean, sigma*np.eye(len(mean)), size=clu_size))
    y.append(i*np.ones(clu_size))
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

z = np.random.multivariate_normal([2.5]*noisy_dim*2, 10*np.eye(noisy_dim*2), size=len(y))
x = np.concatenate([x[:,:signal_dim],z[:,:noisy_dim],x[:,signal_dim:],z[:,noisy_dim:]],axis=1)
in_dim = x.shape[1]     # number of features

## Generate simulation data
x_var = Variable(torch.from_numpy(x).float())
y_var = Variable(torch.from_numpy(y).long())

## Split dataset
x_train, y_train, x_test, y_test, train_idx, test_idx = split_data(x_var, y_var, proportions=proportions, seed=seed)


## Initialize parameters
n_way = num_clusters  # the number of classes in the dataset
n_support = 5         # the number of samples for support set
n_query = 5           # the number of samples for query set
n_episodes = 100
label_balance = True
sampler = []
data_episode = {}
batch_size = n_support + n_query


## initialize with dict
num_c = list(range(y_train.max().item() + 1))
num_c_sample = len([i for i in y_train if y_train[i]==y_train.max().item()])  # the number of training samples for each class
sample_idx = dict.fromkeys(num_c,  torch.zeros(num_c_sample))

##--------------------------------------------------
## split training dataset with support and query set
##--------------------------------------------------
if label_balance == True:
    for key in sample_idx:
        sample_idx[key] = [i for i, x in enumerate(y_train) if y_train[i]==key]  # dictionary to recode the index of samples for each class

    sampler = ZiyiSampler1(sample_idx, batch_size=batch_size, clu_size=num_c_sample)
    data_episode = ZiyiDataLoader2(x_train, y_train, sampler, n_support, n_query)   # type: dictionary
    # print(data_episode[0]['xs'][0])
    # print(data_episode[0]['xs_class'][0])

    # i = 1
    # for key in data_episode:
    #     if i == 1:
    #         print(data_episode[key])
    #         i = i+1

class Pronet(nn.Module):
    def __init__(self, encoder, feature_dim):
        super(Pronet, self).__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.Tensor(self.feature_dim))
        self.weight.data.uniform_(0, 1)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     stdv = 1./ math.sqrt(self.weight.size(0))
    #     self.weight.data.uniform_(-stdv, stdv)

    def loss(self, sample):


    def forward(self, x):  # x or sample?
        # out = nn.functional.linear(x, self.weight)  # x is input data (N*P)


        feature_weight = nn.functional.softmax(self.weight, dim=0)
        x = x * feature_weight







##--------------------------------------------------
## split training dataset with support set
##--------------------------------------------------
# if label_balance == True:
#     for key in sample_idx:
#         sample_idx[key] = [i for i, x in enumerate(y_train) if y_train[i]==key]  # dictionary to recode the index of samples for each class
#
#     sampler = ZiyiSampler1(sample_idx, batch_size=5, clu_size=num_c_sample)
#     data_episode = ZiyiDataLoader1(x_train, y_train, sampler)   # type: dictionary
#
#     print(data_episode[0]['xs'][0])
#     print(data_episode[0]['class'][0])




##--------------------------------------------------
# bs = 5
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size = bs)
#
# test_ds = TensorDataset(x_test, y_test)
# test_dl = DataLoader(test_ds, batch_size = bs)
