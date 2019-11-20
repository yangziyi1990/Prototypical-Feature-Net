import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import itertools
from itertools import chain
# from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, BatchSampler

from data_pre import *
from utils import euclidean_dist
from few_shot import *
from plot_fig import *


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
noisy_dim = 40
clu_size = 100          # number of sample for each class
num_clusters = 4        # number of class
hidden_dims = [100]     # a hidden layer with 100 hidden units
num_iter = 100          # number of iteration
batch_size = 100        # batch? where use it?
lr = 1e-1
weight_decay = 1e-4
train_portions = 0.5   # the proportion of training samples
proportions = [train_portions] * num_clusters
figsize = 10

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

    ## Test part start_______________________________________
    # i = 1
    # for key in data_episode:
    #     if i == 1:
    #         print(type(data_episode[key]['xs']))
    #         print(type(data_episode[key]['xs_class']))
    #         print(data_episode[key]['xs'])
    #         print(data_episode[key]['xs_class'])
    #         print(data_episode[key]['xs_class'].max().item()+1)
    #         print(data_episode[key]['xs'].size(0))
    #         n_class = data_episode[key]['xs_class'].max().item() + 1
    #         n_support = int(data_episode[key]['xs'].size(0)/n_class)
    #         n_query = int(data_episode[key]['xq'].size(0)/n_class)
    #         print(n_class)
    #         print(n_support)
    #         print(n_query)
    #         xs = data_episode[key]['xs']
    #         xq = data_episode[key]['xq']
    #         target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    #         target_inds = Variable(target_inds, requires_grad=False)
    #
    #         x = torch.cat([xs, xq], 0)
    #         z = x
    #         z_dim = z.size(-1)
    #         z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
    #
    #         zq = z[n_class*n_support:]
    #         dists = euclidean_dist(zq, z_proto)
    #
    #         log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
    #         loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    #
    #         print(loss_val)
    #         _, y_hat = log_p_y.max(2)
    #         acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    #         print(acc_val.item())
    #
    #         i = i+1
    ## Test part end_______________________________________________

kwargs_dict = {'feature_dim': in_dim}
model = load_Protonet(**kwargs_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# print(model)
# print(optimizer)

Epoch = 10
feature_iter = {}

iter = 0
for epoch in range(Epoch):
    step = 0
    for key in data_episode:
        sample = data_episode[key]
        model.train()
        optimizer.zero_grad()
        loss, output = model.loss(sample)
        loss.backward()
        optimizer.step()

        addtwodimdict(feature_iter, iter, key, output['fw'])         # feature_iter[epoch_num][step_num]
        print('Epoch:', epoch, '| Steps:', step, '| Loss:', output['loss'], '| Accuracy:', output['acc'])
        # print('Feature_weight:', output['fw'])

        step += 1
    iter += 1


feature_last_num = len(feature_iter)-1
title = 'Feature weights after training the model'
plot_feature_weight_linear(feature = feature_iter[feature_last_num],
                           title = title,
                           signal_dim = signal_dim,
                           noisy_dim = noisy_dim,
                           figsize = figsize)




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
#     # print(data_episode[0]['xs'][0])
#     # print(data_episode[0]['class'][0])
#
#     i = 1
#     for key in data_episode:
#         if i == 1:
#             print(data_episode[key]['xs'])
#             print(type(data_episode[key]['xs']))
#             print(data_episode[key]['class'])
#             print(type(data_episode[key]['class']))
#             i = i+1




##--------------------------------------------------
# bs = 5
# train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size = bs)
#
# test_ds = TensorDataset(x_test, y_test)
# test_dl = DataLoader(test_ds, batch_size = bs)
