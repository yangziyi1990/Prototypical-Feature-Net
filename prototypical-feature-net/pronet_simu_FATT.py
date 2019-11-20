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
from few_shot_FATT import *
from plot_fig import *

##-----------------------------------
## Step 1. Setting initialization parameters
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
num_clusters = 8        # number of class
lr = 1e-1
weight_decay = 1e-4
figsize = 10

##-----------------------------------
## Step 2: Generating meta-training data
##-----------------------------------
means = np.array([[-5,-5], [-5,0], [0,-5], [-2,2], [2,2], [4,-4], [6,6], [-2,-9]])
means = [m*signal_dim for m in means]
sigmas = 1.0*np.ones(len(means))
x = []
y = []

for i, (mean, sigma) in enumerate(zip(means, sigmas)):
    x.append(np.random.multivariate_normal(mean, sigma*np.eye(len(mean)), size=clu_size))
    y.append(i*np.ones(clu_size))
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)


## plot of the true signal for meta_training dataset
# title = 'The true signal for meta-training dataset'
# plot_true_signal(x, y, title, figsize)

z = np.random.multivariate_normal([2.5]*noisy_dim*2, 10*np.eye(noisy_dim*2), size=len(y))
x = np.concatenate([x[:,:signal_dim],z[:,:noisy_dim],x[:,signal_dim:],z[:,noisy_dim:]],axis=1)
in_dim = x.shape[1]     # number of features

## Generate simulation data
x_train = Variable(torch.from_numpy(x).float())
y_train = Variable(torch.from_numpy(y).long())

## plot of the raw data, after adding 40-dimensional Gaussian noise
# title = 'Raw data PCA'
# plot_scatter(x_train, title=title, colors=y, size=figsize)


## Initialize parameters----------------------------
n_way = 4             # the number of classes in the dataset
n_support = 5         # the number of samples for support set
n_query = 5           # the number of samples for query set
label_balance = True
sampler = []
data_episode = {}
batch_size = n_support + n_query

## initialize with dict
num_c = list(range(y_train.max().item() + 1))
num_c_sample = len([i for i in y_train if i==y_train.max().item()])      # the number of training samples for each class
sample_idx = dict.fromkeys(num_c,  torch.zeros(num_c_sample))


##--------------------------------------------------
## Step 3: Split training dataset to support and query set
##--------------------------------------------------
if label_balance == True:
    for key in sample_idx:
        # dictionary to recode the index of samples for each class
        sample_idx[key] = [i for i, x in enumerate(y_train) if y_train[i]==key]

    sampler = ZiyiSampler2(sample_idx, batch_size=batch_size, clu_size=num_c_sample, num_clusters=num_clusters, n_way=n_way)
    # type dictionary, the label of y for each episode begin with 0
    data_episode = ZiyiDataLoader3(x_train, y_train, sampler, n_support, n_query)



##--------------------------------------------------
## Step 4: Generatiing meta-testing data
##--------------------------------------------------
signal_dim_test = 1
noisy_dim_test = 40
clu_size_test = 1000          # number of sample for each class
num_clusters_test = 4        # number of class
test_portions = 0.01          # the proportion of training samples



means_test = np.array([[0,0], [1,0], [0,1], [1,1]]) * 5
means_test = [m*signal_dim for m in means_test]
sigmas_test = 1.0*np.ones(len(means_test))
x_test = []
y_test = []

for i, (mean, sigma) in enumerate(zip(means_test, sigmas_test)):
    x_test.append(np.random.multivariate_normal(mean, sigma*np.eye(len(mean)), size=clu_size_test))
    y_test.append(i*np.ones(clu_size_test))
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

## plot of the true signal for meta_testing dataset
# title = 'The true signal for meta-testing dataset'
# plot_true_signal(x_test, y_test, title, figsize)

z_test = np.random.multivariate_normal([2.5]*noisy_dim*2, 10*np.eye(noisy_dim*2), size=len(y_test))
x_test = np.concatenate([x_test[:,:signal_dim],z_test[:,:noisy_dim],x_test[:,signal_dim:],z_test[:,noisy_dim:]],axis=1)
in_dim_test = x_test.shape[1]     # number of features

## Generate simulation data
x_test = Variable(torch.from_numpy(x_test).float())
y_test = Variable(torch.from_numpy(y_test).long())



## plot of the raw data, after adding 40-dimensional Gaussian noise
# title = 'Raw data PCA'
# plot_scatter(x_test, title=title, colors=y_test, size=figsize)


## Initialize parameters for meta-testing
label_balance1 = True
sampler1 = []
data_episode1 = {}
batch_size1 = len(y_test)//num_clusters_test
n_support1 = int(test_portions * batch_size1)
n_query1 = int((1 - test_portions) * batch_size1)
n_way1 = num_clusters_test

## Initialize with dict
num_c1 = list(range(y_test.max().item() + 1))
num_c_sample1 = len([i for i in y_test if i==y_test.max().item()-1])
sample_idx1 = dict.fromkeys(num_c1,  torch.zeros(num_c_sample1))

##--------------------------------------------------
## split training dataset with support and query set
##--------------------------------------------------
if label_balance == True:
    for key in sample_idx1:
        sample_idx1[key] = [i for i, x in enumerate(y_test) if y_test[i]==key]

    sampler1 = ZiyiSampler1(sample_idx1, batch_size=batch_size1, clu_size=num_c_sample1)
    data_episode1 = ZiyiDataLoader2(x_test, y_test, sampler1, n_support1, n_query1)




##--------------------------------------------------
## Step 5: Meta-training starting
##--------------------------------------------------
kwargs_dict = {'feature_dim': in_dim}
model = load_Protonet_FATT(**kwargs_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

## Plot of the feature weights learned by the proposed method
# feature_last_num = len(feature_iter)-1
# title = 'Feature weights after training the model'
# plot_feature_weight_linear(feature = feature_iter[feature_last_num],
#                            title = title,
#                            signal_dim = signal_dim,
#                            noisy_dim = noisy_dim,
#                            figsize = figsize)

##--------------------------------------------------
## Step 6: Starting meta-testing
##--------------------------------------------------
sample_test = data_episode1[0]
model.eval()
optimizer.zero_grad()
loss, output = model.loss(sample_test)
loss.backward()
optimizer.step()
addtwodimdict(feature_iter, iter, key, output['fw'])  # feature_iter[epoch_num][step_num]
print('Meta-testing', '| Loss:', output['loss'], '| Accuracy:', output['acc'])


## Plot of the feature weights learned by the proposed method from meta-testing
# feature_metatest = output['fw'].data.numpy()
# title = 'Feature weights after training the model'
# plot_feature_weight_linear_metatest(feature = feature_metatest,
#                                     title = title,
#                                     signal_dim = signal_dim,
#                                     noisy_dim = noisy_dim,
#                                     figsize = figsize)

