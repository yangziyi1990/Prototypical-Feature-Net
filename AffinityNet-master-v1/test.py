import sys
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import spectral_clustering
from sklearn.svm import SVC
import os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# requires pytorch > 0.3.1
try:
  import torch
  import torch.nn as nn
  from torch.autograd import Variable
except ModuleNotFoundError:
  pass

# import the code, add path first
# for example
solver_path = 'I:/AffinityNet'
if os.path.exists(solver_path):
  sys.path.append(solver_path)


from utils.solver import Solver
from utils.graph_attention import *
from utils.test_graph_attention import *

use_gpu = True
if torch.cuda.is_available() and use_gpu:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

seed = 0
signal_dim = 1
noisy_dim = 20
clu_size = 1000   # number of sample for each class
num_clusters = 4  # number of class
hidden_dims = [100]   # a hidden layer with 100 hidden units
num_iter = 100    # number of iteration
batch_size = 100  # batch? where use it?
lr = 1e-1
weight_decay = 1e-4
root = '.'
save_folder_prefix = '{0}/data/simulation/knn_pooling_toy/seed{1}'.format(root, seed)
# if not os.path.exists(save_folder_prefix):
#   os.makedirs(save_folder_prefix)
save_fig = False
figsize = 10

means = np.array([[0,0], [1,0], [0,1], [1,1]]) * 5
assert num_clusters == len(means)
means = [m*signal_dim for m in means]
sigmas = 1.0*np.ones(len(means))
x = []
y = []
for i, (mean, sigma) in enumerate(zip(means, sigmas)):
    x.append(np.random.multivariate_normal(mean, sigma*np.eye(len(mean)), size=clu_size))
    y.append(i*np.ones(clu_size))
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

title = 'input'
plt.figure(figsize=(figsize, figsize))
plt.scatter(x[:,0], x[:,1], c=y)
save_folder = save_folder_prefix
if save_fig:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig('{0}/{1}.png'.format(save_folder,title), bbox_inches='tight', dpi=200)
else:
    plt.title(title)
    plt.show()
plt.close()

# in_dim = x.shape[1]
# x_var = Variable(torch.from_numpy(x).float())
# y_var = Variable(torch.from_numpy(y).long())
# num_cls = y_var.data.max().item()+1
#
# model = MultiviewAttention(in_dim=in_dim, hidden_dims=[10, 10, 10], k=50, graph=None,
#                            out_indices=None,
#                            feature_subset=None, kernel='gaussian', nonlinearity_1=None,
#                            nonlinearity_2=None, use_previous_graph=True,
#                            group_index=None, merge=None,
#                            merge_type='affine', reset_graph_every_forward=False,
#                            no_feature_transformation=False, rescale=True, merge_dim=2,
#                            layer_norm=False)
#
# features = FeatureExtractor(model.layers, selected_layers=range(len(model.layers)))
# print(features)
# print(x_var)
# features(x_var.cuda())


z = np.random.multivariate_normal([2.5]*noisy_dim*2, 10*np.eye(noisy_dim*2), size=len(y))
x = np.concatenate([x[:,:signal_dim],z[:,:noisy_dim],x[:,signal_dim:],z[:,noisy_dim:]],axis=1)
in_dim = x.shape[1]    # number of features

def compare_classifiers(X_train, X_test, y_train, y_test, names, classifiers, res=None):
    def eval_classifiers(X_test, y_test):
        acc_test = []
        nmi_test = []
        f1_score_test = []
        confusion_mat_test = []
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
            nmi = sklearn.metrics.adjusted_mutual_info_score(labels_true=y_test, labels_pred=y_pred)
            confusion_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
            f1_score = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
            print('{0}: acc={1:.2f}, nmi={2:.2f}, f1={3:.2f}, confusion_mat:\n{4}'.format(name, acc, nmi,
                                                                                          f1_score, confusion_mat))
            acc_test.append(acc)
            nmi_test.append(nmi)
            f1_score_test.append(f1_score)
            confusion_mat_test.append(confusion_mat)
        return acc_test, nmi_test, f1_score_test, confusion_mat_test

    acc_test, nmi_test, f1_score_test, confusion_mat_test = eval_classifiers(X_test, y_test)
    acc_train, nmi_train, f1_score_train, confusion_mat_train = eval_classifiers(X_train, y_train)
    if res is not None:
        acc_test = acc_test + res['test']['acc']
        nmi_test = nmi_test + res['test']['nmi']
        f1_score_test = f1_score_test + res['test']['f1_score']
        confusion_mat_test = confusion_mat_test + res['test']['confusion_mat']
        acc_train = acc_train + res['train']['acc']
        nmi_train = nmi_train + res['train']['nmi']
        f1_score_train = f1_score_train + res['train']['f1_score']
        confusion_mat_train = confusion_mat_train + res['train']['confusion_mat']
    res = {'train': {'acc': acc_train, 'nmi': nmi_train, 'f1_score': f1_score_train,
                     'confusion_mat': confusion_mat_train},
           'test': {'acc': acc_test, 'nmi': nmi_test, 'f1_score': f1_score_test,
                    'confusion_mat': confusion_mat_test}}
    return res


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

names = ["Neural Net", "Decision Tree", "AdaBoost", "Nearest Neighbors", "Linear SVM",
         "RBF SVM", "Random Forest", "Naive Bayes"]

classifiers = [
    MLPClassifier(alpha=1),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    KNeighborsClassifier(3),
    SVC(kernel="linear"),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()]


def plot_feature_weight(feature_weight_all, colors, title):
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(range(len(feature_weight_all)), feature_weight_all, c=colors, s=20)
    plt.xlabel('index')
    plt.ylabel('weight')
    if save_fig:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig('{0}/{1}.png'.format(save_folder, title), bbox_inches='tight', dpi=200)
    else:
        plt.title(title)
        plt.show()
    plt.close()


def plot_feature_weight_affinitynet(model_layers, title):
    print(title)
    feature_weight_all = np.ones(in_dim)
    for i in range(len(model_layers)):
        feature_weight = nn.functional.softmax(model_layers[i].components[0].a, 0).detach().cpu().data.numpy()
        print('layer{0}: {2}, {1}'.format(
            i, feature_weight[range(0, in_dim, signal_dim + noisy_dim)].tolist(),
            feature_weight[range(0, in_dim, signal_dim + noisy_dim)].sum() / feature_weight.sum()))
        feature_weight_all *= feature_weight
    feature_weight_all /= feature_weight_all.sum()
    print('overall: {1}, {0}'.format(
        feature_weight_all[range(0, in_dim, signal_dim + noisy_dim)].tolist(),
        feature_weight_all[range(0, in_dim, signal_dim + noisy_dim)].sum()))
    # put signal feature in the beginning
    feature_weight_all = np.concatenate([feature_weight_all[:signal_dim],
                                         feature_weight_all[signal_dim + noisy_dim:2 * signal_dim + noisy_dim],
                                         feature_weight_all[signal_dim:signal_dim + noisy_dim],
                                         feature_weight_all[2 * signal_dim + noisy_dim:]])
    colors = ['r'] * 2 * signal_dim + ['b'] * 2 * noisy_dim
    plot_feature_weight(feature_weight_all, colors, title)


def plot_result(loss_train, acc_train, loss_val, acc_val, avg='avg',
                title_prefix='training-affinitynet'):
    title = '{0}_best_val_acc_{1}={2}'.format(title_prefix, avg, acc_val[avg][-1])
    plt.figure(figsize=(figsize, figsize))
    plt.subplot(211)
    plt_loss_train, = plt.plot(loss_train[avg], 'r--')
    plt_loss_val, = plt.plot(loss_val[avg], 'g-')
    plt.legend([plt_loss_train, plt_loss_val], ['train', 'validation'], loc=0)
    plt.ylabel('loss')
    plt.subplot(212)
    plt_acc_train, = plt.plot(acc_train[avg], 'r--')
    plt_acc_val, = plt.plot(acc_val[avg], 'g-')
    plt.legend([plt_acc_train, plt_acc_val], ['train', 'validation'], loc=0)
    plt.ylabel('accuracy %')
    plt.xlabel('iterations')
    if save_fig:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig('{0}/{1}.png'.format(save_folder, title), bbox_inches='tight', dpi=200)
    else:
        plt.title(title)
        plt.show()
    plt.close()


def plot_feature_weight_linear(model, title):
    print(title)
    feature_weight = model.layers.linear0.weight
    for i in range(1, 1 + len(model.layers) // 2):
        feature_weight = torch.mm(getattr(model.layers, 'linear' + str(i)).weight, feature_weight)
    feature_weight = feature_weight.data.abs().mean(0).numpy()
    feature_weight_all = feature_weight / feature_weight.sum()
    print('overall: {0}, {1}'.format(
        feature_weight_all[range(0, in_dim, signal_dim + noisy_dim)].sum(),
        feature_weight_all[range(0, in_dim, signal_dim + noisy_dim)].tolist()))

    # put signal feature in the beginning
    feature_weight_all = np.concatenate([feature_weight_all[:signal_dim],
                                         feature_weight_all[signal_dim + noisy_dim:2 * signal_dim + noisy_dim],
                                         feature_weight_all[signal_dim:signal_dim + noisy_dim],
                                         feature_weight_all[2 * signal_dim + noisy_dim:]])
    colors = ['r'] * 2 * signal_dim + ['b'] * 2 * noisy_dim
    plot_feature_weight(feature_weight_all, colors, title)


def eval_affinitynet(data, res):
    model = MultiviewAttention(in_dim=in_dim, hidden_dims=[in_dim], k=10, graph=None,
                               out_indices=None,
                               feature_subset=None, kernel='gaussian', nonlinearity_1=None,
                               nonlinearity_2=None, use_previous_graph=False,
                               group_index=None, merge=None,
                               merge_type='affine', reset_graph_every_forward=False,
                               no_feature_transformation=True, rescale=True, merge_dim=2)

    if x_var.numel() < 10 ** 6:
        title = 'raw data PCA'
        plot_scatter(x_var, title=title, colors=y, folder=save_folder, save_fig=save_fig,
                     size=figsize)

        y_pred = model(x_var)
        title = 'before training output PCA'
        plot_scatter(y_pred, title=title, colors=y, folder=save_folder, save_fig=save_fig,
                     size=figsize)

    title = 'feature weight distribution: before training'
    plot_feature_weight_affinitynet(model.layers, title)

    model = nn.Sequential(model, DenseLinear(in_dim, hidden_dims + [num_cls]))

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    solver = Solver(model, data, optimizer, loss_fn)

    loss_train, acc_train, loss_val, acc_val = solver.train_eval(
        num_iter=num_iter, batch_size=batch_size, X=None, y=None, X_val=None, y_val=None,
        X_test=None, y_test=None, eval_test=False, balanced_sample=True)

    plot_result(loss_train, acc_train, loss_val, acc_val, avg='avg')
    plot_result(loss_train, acc_train, loss_val, acc_val, avg='batch')

    title = 'Feature weights after training'
    plot_feature_weight_affinitynet(model[0].layers, title)

    acc, nmi, confusion_mat, f1_score = visualize_val(
        data['X_train'], data['y_train'], solver, batch_size=batch_size,
        title='affinitynet X_train', topk=1, save_fig=save_fig, save_folder=save_folder)
    res[train_portion]['train']['acc'].append(acc)
    res[train_portion]['train']['nmi'].append(nmi)
    res[train_portion]['train']['f1_score'].append(f1_score)
    res[train_portion]['train']['confusion_mat'].append(confusion_mat)

    acc, nmi, confusion_mat, f1_score = visualize_val(
        data['X_val'], data['y_val'], solver, batch_size=batch_size, title='affinitynet X_val', topk=1,
        save_fig=save_fig, save_folder=save_folder)
    res[train_portion]['test']['acc'].append(acc)
    res[train_portion]['test']['nmi'].append(nmi)
    res[train_portion]['test']['f1_score'].append(f1_score)
    res[train_portion]['test']['confusion_mat'].append(confusion_mat)

    cnt = 0
    for n, p in model.named_parameters():
        print(n, p.numel())
        cnt += p.numel()
    print('total param:{0}'.format(cnt))

    model = DenseLinear(in_dim, hidden_dims + [num_cls])

    cnt = 0
    for n, p in model.named_parameters():
        print(n, p.numel())
        cnt += p.numel()
    print('total param:{0}'.format(cnt))

    title = 'Feature weights before training Linear'
    plot_feature_weight_linear(model, title)

    # set a smaller learning rate for DenseLinear
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    solver = Solver(model, data, optimizer, loss_fn)

    loss_train, acc_train, loss_val, acc_val = solver.train_eval(
        num_iter=num_iter, batch_size=batch_size, X=None, y=None, X_val=None, y_val=None,
        X_test=None, y_test=None, eval_test=False, balanced_sample=True)

    plot_result(loss_train, acc_train, loss_val, acc_val, avg='avg', title_prefix='training-linear')
    plot_result(loss_train, acc_train, loss_val, acc_val, avg='batch', title_prefix='training-linear')

    title = 'Feature weights after training Linear'
    plot_feature_weight_linear(model, title)

    acc, nmi, confusion_mat, f1_score = visualize_val(
        data['X_train'], data['y_train'], solver, batch_size=batch_size,
        title='linear X_train', topk=1, save_fig=save_fig, save_folder=save_folder)
    res[train_portion]['train']['acc'].append(acc)
    res[train_portion]['train']['nmi'].append(nmi)
    res[train_portion]['train']['f1_score'].append(f1_score)
    res[train_portion]['train']['confusion_mat'].append(confusion_mat)

    acc, nmi, confusion_mat, f1_score = visualize_val(
        data['X_val'], data['y_val'], solver, batch_size=batch_size, title='linear X_val', topk=1,
        save_fig=save_fig, save_folder=save_folder)
    res[train_portion]['test']['acc'].append(acc)
    res[train_portion]['test']['nmi'].append(nmi)
    res[train_portion]['test']['f1_score'].append(f1_score)
    res[train_portion]['test']['confusion_mat'].append(confusion_mat)


x_var = Variable(torch.from_numpy(x).float())
y_var = Variable(torch.from_numpy(y).long())
num_cls = y_var.data.max().item() + 1

train_portions = [0.01]  # [0.005, 0.01, 0.1, 0.2, 0.5, 0.8]
res = {}

for train_portion in train_portions:
    proportions = [train_portion] * num_cls
    x_train, y_train, x_test, y_test, train_idx, test_idx = split_data(
        x_var, y_var, proportions=proportions, seed=seed)
    print('train size: {0}, test size: {1}'.format(y_train.size(0), y_test.size(0)))

    data = {'X_train': x_train.data, 'y_train': y_train.data, 'X_val': x_test.data, 'y_val': y_test.data,
            'X_test': x_test.data, 'y_test': y_test.data}
    save_folder = '{0}/train_portion-{1}/'.format(save_folder_prefix, train_portion)

    res[train_portion] = compare_classifiers(x_train.data.numpy(), x_test.data.numpy(),
                                             y_train.data.numpy(), y_test.data.numpy(),
                                             names, classifiers)

    eval_affinitynet(data, res)

# # if not os.path.exists(save_folder_prefix):
# #     os.mkdir(save_folder_prefix)
# # with open('{0}/res.pkl'.format(save_folder_prefix), 'wb') as f:
# #     pickle.dump(res, f)