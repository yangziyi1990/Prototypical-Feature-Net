import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.decomposition import PCA, IncrementalPCA


def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})


def plot_true_signal(x, y, title, figsize):
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title(title)
    plt.show()
    plt.close()


def plot_feature_weight(feature_weight_all, colors, title, figsize):
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(range(len(feature_weight_all)), feature_weight_all, c=colors, s=20)
    plt.xlabel('index')
    plt.ylabel('weight')
    plt.title(title)
    plt.show()
    plt.close()

def plot_feature_weight_linear(feature, title, signal_dim, noisy_dim, figsize):
    num = len(feature) - 1
    feature_weight = feature[num].data.numpy()
    feature_weight = np.concatenate([feature_weight[:signal_dim],
                                     feature_weight[signal_dim + noisy_dim:2 * signal_dim + noisy_dim],
                                     feature_weight[signal_dim:signal_dim + noisy_dim],
                                     feature_weight[2 * signal_dim + noisy_dim:]])
    colors = ['r'] * 2 * signal_dim + ['b'] * 2 * noisy_dim
    plot_feature_weight(feature_weight, colors, title, figsize)


def plot_feature_weight_linear_metatest(feature, title, signal_dim, noisy_dim, figsize):
    feature_weight = np.concatenate([feature[:signal_dim],
                                     feature[signal_dim + noisy_dim:2 * signal_dim + noisy_dim],
                                     feature[signal_dim:signal_dim + noisy_dim],
                                     feature[2 * signal_dim + noisy_dim:]])
    colors = ['r'] * 2 * signal_dim + ['b'] * 2 * noisy_dim
    plot_feature_weight(feature_weight, colors, title, figsize)

def pca(x, n_components=2, verbose=False):
    r"""PCA for 2-D visualization
    """
    if len(x)>10000:
        pca = IncrementalPCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    if isinstance(x, Variable):
        x = x.cpu().numpy().copy()
    pca.fit(x)
    if verbose:
        print(pca.explained_variance_, pca.noise_variance_)
        plt.title('explained_variance')
        plt.plot(pca.explained_variance_.tolist() + [pca.noise_variance_], 'ro')
        plt.show()
    return pca.fit_transform(x)



def plot_scatter(y_=None, model_=None, x_=None, title='', labels=None, colors=None, size=15,
                 marker_size=20, folder='.', save_fig=False):
    r"""2D scatter plot
    """
    if y_ is None:
        assert model_ is not None and x_ is not None
        y_ = model_(x_.contiguous())
    if colors is not None:
        assert len(colors) == len(y_)
    else:
        if labels is not None:
            assert len(y_) == len(labels)
            color = sorted(matplotlib.colors.BASE_COLORS)
            colors = [color[i] for i in labels]
    if isinstance(y_, Variable):
        y_ = y_.data.cpu().numpy()
    if y_.shape[1] > 2:
        y_ = pca(y_)
    plt.figure(figsize=(size, size))
    plt.scatter(y_[:,0],y_[:,1], c=colors, s=marker_size)
    if save_fig:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+'/'+title+'.png', bbox_inches='tight', dpi=200)
    else:
        plt.title(title)
        plt.show()
    plt.close()


def plot_result(loss_train_Proto, loss_train_Proto_IATT, loss_train_Proto_FATT, loss_train_Proto_HATT,
                acc_vali_Proto, acc_vali_Proto_IATT, acc_vali_Proto_FATT, acc_vali_Proto_HATT,
                iter_times, figsize, title):

    plt.figure(figsize=(figsize, figsize))
    plt.subplot(211)
    plt.title(title)
    x_iter = range(iter_times)
    plt_loss_Proto, = plt.plot(x_iter, loss_train_Proto, '-')
    plt_loss_Proto_IATT, = plt.plot(x_iter, loss_train_Proto_IATT, '-')
    plt_loss_Proto_FATT, = plt.plot(x_iter, loss_train_Proto_FATT, '-')
    plt_loss_Proto_HATT, = plt.plot(x_iter, loss_train_Proto_HATT, '-')

    plt.legend([plt_loss_Proto, plt_loss_Proto_IATT, plt_loss_Proto_FATT, plt_loss_Proto_HATT], ['Proto', 'Proto_IATT', 'Proto_FATT', 'Proto_HATT'], loc=0)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(212)
    plt_acc_Proto, = plt.plot(x_iter, acc_vali_Proto, '-')
    plt_acc_Proto_IATT, = plt.plot(x_iter, acc_vali_Proto_IATT, '-')
    plt_acc_Proto_FATT, = plt.plot(x_iter, acc_vali_Proto_FATT, '-')
    plt_acc_Proto_HATT, = plt.plot(x_iter, acc_vali_Proto_HATT, '-')

    plt.legend([plt_acc_Proto, plt_acc_Proto_IATT, plt_acc_Proto_FATT, plt_acc_Proto_HATT], ['Proto', 'Proto_IATT', 'Proto_FATT', 'Proto_HATT'], loc=0)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()
    plt.close()