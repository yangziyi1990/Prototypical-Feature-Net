import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})


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