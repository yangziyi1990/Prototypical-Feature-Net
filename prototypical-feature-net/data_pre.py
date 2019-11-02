import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from functools import partial


if torch.cuda.is_available():
    dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor}
else:
    dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor}

def convert_dict(k, v):
    return { k: v }

def randperm(idx, random_examples=False, seed=None):
    """Randomly permute indices

    Args:
        idx: torch.LongTensor, indices to be permuted
        random_examples: bool, if True, return a random permutation
        seed: if int, then set seed before random permutation

    """
    n = len(idx)
    if isinstance(seed, int):
        torch.manual_seed(seed)
        return idx[torch.randperm(n)]
    if random_examples:
        return idx[torch.randperm(n)]
    else:
        return idx

def split_train_test(x_var, y_var, train_indices, y_true=None, seed=None):
    r"""Split data into training and test (validation) set

    Arg:
        x_var, y_var: Variable or torch.Tensor, the first dimension will be splitted
        train_indices: torch.LongTensor
        y_true: y_test = y_var[test_indices] if y_true is None else y_true[test_indices]

    Returns:

    Examples:

        >>>
    """
    test_indices = dtype['long'](sorted(set(range(x_var.size(0))).difference(train_indices.cpu().numpy())))
    if seed is not None:
        train_indices = randperm(train_indices, random_examples=True, seed=seed)
        test_indices = randperm(test_indices, random_examples=True, seed=seed)
    x_train = x_var[train_indices]
    y_train = y_var[train_indices]
    x_test = x_var[test_indices]
    if y_true is None:
        y_test = y_var[test_indices]
    else:
        y_test = y_true[test_indices]
    return x_train, y_train, x_test, y_test, train_indices, test_indices


def split_data(x_var, y_var, num_examples=1, proportions=None, seed=None, random_examples=False):
    num_clusters = y_var.max().item() + 1 # assume y_var is LongTensor starting from 0 to num_cls-1
    if proportions is not None:
        if isinstance(proportions, float):
            assert proportions > 0 and proportions < 1
            proportions = [proportions]*num_clusters
        num_examples = [max(1,round(torch.nonzero(y_var==i).size(0) * proportions[i])) for i in range(num_clusters)]
    if isinstance(num_examples, int):
        num_examples_per_class = num_examples
        num_examples = [num_examples_per_class]*num_clusters
    assert num_clusters == len(num_examples)
    train_indices = [randperm(torch.nonzero(y_var==i), random_examples, seed)[:num_examples[i],0]
                     for i in range(num_clusters)]
    train_indices = torch.cat(train_indices, dim=0).data
    return split_train_test(x_var, y_var, train_indices, seed=seed)

def ZiyiSampler(seq_list, batch_size, clu_size):

    sampler_list = []
    for i in range((clu_size-1)//batch_size + 1):
        sampler_list.append(seq_list[i*batch_size:i*batch_size+batch_size])

    return sampler_list


def ZiyiSampler1(seq_dict, batch_size, clu_size):
## the type of seq_dict is dictionary
    sampler_dict = {}
    for i in range((clu_size-1)//batch_size + 1):
        sampler_list = []
        for key in seq_dict:
            sampler_list.extend(seq_dict[key][i*batch_size:i*batch_size+batch_size])
        sampler_dict[i] = sampler_list

    sampler = []
    for key in sampler_dict:
        sampler.append(sampler_dict[key])

    return sampler


def ZiyiDataLoader(x, y, sampler):

    x_episode = {}
    y_episode = {}
    for i in range(len(sampler)):
        data_x = x[sampler[i],]
        data_y = y[sampler[i]]
        x_episode[i] = data_x
        y_episode[i] = data_y

    return x_episode, y_episode

def ZiyiDataLoader1(x, y, sampler):

    data_episode = {}
    for i in range(len(sampler)):
        data_x = x[sampler[i],]
        data_y = y[sampler[i]]
        data_episode[i] = {'xs': data_x, 'class': data_y}

    return data_episode

def ZiyiDataLoader2(x, y, sampler, n_support, n_query):

    data_episode = {}
    for i in range(len(sampler)):
        data_xs = []
        data_xq = []
        label_xs = []
        label_xq = []

        for j in range(0, len(y[sampler[i]]), n_support + n_query):
            data_xs.extend(x[sampler[i][j: j + n_support], ].numpy())
            data_xq.extend(x[sampler[i][j + n_support: j + n_support + n_query], ].numpy())
            label_xs.extend(y[sampler[i]][j: j + n_support])
            label_xq.extend(y[sampler[i]][j + n_support: j + n_support + n_query])

        data_xs = torch.FloatTensor(data_xs)
        data_xq = torch.FloatTensor(data_xq)
        label_xs = torch.LongTensor(label_xs)
        label_xq = torch.LongTensor(label_xq)

        data_episode[i] = {'xs': data_xs, 'xs_class': label_xs,
                           'xq': data_xq, 'xq_class': label_xq}

    return data_episode