import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import euclidean_dist, register_model


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder


    def loss(self, sample):
        xs = Variable(sample['xs'])
        xq = Variable(sample['xq'])

        n_class = sample['xs_class'].max().item() + 1
        n_support = int(sample['xs'].size(0)/n_class)
        n_query = int(sample['xq'].size(0)/n_class)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        x = torch.cat([xs, xq], 0)
        z = self.encoder.forward(x)                                              # sample is calculated by feature attention layer

        z_dim = z.size(-1)
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)  # Calculate the prototype for support set
        zq = z[n_class*n_support:]                                               # extract the query set samples

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


class feature_attention_layer(nn.Module):

    def __init__(self, feature_dim):
        super(feature_attention_layer, self).__init__()
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.Tensor(self.feature_dim))
        self.weight.data.uniform_(0, 1)
        #     self.reset_parameters()
        #
        # def reset_parameters(self):
        #     stdv = 1./ math.sqrt(self.weight.size(0))
        #     self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        feature_weight = nn.functional.softmax(self.weight, dim=0)
        out = x * feature_weight

        return out

@register_model('protonet_conv')
def load_Protonet(**kwargs):     # object or nn.Module??
                                 # Do we need add a Neural network layer after feature attention layer?
    feature_dim = kwargs['feature_dim']

    def feature_lay(in_dim):
        return nn.Sequential(
            feature_attention_layer(in_dim)
        )

    encoder = nn.Sequential(
        feature_lay(feature_dim)
    )

    return Protonet(encoder)