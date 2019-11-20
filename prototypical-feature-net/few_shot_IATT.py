import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import register_model


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def loss(self, sample):
        xs = Variable(sample['xs'])
        xq = Variable(sample['xq'])

        n_class = sample['xs_class'].max().item() + 1
        n_support = int(sample['xs'].size(0)/n_class)
        n_query = int(sample['xq'].size(0)/n_class)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        x = torch.cat([xs, xq], 0)


        ## feature attention layer
        z, feature_weight = self.encoder[0].forward(x)          # sample is calculated by feature attention layer


        # instance attention layer
        z_new = {'data': z, 'n_class': n_class, 'n_support': n_support, 'n_query': n_query}
        z_proto = self.encoder[1].forward(z_new)
        zq = z[n_class * n_support:]

        # Prototypical Networks
        dists = self.__batch_dist__(z_proto, zq)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'fw': feature_weight
        }


class feature_attention_layer(nn.Module):

    def __init__(self, feature_dim, ):
        super(feature_attention_layer, self).__init__()
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.Tensor(self.feature_dim))
        self.weight.data.uniform_(0, 1)


    def forward(self, x):
        feature_weight = self.weight
        out = x * feature_weight

        return out, feature_weight


class instance_attention_layer(nn.Module):

    def __init__(self, feature_dim):
        super(instance_attention_layer, self).__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, feature_dim, bias=True)

    def forward(self, x):
        z = x['data']
        n_class = x['n_class']
        n_support = x['n_support']
        n_query = x['n_query']

        support = z[:n_class*n_support].view(n_class, n_support, self.feature_dim).unsqueeze(0).expand(n_query*n_class, -1, -1, -1)
        # support_for_att = self.fc(support)
        # query_for_att = self.fc(z[n_class*n_support:].unsqueeze(1).unsqueeze(2).expand(-1, n_class, n_support, -1))
        support_for_att = support
        query_for_att = z[n_class * n_support:].unsqueeze(1).unsqueeze(2).expand(-1, n_class, n_support, -1)
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)
        support_proto = (support * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.feature_dim)).sum(2)

        return support_proto




@register_model('protonet_conv')
def load_Protonet_IATT(**kwargs):     # Do we need add a Neural network layer after feature attention layer?

    feature_dim = kwargs['feature_dim']

    def feature_lay(in_dim):
        return nn.Sequential(
            feature_attention_layer(in_dim)
        )

    def instance_lay(in_dim):
        return  nn.Sequential(
            instance_attention_layer(in_dim)
        )

    encoder = nn.Sequential(
        feature_lay(feature_dim),
        instance_lay(feature_dim)
    )

    return Protonet(encoder)