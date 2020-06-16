import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('input is ', input.shape)
        # print('weight is ', self.weight.shape)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=64, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # print('in_channel is ',in_channel)
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        # print('in init after get_A ',_adj)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.linear = nn.Linear(20, 5)


    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        inp = inp[0]
        adj1 = gen_adj(self.A)
        # print('adj1 matrix is ', adj1)
        adj = gen_adj(self.A).detach()
        # print('check nan 6 ', adj)
        x = self.gc1(inp, adj)
        # print('check nan 5 ', x)
        x = self.relu(x)
        # print('check nan 4 ', x)
        x = self.gc2(x, adj)
        # print('check nan 3 ', x)
        x = x.transpose(0, 1)
        # fuse representation learning and gcn feature to get multi-label prediction
        # print('check nan 2 ', x)
        x = torch.matmul(feature, x)
        # print('before linear x.shape is ',x.shape)
        # TODO
        # adding a layer to reduce into 5-dim, do the classification job
        x = self.linear(x)
        # return F.log_softmax(x, dim=1)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=128):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
