import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from loss import calculate_probability_level


class ResNet50(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet50, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # Encoder
        self.f = nn.Sequential(*self.f)

        # Projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True))

    def get_feature(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature

    def forward(self, x):
        feature = self.get_feature(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Model(nn.Module):
    def __init__(self, resnet, levels=5):
        super(Model, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        # Binary projection head
        self.cluster_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, self.cluster_num - 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)
        c = self.cluster_projector(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), c

    def forward_cluster(self, x):
        feature = self.resnet.get_feature(x)
        c = self.cluster_projector(feature)

        probability_vector = calculate_probability_level(c, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector
