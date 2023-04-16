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


class BaseModel(nn.Module):
    def __init__(self, resnet, levels=5):
        super(BaseModel, self).__init__()
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


class Model(nn.Module):
    def __init__(self, resnet, levels=5):
        super(Model, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        self.router = nn.Sequential(
            nn.Linear(2048, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(transformed)  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 15, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(transformed)  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(transformed)  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 16, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(transformed)  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


class Model2(nn.Module):
    def __init__(self, resnet, levels=5):
        super(Model2, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        self.router = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 15, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 16, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


class Model3(nn.Module):
    def __init__(self, resnet, levels=5):
        super(Model3, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.node_encoder = self.transformers = nn.Sequential(
            nn.Linear(self.cluster_num - 1, self.cluster_num - 1),
            nn.ReLU(inplace=True),
            nn.Linear(self.cluster_num - 1, self.cluster_num - 1),
        )
        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        self.router = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.
        node_encoded = self.node_encoder(node_encoded)

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 15, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            node_encoded = self.node_encoder(node_encoded)
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.
        node_encoded = self.node_encoder(node_encoded)

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 16, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.
            node_encoded = self.node_encoder(node_encoded)
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


class Model4(nn.Module):
    """The same representation for each node."""
    def __init__(self, resnet, levels=5):
        super(Model4, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        self.router = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.

            transformed = self.transformers(
                torch.cat((feature, node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.

            transformed = self.transformers(
                torch.cat((feature, node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


class Model5(nn.Module):
    """Create 15 different networks."""
    def __init__(self, resnet, levels=5):
        super(Model5, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            )
            for _ in range(self.cluster_num - 1)])
        self.router = nn.ModuleList([
            nn.Sequential(
            nn.Linear(2048, 2),
            nn.Softmax(1)
            )
            for _ in range(self.cluster_num - 1)])

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x).to('cuda')
        out = self.resnet.g(feature).to('cuda')

        transformed = self.transformers[0](feature).to('cuda')  # transformed.shape = [bs, 2048]
        pr = self.router[0](transformed)  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            transformed = self.transformers[node](feature).to('cuda')  # [batch_size, 2048]
            pr = self.router[node](transformed)  # [bs, 2]

            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x).to('cuda')

        transformed = self.transformers[0](feature).to('cuda')  # transformed.shape = [bs, 2048]
        pr = self.router[0](transformed)  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            transformed = self.transformers[node](feature).to('cuda')  # [batch_size, 2048]
            pr = self.router[node](transformed)  # [bs, 2]

            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


def update_path(node_encoded, node):
    if node > 0:
        parent = int((node - 1) / 2)
        node_encoded[:, parent] = 1
        update_path(node_encoded, parent)

    return node_encoded


class Model6(nn.Module):
    """Encode each node not as a one hot vector but as a path from root."""
    def __init__(self, resnet, levels=5):
        super(Model6, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        self.router = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, self.cluster_num - 1)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 15, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded = node_encoded.fill_(0)
            node_encoded[:, node] = 1.
            node_encoded = update_path(node_encoded, node)
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        transformed = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]
        pr = self.router(torch.cat((transformed, node_encoded), 1))  # pr.shape = [bs, 2]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        representations = torch.ones((batch_size, 15, 2048)).to('cuda')  # representations.shape = [bs, 16, 2048]
        representations[:, 0, :] = transformed

        for node in range(1, 15):
            node_encoded = node_encoded.fill_(0)
            node_encoded[:, node] = 1.
            node_encoded = update_path(node_encoded, node)
            parent = int((node - 1) / 2)

            transformed = self.transformers(
                torch.cat((representations[:, parent, :], node_encoded), 1))  # [batch_size, 2048]
            pr = self.router(torch.cat((transformed, node_encoded), 1))  # [bs, 2]

            representations[:, node, :] = transformed
            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector


class Model7(nn.Module):
    """The same representation for each node."""
    def __init__(self, resnet, levels=5):
        super(Model7, self).__init__()
        self.resnet = resnet
        self.levels = levels
        self.cluster_num = 2 ** (self.levels - 1)

        self.transformers = nn.Sequential(
            nn.Linear(2048 + self.cluster_num - 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
            nn.Softmax(1)
        )


    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)
        out = self.resnet.g(feature)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        pr = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 15]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.

            pr = self.transformers(
                torch.cat((feature, node_encoded), 1))  # [batch_size, 2048]

            probabilities[:, node] = pr[:, 0]

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), probabilities

    def forward_cluster(self, x):
        batch_size = x.shape[0]
        feature = self.resnet.get_feature(x)

        node_encoded = torch.zeros((batch_size, 15)).to('cuda')  # node_encoded.shape = [bs, 15]
        node_encoded[:, 0] = 1.

        pr = self.transformers(torch.cat((feature, node_encoded), 1))  # transformed.shape = [bs, 2048]

        probabilities = torch.ones((batch_size, 15)).to('cuda')  # probabilities.shape = [bs, 16]
        probabilities[:, 0] = pr[:, 0]

        for node in range(1, 15):
            node_encoded[:, node - 1] = 0.
            node_encoded[:, node] = 1.

            pr = self.transformers(
                torch.cat((feature, node_encoded), 1))  # [batch_size, 2048]
            probabilities[:, node] = pr[:, 0]

        probability_vector = calculate_probability_level(probabilities, self.levels)
        cluster_idx = torch.argmax(probability_vector, dim=1)
        return cluster_idx, probability_vector