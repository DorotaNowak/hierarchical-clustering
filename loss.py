import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_probability_level(feature, level):
    leaves = 2 ** (level - 1)
    probability_vector = torch.ones((feature.shape[0], leaves)).to("cuda")
    for u in torch.arange(2 ** (level - 1), 2 ** level, dtype=torch.long):
        w = u
        while u > 1:
            if u / 2 == torch.floor(u / 2):
                # Go left
                u = torch.floor(u / 2)
                u = u.long()
                probability_vector[:, w - leaves] *= feature[:, u - 1]
            else:
                # Go right
                u = torch.floor(u / 2)
                u = u.long()
                probability_vector[:, w - leaves] *= (1 - feature[:, u - 1])

    return probability_vector


def mask_correlated_samples(leaves):
    N = 2 * leaves
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(leaves):
        mask[i, leaves + i] = 0
        mask[leaves + i, i] = 0
    mask = mask.bool()
    return mask


def base_binary_loss(c_i, c_j, node_mask, levels=5):
    features = torch.cat((c_i, c_j), dim=0)  # 256x15
    batch_size = c_i.shape[0]  # 128

    loss_value = torch.tensor([0], device="cuda", dtype=torch.float32)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)  # 256
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to("cuda")

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to("cuda")
    labels = labels * ~mask  # 256x256

    for level in range(2, levels + 1):
        prob_features = calculate_probability_level(features, level)  # 256x2, 256x4, 256x8,...
        prob_features = prob_features * torch.abs(
            node_mask[2 ** (level - 1) - 1: 2 ** level - 1])  #
        # Calculate loss on positive classes
        # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
        loss_value -= torch.mean((torch.bmm(
            torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) + 1e-8),
            torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
        # Calculate loss on negative classes
        labels[:batch_size, :batch_size] = 1
        labels[batch_size:2 * batch_size, batch_size:2 * batch_size] = 1
        loss_value += torch.mean((torch.bmm(
            torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8),
            torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))

    return loss_value


def binary_loss(c_i, c_j, node_mask, levels=5, temperature=1.0):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0

    ne_i = node_entropy(c_i[:, 0].mean())
    ne_j = node_entropy(c_j[:, 0].mean())
    ne_loss = ne_i + ne_j
    total_loss -= 0.5 * ne_loss

    for level in range(2, levels + 1):
        h_i = calculate_probability_level(c_i, level) * torch.abs(
            node_mask[2 ** (level - 1) - 1: 2 ** level - 1])  # 128x16
        h_j = calculate_probability_level(c_j, level) * torch.abs(
            node_mask[2 ** (level - 1) - 1: 2 ** level - 1])  # 128x16

        # foreach node on level level
        weight = 2 ** - level
        if level < levels:
            for node in range(2 ** (level - 1)):
                if node_mask[2 ** (level - 1) - 1 + node] == 1:
                    denominator_i = h_i[:, node].sum() + 1e-8
                    numerator_i = torch.sum(h_i[:, node] * c_i[:, 2 ** (level - 1) - 1 + node])
                    denominator_j = h_j[:, node].sum() + 1e-8
                    numerator_j = torch.sum(h_j[:, node] * c_j[:, 2 ** (level - 1) - 1 + node])
                    ne_i = node_entropy(numerator_i / denominator_i)
                    ne_j = node_entropy(numerator_j / denominator_j)
                    ne_loss = ne_i + ne_j
                    total_loss -= weight * ne_loss

        h_i = F.normalize(h_i.T, dim=1)
        h_j = F.normalize(h_j.T, dim=1)

        loss = instance_loss(h_i, h_j, 1, 2 ** (level - 1))

        total_loss += loss

    return total_loss


def instance_loss(out_1, out_2, temperature=0.5, batch_size=128):
    out = torch.cat([out_1, out_2], dim=0)  # 256x128
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # 256x256
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()  # 256x256
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)  # 256x255
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)  # 128
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # 256
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


def entropy(h):
    p = h.sum(0).view(-1)
    p /= p.sum()
    p += 1e-8
    ent = math.log(p.size(0)) + (p * torch.log(p)).sum()
    return ent


def node_entropy(h):
    return 0.5 * torch.log(h + 1e-8) + 0.5 * torch.log(1 - h + 1e-8)
