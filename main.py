import argparse
import os
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cluster import inference, evaluate
from model import ResNet50, BaseModel, Model2, Model3, Model4, Model6, Model7
from loss import binary_loss, base_binary_loss, instance_loss


def get_leaf_to_delete(model, loader, device, mask):
    model.eval()
    clusters = np.zeros(16)

    for step, z in enumerate(loader):
        x = z[0][0]
        x = x.to(device)
        with torch.no_grad():
            c, probabilities = model.forward_cluster(x)
            probabilities = torch.sum(probabilities, dim=0)
            clusters += probabilities.cpu().detach().numpy()
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    sorted_clusters = np.argsort(clusters)  # 0:15
    print(mask[15:])
    print(sorted_clusters)
    i = 0
    while True:
        if mask[sorted_clusters[i] + 15] == 1:
            print(f"Cluster to delete: {sorted_clusters[i]}")
            return sorted_clusters[i] + 15
        i += 1


def update_mask(mask, leaf):
    mask[leaf] = 0
    parent = int((leaf - 1) / 2)
    if mask[parent] == 1:
        mask[parent] = -1
    else:
        update_mask(mask, parent)

    return mask


# Train for one epoch
def train(net, data_loader, train_optimizer, mask, temperature, batch_size, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_feature_loss, total_cluster_loss = 0.0, 0.0
    for (pos_1, pos_2), target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1, c_1 = net(pos_1)
        feature_2, out_2, c_2 = net(pos_2)
        feature_loss = instance_loss(out_1, out_2, temperature, batch_size)
        cluster_loss = binary_loss(c_1, c_2, mask)
        loss = feature_loss + cluster_loss
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        total_feature_loss += feature_loss.item() * batch_size
        total_cluster_loss += cluster_loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num, total_feature_loss / total_num, total_cluster_loss / total_num


def run(model, train_loader, optimizer, mask, total_epoch, temperature, batch_size, epoch):
    train_loss, feature_loss, cluster_loss = train(model, train_loader, optimizer, mask, temperature, batch_size, epoch)
    results['train_loss'].append(train_loss)
    writer.add_scalar("Loss/train", train_loss, total_epoch)
    writer.add_scalar("Loss/instance_loss", feature_loss, total_epoch)
    writer.add_scalar("Loss/cluster_loss", cluster_loss, total_epoch)

    pred, true = inference(test_loader, model, device, mask[first_leaf_idx:all_nodes])
    nmi, ari, f = evaluate(true, pred)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f}'.format(nmi, ari, f))
    results['nmi'].append(nmi)
    results['ari'].append(ari)
    results['f'].append(f)
    writer.add_scalar("Metrics/nmi", nmi, total_epoch)
    writer.add_scalar("Metrics/ari", ari, total_epoch)
    writer.add_scalar("Metrics/f", f, total_epoch)

    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
    results['test_acc@1'].append(test_acc_1)
    writer.add_scalar("Accuracy/@1", test_acc_1, total_epoch)
    results['test_acc@5'].append(test_acc_5)
    writer.add_scalar("Accuracy/@5", test_acc_5, total_epoch)
    # Save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, total_epoch+1))
    data_frame.to_csv(f'{path}/{save_name_pre}_statistics.csv', index_label='total_epoch')


# Test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # Generate feature bank
        for (data, _), target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out, ci = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # Loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for (data, _), target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out, ci = net(data)

            total_num += data.size(0)
            # Compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # Counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # Weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset_name', default=None, type=str, help='Name of the dataset')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train combined model')
    parser.add_argument('--pruning_epochs', default=2, type=int, help='Number of epochs to train between pruning steps')
    parser.add_argument('--path', default=None, type=str, help='Path to save the model')
    parser.add_argument('--model', default='base', type=str, help='Model to use')
    parser.add_argument('--tree_height', default=None, type=int, help='The height of a tree to train')

    # Arg parse
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs, pruning_epochs = args.batch_size, args.epochs, args.pruning_epochs
    path = args.path
    model_type = args.model

    # Initialize summary writer
    writer = SummaryWriter(log_dir=f"runs/{dataset_name}/{model_type}")

    # Prepare the data
    train_data = utils.SimCLRDataset(dataset_name, 'train', True).get_dataset()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.SimCLRDataset(dataset_name, 'test', True).get_dataset()
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.SimCLRDataset(dataset_name, 'test', False).get_dataset()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Find the height of a tree to train starting from 0
    c = len(memory_data.classes)
    if args.tree_height is not None:
        height = args.tree_height
    else:
        height = 1
        while c > 2 ** height:
            height += 1
    print(f'Tree height: {height}')
    levels = height + 1

    # Backbone model
    resnet = ResNet50(dataset_name, feature_dim).cuda()
    resnet_optimizer = optim.Adam(resnet.parameters(), lr=5e-4, weight_decay=1e-6)
    resnet_path = f'results/{dataset_name}/resnet/128_0.5_200_128_1000_500_model.pth'
    checkpoint = torch.load(resnet_path)
    resnet.load_state_dict(checkpoint['state_dict'])
    resnet_optimizer.load_state_dict(checkpoint['optimizer'])

    # Main model
    if model_type == 'base':
        model = BaseModel(resnet, levels).cuda()
    elif model_type == 'model2':
        model = Model2(resnet, levels).cuda()
    elif model_type == 'model3':
        model = Model3(resnet, levels).cuda()
    elif model_type == 'model4':
        model = Model4(resnet, levels).cuda()
    elif model_type == 'model6':
        model = Model6(resnet, levels).cuda()
    elif model_type == 'model7':
        model = Model7(resnet, levels).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # Training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'nmi': [], 'ari': [], 'f': []}
    save_name_pre = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}'

    # Find the first leaf index, from root = 0
    first_leaf_idx = 2 ** height - 1
    all_nodes = 2 ** (height + 1) - 1
    mask = torch.ones(all_nodes).cuda()
    leaves_to_delete = 2 ** height - c

    # Path to save the results
    if path is None:
        path = f'results/{dataset_name}/{model_type}'

    if not os.path.exists(path):
        os.makedirs(path)

    total_epochs = 0

    # Train loop
    for epoch in range(1, epochs + 1):
        total_epochs += 1
        run(model, train_loader, optimizer, mask, total_epochs, temperature, batch_size, epoch)

    # Pruning
    for i in range(leaves_to_delete):
        print(f"Iteration {i}")
        for epoch in range(1, pruning_epochs + 1):
            total_epochs += 1
            print(total_epochs)
            run(model, train_loader, optimizer, mask, total_epochs, temperature, batch_size, epoch)

        leaf = get_leaf_to_delete(model, memory_loader, 'cuda', mask)
        print(f"Leaf to delete: {leaf}")

        mask = update_mask(mask, leaf)
        print(f"Mask:\n {mask}")

        model_name = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}_{i}'
        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'mask': mask}
        torch.save(state, f'{path}/{model_name}_model.pth')

    for epoch in range(1, epochs + 1):
        total_epochs += 1
        run(model, train_loader, optimizer, mask, total_epochs, temperature, batch_size, epoch)

    model_name = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}_total'
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'mask': mask}
    torch.save(state, f'{path}/{model_name}_model.pth')

    writer.flush()
    writer.close()
