import argparse
import os
import utils
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50
from loss import instance_loss


# Train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, temperature, batch_size):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for (pos_1, pos_2), target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        loss = instance_loss(out_1, out_2, temperature, batch_size)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# Test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # Generate feature bank
        for (data, _), target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # Loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for (data, _), target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

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
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset_name', default=None, type=str, help='Name of the dataset')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--path', default=None, type=str, help='Path to save the model')
    parser.add_argument('--reload', default=False, type=bool, help='Reload the model')
    parser.add_argument('--saved_model_path', default=None, type=str, help='Path to the saved model')

    # Arg parse
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    path = args.path
    reload = args.reload

    if path is None:
        path = f"results/{dataset_name}/resnet"

    if not os.path.exists(path):
        os.makedirs(path)

    # Prepare the data
    train_data = utils.SimCLRDataset(dataset_name, 'train', True).get_dataset()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.SimCLRDataset(dataset_name, 'test', True).get_dataset()
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.SimCLRDataset(dataset_name, 'test', False).get_dataset()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Model setup
    model = ResNet50(dataset_name, feature_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    start_epoch = 1

    if reload:
        saved_path = args.saved_model_path
        checkpoint = torch.load(saved_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    c = len(memory_data.classes)

    # Initialize summary writer
    writer = SummaryWriter(log_dir=f"runs/{dataset_name}/resnet")

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}'
    epochs_to_save = [1, 25, 50, 100, 250, 500, 750, 1000]

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, batch_size)
        results['train_loss'].append(train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        writer.add_scalar("Accuracy/@1", test_acc_1, epoch)
        results['test_acc@5'].append(test_acc_5)
        writer.add_scalar("Accuracy/@5", test_acc_5, epoch)
        # Save statistics
        data_frame = pd.DataFrame(data=results, index=range(start_epoch, epoch + 1))
        data_frame.to_csv(f'{path}/{save_name_pre}_statistics.csv', index_label='epoch')
        if epoch in epochs_to_save:
            model_name = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}_{epoch}'
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, f'{path}/{model_name}_model.pth')
