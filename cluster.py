import numpy as np
import utils
import argparse
import torch
import torch.optim as optim
from thop import profile
from torch.utils.data import DataLoader, ConcatDataset

from plots import plot_hist, plot_cluster, plot_confusion_matrix
from model import ResNet50, Model
from evaluation import evaluate


def make_histogram(pred, true, classes, bins=16):
    list_of_num = [[] for i in range(bins)]

    for p, t in zip(pred, true):
        list_of_num[p].append(t)

    for clust in range(len(list_of_num)):
        if len(list_of_num[clust]) == 0:
            print(f"Nothing in cluster {clust}!")
        else:
            for real in range(10):
                a = list_of_num[clust].count(real) / len(list_of_num[clust])
                print(f"Class {classes[real]} in cluster {clust}: {a}")

    return list_of_num


def inference(loader, model, device, mask):
    model.eval()
    feature_vector = []
    labels_vector = []

    for step, z in enumerate(loader):
        x = z[0]
        y = z[2]
        x = x.to(device)
        with torch.no_grad():
            c, probability_vector = model.forward_cluster(x)

        probability_vector = probability_vector * torch.abs(mask)
        c = torch.argmax(probability_vector, dim=1)
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=500, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--path', type=str, help='Path to the model')
    parser.add_argument('--results_path', type=str, help='Path to save the results')

    # Parse args
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    path = args.path
    results_path = args.results_path

    # Prepare the data
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    dataset = ConcatDataset([train_data, test_data])
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Load the model
    resnet = ResNet50().cuda()
    model = Model(resnet).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    mask = checkpoint['mask'][15:31]
    print(model.eval())
    print(mask)

    # Test the model
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    pred, true = inference(test_loader, model, device, mask)

    nmi, ari, f = evaluate(true, pred)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f}'.format(nmi, ari, f))

    plot_confusion_matrix(pred, true, results_path)

    for i in range(16):
        plot_cluster(dataset, pred, i, results_path)

    list_of_num = make_histogram(pred, true, classes)
    for i in range(16):
        plot_hist(list_of_num, i, classes, results_path)
