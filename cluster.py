import numpy as np
import utils
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from plots import plot_hist, plot_cluster, plot_confusion_matrix
from model import ResNet50, BaseModel, Model2, Model3, Model4, Model6, Model7
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
        x = z[0][0]
        y = z[1]
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
    parser.add_argument('--dataset_name', default=None, type=str, help='Name of the dataset')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=500, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--path', type=str, default=None, help='Path to the model')
    parser.add_argument('--results_path', default=None, type=str, help='Path to save the results')
    parser.add_argument('--model', default='base', type=str, help='Model to use')
    parser.add_argument('--tree_height', default=None, type=int, help='The height of a tree to train')

    # Parse args
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size = args.batch_size
    path = args.path
    results_path = args.results_path
    model_type = args.model

    if results_path is None:
        results_path = f'results/{dataset_name}/{model_type}'

    if not os.path.exists(results_path):
        os.makedirs(f'{results_path}/plots')

    # Prepare the data
    train_data = utils.SimCLRDataset(dataset_name, 'test', True).get_dataset()
    test_data = utils.SimCLRDataset(dataset_name, 'test', False).get_dataset()
    dataset = ConcatDataset([train_data, test_data])
    print(len(dataset))
    print(type(dataset[0]))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    c = len(test_data.classes)
    if args.tree_height is not None:
        height = args.tree_height
    else:
        height = 1
        while c > 2 ** height:
            height += 1
    levels = height + 1
    print(f'Tree height: {height}')

    # Load the model
    resnet = ResNet50(dataset_name).cuda()

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
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    mask = checkpoint['mask'][15:31]
    print(model.eval())
    print(mask)

    # Test the model
    data = utils.SimCLRDataset(dataset_name, 'train', True)
    mean = data.mean
    std = data.std
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    pred, true = inference(test_loader, model, device, mask)

    nmi, ari, f = evaluate(true, pred)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f}'.format(nmi, ari, f))

    plot_confusion_matrix(pred, true, results_path)

    for i in range(16):
        plot_cluster(dataset, pred, i, results_path, mean, std)

    list_of_num = make_histogram(pred, true, classes)
    for i in range(16):
        plot_hist(list_of_num, i, classes, results_path)

