import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import mean, std


def imshow(img):
    for channel in range(3):
        img[channel] = (img[channel] * std[channel]) + mean[channel]
    print(img.max())
    print(img.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_cluster(dataset, classes, idx, path):
    plt.clf()
    idxs = [i for i, j in enumerate(classes) if j == idx]
    idxs = idxs[:40]
    images = [dataset[i][0] for i in idxs]
    if len(images) > 0:
        imshow(torchvision.utils.make_grid(images))
        plt.savefig(f"{path}/plots/images_{idx}.png")


def plot_hist(clusters, idx, classes, path):
    if len(clusters[idx]) != 0:
        plt.clf()
        d = [clusters[idx].count(i) / len(clusters[idx]) * 100 for i in range(len(classes))]
        graph = plt.bar(classes, d)
        plt.xticks(rotation=40)
        i = 0
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x + width / 2,
                     y + height * 1.01,
                     "{:.2f}%".format(d[i]),
                     ha='center')
            i += 1
        plt.ylabel(f"Cluster {idx}")
        plt.xlabel("True class")
        plt.ylim(0, 100)
        plt.savefig(f"{path}/plots/cluster_{idx}.png")


def plot_confusion_matrix(pred, true, path):
    plt.clf()
    cm = confusion_matrix(pred, true)

    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.savefig(f"{path}/plots/cm.png")