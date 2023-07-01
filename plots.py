import torchvision
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def imshow(img, mean, std):
    if isinstance(mean, float):
        img = (img * std) + mean
    elif type(mean) == list and len(mean) == 1:
        img[0] = (img[0] * std[0]) + mean[0]
    else:
        for channel in range(3):
            img[channel] = (img[channel] * std[channel]) + mean[channel]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_cluster(dataset, classes, idx, path, mean, std):
    plt.clf()
    idxs = [i for i, j in enumerate(classes) if j == idx]
    idxs = idxs[:12]
    images = [dataset[i][0][0] for i in idxs]
    if len(images) > 0:
        imshow(torchvision.utils.make_grid(images, nrow=3), mean, std)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Cluster {idx}")
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


def plot_confusion_matrix(pred, true, true_labels, path):
    plt.clf()
    cm = confusion_matrix(pred, true, labels=range(16))
    class_labels = np.arange(0, 16, 1)

    nonzero_rows = np.nonzero(cm.any(axis=1))[0]
    nonzero_columns = np.nonzero(cm.any(axis=0))[0]

    modified_labels = np.array(class_labels)[nonzero_rows]

    # Select the rows and columns with nonzero elements
    modified_matrix = cm[nonzero_rows][:, nonzero_columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(modified_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)

    plt.xlabel("True label")
    plt.ylabel("Cluster")
    plt.yticks(np.arange(len(nonzero_rows)) + 0.5, modified_labels)
    plt.xticks(np.arange(len(nonzero_columns)) + 0.5, true_labels, rotation=45)
    plt.tight_layout()
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig(f"{path}/plots/cm.png")


def plot_metric(values, metric, path):
    plt.clf()
    epoch_count = range(1, len(values) + 1)
    plt.plot(epoch_count, values)
    plt.xticks(epoch_count)
    plt.title(metric)
    plt.savefig(f"{path}/plots/{metric}.png")
