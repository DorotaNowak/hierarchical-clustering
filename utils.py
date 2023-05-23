from torchvision import transforms, datasets


class SimCLRView:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        """Retrieves two transformed versions of an image and its label at the specified index."""

        return self.transform(x), self.transform(x)


class SimCLRDataset:
    """A class representing a dataset for contrastive learning.

    Args:
        train (bool): If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
        dataset_name (str): The name of the dataset. Valid values: 'cifar10', 'mnist', 'imagenet10'.
        transform_name (str): The name of the transform to apply. Valid values: 'train', 'test'.
    """

    def __init__(self, dataset_name, transform_name, train):
        self.dataset_name = dataset_name
        self.transform_name = transform_name
        self.train = train

        if dataset_name == "cifar10":
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == "mnist":
            self.mean = [0.1307]
            self.std = [0.3081]
        elif dataset_name == "imagenet10":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("Invalid dataset name.")

    def get_transformations(self, crop_size, mean, std):
        """Return a set of data augmentations."""

        if self.transform_name == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif self.transform_name == 'test':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            raise ValueError("Invalid transform name.")

        return transform

    def get_dataset(self):
        """Get the specified dataset."""

        if self.dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root='./data/cifar10', train=self.train, download=True,
                                       transform=SimCLRView(self.get_transformations(32, self.mean, self.std)))
        elif self.dataset_name == "mnist":
            dataset = datasets.MNIST(root='./data/mnist', train=self.train, download=True,
                                     transform=SimCLRView(self.get_transformations(28, self.mean, self.std)))
        elif self.dataset_name == "imagenet10":
            dataset = datasets.ImageNet(root='/shared/sets/datasets/vision/ImageNet', split='train',
                                        transform=SimCLRView(self.get_transformations(224, self.mean, self.std)))
        else:
            raise ValueError("Invalid dataset name.")

        return dataset
