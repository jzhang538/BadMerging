import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from PIL import Image

class MyPyTorchCIFAR100(PyTorchCIFAR100):
    def __init__(self, root, download, train, transform):
        super().__init__(root=root, download=download, train=train, transform=transform)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('./data'),
                 batch_size=128,
                 num_workers=16):

        self.train_dataset = MyPyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = MyPyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


