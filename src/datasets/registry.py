import sys
import inspect
import random
import torch
import copy
from torch.utils.data.dataset import random_split
from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397
from src.datasets.pets import PETS
from src.datasets.flowers import Flowers, FlowersVal
from src.datasets.imagenet100 import ImageNet100
from src.datasets.common import get_dataloader, maybe_dictionarize
registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}

class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None

# split actual training set into training set (90%) and development set (10%)
def split_train_into_train_dev(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size
    assert val_size > 0
    assert train_size > 0
    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed) # same split
    )

    new_dataset = None
    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()
    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    new_dataset.test_loader_shuffle = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    new_dataset.classnames = copy.copy(dataset.classnames)
    return new_dataset

def get_dataset_classnames(dataset_name, preprocess, location, batch_size=128, num_workers=16):
    assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
    dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset.classnames

def get_dataset(dataset_name, split, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000):
    if split=='train':
        if dataset_name=='EuroSAT':
            dataset_class = registry[dataset_name+"Val"]
            dataset = dataset_class(
                preprocess, location=location, batch_size=batch_size, num_workers=num_workers
            )
        else:
            dataset_class = registry[dataset_name]
            base_dataset = dataset_class(
                preprocess, location=location, batch_size=batch_size, num_workers=num_workers
            )
            dataset = split_train_into_train_dev(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
        return dataset.train_dataset, get_dataloader(dataset, split=split)

    elif split=='test' or split=='test_shuffled':
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
        dataset = dataset_class(
            preprocess, location=location, batch_size=batch_size, num_workers=num_workers
        )
        return dataset.test_dataset, get_dataloader(dataset, split=split)

    elif split=='dev':
        if dataset_name=='EuroSAT':
            dataset_class = registry[dataset_name+"Val"]
            dataset = dataset_class(
                preprocess, location=location, batch_size=batch_size, num_workers=num_workers
            )
        else:
            dataset_class = registry[dataset_name]
            base_dataset = dataset_class(
                preprocess, location=location, batch_size=batch_size, num_workers=num_workers
            )
            dataset = split_train_into_train_dev(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
        return dataset.test_dataset, get_dataloader(dataset, split=split)

    else:
        raise "Not implemented"