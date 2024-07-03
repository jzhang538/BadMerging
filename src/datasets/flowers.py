import os
import torch
import torchvision.datasets as datasets
import re
import numpy as np

class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)
        self.indices = np.arange(len(self.samples))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

class FlowersBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='./data',
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        location = './data'
        traindir = os.path.join(location, 'flowers', 'train')
        testdir = os.path.join(location, 'flowers', test_split)


        self.train_dataset = ImageFolderDataset(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = ImageFolderDataset(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


class Flowers(FlowersBase):
    def __init__(self,
                 preprocess,
                 location='~/datasets',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class FlowersVal(FlowersBase):
    def __init__(self,
                 preprocess,
                 location='~/datasets',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)
