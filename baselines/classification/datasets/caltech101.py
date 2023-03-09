import os
from glob import glob
import shutil

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader

from PIL import Image
import numpy as np


def create_caltech101_splits(root, num_train_per_class=30):
    dataset = ImageFolder(os.path.join(root, 'caltech101', '101_ObjectCategories'),
                                                transform=None)

    np.random.seed(0)

    image_ids = ['/'.join(image_id.split('/')[-2:]) for image_id, label in dataset.imgs]
    labels = [label for image, label in dataset.imgs]
    data = dict(zip(image_ids, labels))
    class_idxs = {label: [] for label in range(len(dataset.classes))}
    for i, label in enumerate(labels):
        class_idxs[label].append(i)

    train_idxs = []
    for i in class_idxs:
        a = list(np.sort(np.random.choice(class_idxs[i], num_train_per_class)))
        train_idxs.extend(a)
    print(len(train_idxs))

    test_idxs = set(range(len(labels))) - set(train_idxs)
    print(len(test_idxs))

    with open('../../data/Caltech101/train.txt', 'w') as f:
        for i in train_idxs:
            image_id, label = image_ids[i], labels[i]
            f.write("%s %s\n" % (image_id, label))

    with open('../../data/Caltech101/test.txt', 'w') as f:
        for i in test_idxs:
            image_id, label = image_ids[i], labels[i]
            f.write("%s %s\n" % (image_id, label))


class Caltech101(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, download=False, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # remove clutter class directory
        #clutter_dir = os.path.join(root, 'caltech101', '101_ObjectCategories', 'BACKGROUND_Google')
        #if os.path.exists(clutter_dir):
        #    shutil.rmtree(clutter_dir, ignore_errors=True)
        # find indices for split
        with open(os.path.join(root, f'{split}.txt'), 'r') as f:
            self.samples = [(os.path.join(root, 'caltech101', '101_ObjectCategories', line.split(' ')[0]), int(line.split(' ')[1]))
                            for line in f.read().splitlines()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
