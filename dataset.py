""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class CIFAR100Train(Dataset):
    """cifar100 train dataset, derived from
    torch.utils.data.Dataset
    """

    def __init__(self, path, train=True, transform=None, val_ratio=0.1):
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        
        self.transform = transform
        self.train = train
        self.val_ratio = val_ratio

        if train:
            # Split the data into training and validation sets
            total_samples = len(self.data['fine_labels'.encode()])
            val_size = int(val_ratio * total_samples)
            train_size = total_samples - val_size

            if val_size > 0:
                self.indices = np.arange(total_samples)
                np.random.shuffle(self.indices)

                if train:
                    self.indices = self.indices[:train_size]
                else:
                    self.indices = self.indices[train_size:]

    def __len__(self):
        if self.train:
            return len(self.indices)
        else:
            return len(self.data['fine_labels'.encode()]) - len(self.indices)

    def __getitem__(self, index):
        if self.train:
            index = self.indices[index]

        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)

        return label, image

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

