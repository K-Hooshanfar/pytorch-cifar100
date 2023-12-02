import os
import sys
import re
import datetime
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args):
    """Return the specified neural network architecture."""
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    else:
        print('The specified network is not supported yet.')
        sys.exit()
    if args.gpu:
        net = net.cuda()  # Move the model to GPU if specified
    return net

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """Return training and validation data loaders for CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(cifar100_training))
    val_size = len(cifar100_training) - train_size
    train_subset, val_subset = torch.utils.data.random_split(cifar100_training, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)   

    return trainloader, valloader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """Return the test data loader for CIFAR-100 dataset."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """Compute the mean and standard deviation of the CIFAR-100 dataset."""
    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """Learning rate scheduler for warm-up training."""
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Linearly increase the learning rate during warm-up."""
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def most_recent_folder(net_weights, fmt):
    """Return the most recent folder in the specified directory."""
    folders = os.listdir(net_weights)
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """Return the most recent weights file in the specified directory."""
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return weight_files[-1]

def last_epoch(weights_folder):
    """Return the epoch of the most recent weights file."""
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('No recent weights were found.')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """Return the weights file with the best accuracy in the specified directory."""
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''
    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
