# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101

## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

The supported net args are:
```

vgg16
resnet18

```


## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)




