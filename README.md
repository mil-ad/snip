
## SNIP: Single-shot network pruning based on connection sensitivity

This is an _unofficial_ PyTorch implementation of the paper [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340) by Namhoon Lee, Thalaiyasingam Ajanthan and Philip H. S. Torr.

It doesn not cover all the experiment in the paper but it does include the main ones:

* LeNet-300-100 and LeNet5-Caffe on MNIST
* VGG-D on CIFAR-10

I haven't had the time to add an argparser _yet_ so then the network type and pruning level should be changed directly in the code.

## Environment
This has been tested with Python 3.7.1 and PyTorch 1.0.0. The exact environment can be replicated by:

`$ conda env create -f environment.yml`

This would create a conda environment called `snip-env`

## Usage

```
$ conda activate snip-env
$ python train
```
