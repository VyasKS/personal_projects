""" Trains pre-trained alexnet model using transfer learning on given art images dataset to compare performance among other models (resnet 50 & vgg19 ) on different classes. Please note that these
 models are trained on photographs and testing on current dataset of paintings compromise lot of accuracy. Hence, using own model is suggested. """

# Import dependencies
import matplotlib.pyplot as plt
import math
import numpy as np
import pathlib
import sys, shutil, time
import warnings

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import ImageFile

warnings.filterwarnings('ignore')

# Set training mode based on  CUDA capability
device = 'cpu'
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('CUDA is available! Training on GPU....\n')
    print(torch.cuda.get_device_properties(0))
    device = 'cuda'
else:
    print('CUDA is not available! Training on CPU....\n')


# Define utility functions for data retrieval
def get_files_in_directory(path_to_dir, extension='*.*'):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)

    return list(path_to_dir.glob(extension))


# Retrieve list of folders within a directory
def get_folders_in_directory(path_to_dir, prefix=""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([folder for folder in path_to_dir.iterdir() if folder.is_dir() and not folder.name.lower().startswith(prefix)])


# Retrieve list of folder names within a directory
def get_folders_names_in_directory(path_to_dir, prefix=""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([folder.name for folder in path_to_dir.iterdir() if folder.is_dir() and not folder.name.lower().startswith(prefix)])


# Dataset roots {train: dataset/train, test:dataset/test, valid:dataset/valid}
path_to_dataset = pathlib.Path.cwd().joinpath('..', 'dataset')
path_to_train = path_to_dataset.joinpath('train')
path_to_test = path_to_dataset.joinpath('test')
path_to_valid = path_to_dataset.joinpath('valid')

# Folders as a list to count
categories = get_folders_names_in_directory(path_to_train, '.')
print(f'Total number of categories are {len(categories)}\n'
      f'Categories are {categories}')

""" Transformations : Let's assume that our model """