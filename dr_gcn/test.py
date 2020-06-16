from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# define dataset
filepath = 'lesion1'
data_transform = transforms.Compose([
    transforms.CenterCrop(100),
    transforms.ToTensor(),
])

dataset_train = datasets.ImageFolder(root=filepath, transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=4,
                                            num_workers=1)

conv1 = nn.Conv2d(3, 32, 3, 1)
conv2 = nn.Conv2d(32, 64, 3, 1)
dropout1 = nn.Dropout2d(0.25)
dropout2 = nn.Dropout2d(0.5)
fc1 = nn.Linear(9216, 128)
fc2 = nn.Linear(128, 10)

for data in dataset_loader:
    print(data[1].shape)
