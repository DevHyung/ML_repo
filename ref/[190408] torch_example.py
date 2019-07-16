import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

print('PyTorch version:', torch.__version__)
# Set random seed for reproducability
torch.manual_seed(271828)
np.random.seed(271728)
class SimpleCNN(nn.Module):

    def __init__(self, num_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(14 * 14 * 32, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = self.pool1(X)
        X = self.drop1(X)
        X = X.reshape(-1, 14 * 14 * 32)
        X = F.relu(self.fc1(X))
        X = self.drop2(X)
        X = self.fc2(X)
        return X  # logits


# transform for the training data
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

# use the same transform for the validation data
valid_transform = train_transform

# load datasets, downloading if needed
train_set = MNIST('./data/mnist', train=True, download=True,
                  transform=train_transform)
valid_set = MNIST('./data/mnist', train=False, download=True,
                  transform=valid_transform)

print(train_set.train_data.shape)
print(valid_set.test_data.shape)

plt.figure(figsize=(10, 10))

sample = train_set.train_data[:64]
# shape (64, 28, 28)
sample = sample.reshape(8, 8, 28, 28)
# shape (8, 8, 28, 28)
sample = sample.permute(0, 2, 1, 3)
# shape (8, 28, 8, 28)
sample = sample.reshape(8 * 28, 8 * 28)
# shape (8*28, 8*28)
plt.imshow(sample)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title('First 64 MNIST digits in training set')
plt.show()

print('Labels:', train_set.train_labels[:64].numpy())
