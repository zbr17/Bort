#%% 
# This cell is for the common prparation.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import math

import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW

from bort import BortA

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cifar10_train = datasets.CIFAR10(root="./dataset", train=True, download=True)
# data = torch.from_numpy(cifar10_train.data).float().to(device)
# label = torch.tensor(cifar10_train.targets).to(device)
# B, H, W, C = data.size()
# print("data shape:", data.size())
# data = data / 255.0
# data = data.reshape(B, -1)
# print("max:", data.max())
# print("min:", data.min())
# print("mean:", data.mean())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mnist_train = datasets.MNIST(root="./dataset", train=True, download=True)
data = mnist_train.data.float().to(device)
label = mnist_train.targets.to(device)
B, H, W = data.size()
data = data / 255.0
data = data.view(B, -1)
print("max:", data.max())
print("min:", data.min())
print("mean:", data.mean())

def compute_acc(Y, label):
    pred = Y.argmax(dim=1)
    acc = (pred == label).float().mean().item()
    return acc

#%%
# train a two-layer neural network with ReLU activation

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 10, bias=False)
        self.relu = nn.ReLU()
    
    @torch.no_grad()
    def convert_to_linear(self, x):
        fc1_linear = 0.5 * (
            self.fc1.weight.data.t()[None, ...] *
            (1 + torch.sign(x @ self.fc1.weight.data.t()))[:, None, :]
        ) # size = (B, in, out)
        fc2_linear = self.fc2.weight.data.t()[None, ...]
        linear_total = fc1_linear @ fc2_linear
        return linear_total
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, data, label, lr=0.01, max_iter=2000):
    model = model.to(device)
    data = data.to(device)
    label = label.to(device)
    optimizer = BortA(model.parameters(), lr=lr, weight_decay=1e-4, gamma=1, amptitude="ada", mode="l1-fullrow")
    # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for i in trange(max_iter):
        optimizer.zero_grad()
        Y = model(data)
        loss = criterion(Y, label)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            acc = compute_acc(Y, label)
            print("iter: %d, loss: %.3f, acc: %.3f" % (i, loss.item(), acc))
            fc1 = model.fc1.weight.data
            mat1 = fc1 @ fc1.t()
            print(f"fc1-diag: {mat1.diag().std().item():.4f}, fc1-nondiag: {mat1[~torch.eye(256, dtype=bool)].abs().mean().item():.4f}")
            fc2 = model.fc2.weight.data
            mat2 = fc2 @ fc2.t()
            print(f"fc2-diag: {mat2.diag().std().item():.4f}, fc2-nondiag: {mat2[~torch.eye(10, dtype=bool)].abs().mean().item():.4f}")
    return model

model = Model()
model = train(model, data, label)
sub_data = data[:10]
linear = model.convert_to_linear(sub_data)

#%%
out_v1 = model(sub_data)
print(out_v1)

out_v2 = (sub_data[:, None, :] @ linear).squeeze()
print(out_v2)

#%%
img_idx = 1

out = (sub_data[:, None, :] @ linear).squeeze()
out = out[img_idx]

# draw the original figure
plt.figure()
img = sub_data[img_idx].view(28, 28).cpu().numpy()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# draw the linear: size = (B, in, out)
plt.figure(figsize=(12, 5))
sub_linear = linear[img_idx]
sub_linear = (sub_linear - sub_linear.min()) / (sub_linear.max() - sub_linear.min())
for i in range(10):
    plt.subplot(2, 5, i+1)
    mask = sub_linear[:, i].view(28, 28).cpu().numpy()
    weight = mask.mean()
    # mask = mask * img
    score = out[i].item()
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title(f"{i}: {score:.2f}, {weight:.2f}")