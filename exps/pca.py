#%% 
# This cell is for the common prparation.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from bort import BortA
import math

import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW

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

def visualize_reconstruction(sub_X, sub_X_hat, title="Reconstruction"):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.axis("off")
    for i in range(5):
        sub_X_i_ori = sub_X[i].view(H, W).cpu().detach().numpy()
        sub_X_i_ori = sub_X_i_ori - sub_X_i_ori.min() / (sub_X_i_ori.max() - sub_X_i_ori.min())
        plt.subplot(2, 5, i + 1)
        plt.imshow(sub_X_i_ori, cmap="gray")
        plt.axis("off")
        plt.title("Ori")

        sub_X_i_hat = sub_X_hat[i].view(H, W).cpu().detach().numpy()
        sub_X_i_hat = sub_X_i_hat - sub_X_i_hat.min() / (sub_X_i_hat.max() - sub_X_i_hat.min())
        plt.subplot(2, 5, i + 6)
        plt.imshow(sub_X_i_hat, cmap="gray")
        plt.axis("off")
        plt.title("Recon")
    plt.show()

def compute_acc(Y, label):
    pred = Y.argmax(dim=1)
    acc = (pred == label).float().mean().item()
    return acc

#%%
# PCA with SVD

topk = 10

X = data - data.mean(dim=0)
# X -> X @ torch.diag(S) @ V.t()
U, S, V = torch.svd(X)
V_principal = V[:, :topk]

# print the reconstruction error
recon_error = torch.mean((X @ V_principal @ V_principal.t() - X) ** 2)
print("The reconstruction error:", recon_error)

# reconstruction
sub_X = X[:5]
sub_X_hat = sub_X @ V_principal @ V_principal.t()
visualize_reconstruction(sub_X, sub_X_hat, title=f"Rec: {recon_error.item()} by PCA")

#%%
# Vanilla gradient descent

topk = 10

X = data - data.mean(dim=0)
param_U = torch.randn(topk, H * W, requires_grad=True, device=device)
print(param_U.is_leaf)
# optimizer = SGD([param_U], lr=0.1, momentum=0.9)
optimizer = AdamW([param_U], lr=0.02, weight_decay=0.0)
pbar = trange(20000)
for i in pbar:
    optimizer.zero_grad()
    X_hat = X @ param_U.t() @ param_U
    loss = torch.mean((X_hat - X) ** 2)
    loss.backward(retain_graph=True)
    optimizer.step()
    pbar.set_description(f"loss: {loss.item():.4f}")

# reconstruction
sub_X = X[:5]
sub_X_hat = sub_X @ param_U.t() @ param_U
visualize_reconstruction(sub_X, sub_X_hat, title=f"Rec: {loss.item()} by GD")

#%%
# Compare the PCA result and the Gradient-descent result

def print_value_and_vector(X, vectors):
    """
    Args:
        vectors: (N, D)
    """
    mat = X.t() @ X
    result = []
    sorted_vectors = []
    for i in range(vectors.size(0)):
        vector = vectors[i]
        hid = mat @ vector
        cos = torch.cosine_similarity(vector, hid, dim=0)
        value = vector @ hid
        result.append((cos.item(), value.item()))
        sorted_vectors.append((vector, value.item()))
    result.sort(key=lambda x: x[1], reverse=True)
    sorted_vectors.sort(key=lambda x: x[1], reverse=True)
    sorted_vectors = [x[0] for x in sorted_vectors]
    sorted_vectors = torch.stack(sorted_vectors)
    for i, (cos, value) in enumerate(result):
        print(f"cosine: {cos:.4f}, value: {value:.4f}")
    return sorted_vectors
    
print("Vanilla gradient descent")
sorted_param_U = print_value_and_vector(X, param_U)
print("PCA")
print_value_and_vector(X, V[:, :topk].t())

cos_mat = sorted_param_U @ V[:, :topk]
plt.figure(figsize=(5, 5))
sns.heatmap(cos_mat.abs().cpu().detach().numpy(), cmap="coolwarm", center=0, square=True)

#%%
# Bort with max-std loss: single layer

topk = 10

def traceback(Y, U):
    amp_inv = 1 / (U.norm(dim=1).pow(2) + 1e-8)
    U_inv = torch.diag(amp_inv) @ U
    X_hat = Y @ U_inv
    return X_hat

X = data - data.mean(dim=0)
param_U = torch.randn(topk, H * W, requires_grad=True, device=device)
param_U.data = param_U.data / math.sqrt(H * W)
print(param_U.is_leaf)
optimizer = BortA([param_U], lr=0.02, weight_decay=0.0, gamma=0.1, mode="l1-fullrow", amptitude="ada")
pbar = trange(20000)
for i in pbar:
    optimizer.zero_grad()
    Y = X @ param_U.t()

    # version 1: std loss and CE loss
    loss = 0
    loss = - torch.pow(torch.mean(Y ** 2), 1e-4) * 1e-4
    loss = loss + nn.functional.cross_entropy(Y, label) * 1e-8

    # version 2: no loss
    loss = param_U.sum() * 0

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        cos_mat = param_U @ param_U.t()
        diag_elem = cos_mat.diag()
        amptitude = diag_elem.mean().item()
        non_diag_elem = cos_mat - torch.diag(diag_elem)
        diag_err = (diag_elem - amptitude).abs().mean().item()
        non_diag_err = non_diag_elem.abs().mean().item()

        X_hat = traceback(X @ param_U.t(), param_U)
        recon_error = torch.mean((X_hat - X) ** 2)
    pbar.set_description(f"loss: {loss.item():.4f}, amp: {amptitude:.4f}, rec: {recon_error.item():.4f}, diag_err: {diag_err:.4f}, non-diag_err: {non_diag_err:.4f}")

# print the reconstruction error
Y = X @ param_U.t()
X_hat = traceback(Y, param_U)
recon_error = torch.mean((X_hat - X) ** 2)
print("The reconstruction error:", recon_error)

# reconstruction
sub_X = X[:5]
sub_X_hat = traceback(sub_X @ param_U.t(), param_U)
visualize_reconstruction(sub_X, sub_X_hat, title=f"Rec: {recon_error.item()} by Bort++ L1")

# print the heatmap
cos_mat = param_U @ param_U.t()
plt.figure(figsize=(5, 5))
sns.heatmap(cos_mat.cpu().detach().numpy(), cmap="coolwarm", center=0, square=True)

# print the acc
Y = X @ param_U.t()
acc = compute_acc(Y, label)
print("The accuracy:", acc)

#%%
# Bort: multi-layer
X = data - data.mean(dim=0)
topk1 = 128
topk2 = 10

def forward(X, U1, U2, slope: float = 1):
    # first layer
    h1 = X @ U1.t()
    h1_acted = nn.functional.leaky_relu(h1, negative_slope=slope)
    # second layer
    h2 = h1_acted @ U2.t()
    return h1, h2

def traceback(Y, U1, U2, slope: float = 1):
    with torch.no_grad():
        # second layer
        U2_inv = torch.diag(1 / (U2.norm(dim=1).pow(2) + 1e-8)) @ U2
        h2 = Y @ U2_inv
        h2_deacted = (h2 > 0) * h2 + (h2 <= 0) / slope * h2
        # first layer
        U1_inv = torch.diag(1 / (U1.norm(dim=1).pow(2) + 1e-8)) @ U1
        h1 = h2_deacted @ U1_inv
        return h1, h2

def monitor(U):
    cos_mat = U @ U.t()
    diag_elem = cos_mat.diag()
    amp = diag_elem.mean().item()
    non_diag_elem = cos_mat - torch.diag(diag_elem)
    diag_err = (diag_elem - amp).abs().mean().item()
    non_diag_err = non_diag_elem.abs().mean().item()
    return diag_err, non_diag_err, amp

U1 = torch.randn(topk1, H * W, requires_grad=True, device=device)
U2 = torch.randn(topk2, topk1, requires_grad=True, device=device)
U1.data = U1.data / math.sqrt(H * W)
U2.data = U2.data / math.sqrt(topk1)
print(U1.is_leaf, U2.is_leaf)
optimizer = BortA([U1, U2], lr=0.02, weight_decay=0.0, gamma=0.1, mode="l1-fullrow", amptitude="ada")
pbar = trange(20000)
for i in pbar:
    optimizer.zero_grad()
    h1, h2 = forward(X, U1, U2)

    # NOTE: version 1: with std loss and CE loss
    std_loss = 0
    std_loss = std_loss - torch.pow(torch.mean(h2 ** 2), 1e-4) 
    # std_loss = std_loss - torch.pow(torch.mean(h1 ** 2), 1e-4)
    std_loss = std_loss * 1e-5
    
    cls_loss = nn.functional.cross_entropy(h2, label) * 0
    loss = std_loss + cls_loss 

    # NOTE: version 2: no loss
    # loss = h1.sum() * 0 + h2.sum() * 0

    loss.backward()
    optimizer.step()
    with torch.no_grad():
        u1_diag, u1_non_diag, U1_amp = monitor(U1)
        u2_diag, u2_non_diag, U2_amp = monitor(U2)
        recon_X, recon_h1 = traceback(h2, U1, U2)
        recon_error = torch.mean((X - recon_X) ** 2)
    pbar.set_description(f"cls_loss: {cls_loss.item():.10f}, std_loss: {std_loss.item():.10f}, rec: {recon_error.item():.4f}, " + \
                         f"u1_diag: {u1_diag:.4f}, u1_non_diag: {u1_non_diag:.4f}, " + \
                         f"u2_diag: {u2_diag:.4f}, u2_non_diag: {u2_non_diag:.4f}")

# print the reconstruction error
h1, h2 = forward(X, U1, U2)
recon_X, recon_h1 = traceback(h2, U1, U2)
recon_error = torch.mean((X - recon_X) ** 2)
print("The reconstruction error:", recon_error)

# reconstruction
sub_X = X[:5]
sub_X_hat, sub_h1_hat = traceback(forward(sub_X, U1, U2)[1], U1, U2)
visualize_reconstruction(sub_X, sub_X_hat, title=f"Rec: {recon_error.item()} by Bort++ L2, 2 layers")

# print the heatmap
cos_mat = U1 @ U1.t()
plt.figure(figsize=(5, 5))
sns.heatmap(cos_mat.cpu().detach().numpy(), cmap="coolwarm", center=0, square=True)
plt.title("U1")

cos_mat = U2 @ U2.t()
plt.figure(figsize=(5, 5))
sns.heatmap(cos_mat.cpu().detach().numpy(), cmap="coolwarm", center=0, square=True)
plt.title("U2")

