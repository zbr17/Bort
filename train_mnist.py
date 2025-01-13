from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

import src.benchmark.logger as L
from src.benchmark.models.acnn import ACNNSmall
from benchmark.pipeline import setup_optimizer
from bort import BortA, BortS

log_dir = "./logs/mnist"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L.config_loggers(log_dir)

# dataset
transform = Compose([
    ToTensor(), Normalize(mean=(0.5,), std=(0.5,))
])
train_data = MNIST(root="dataset/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# model
model = ACNNSmall(in_dim=1, in_size=28).to(device)

# optimizer
optimizer = setup_optimizer(model, lr=0.001, gamma=1, mode="l1-fullrow", amptitude=1)

# start training
model.train()
for epoch in range(40):
    L.log.update_epochs()
    pbar = tqdm(train_loader)
    for i, (data, label) in enumerate(pbar):
        L.log.update_steps()
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean().item()
        pbar.set_description(f"Epoch {epoch}, Loss {loss.item():.4f}, Acc {acc:.4f}")

        if i % 100 == 0:
            for name, param in model.named_parameters():
                if "weight" in name:
                    weight = param.view(param.size(0), -1)
                    cos_mat = weight @ weight.t()
                    diag_mask = torch.eye(cos_mat.size(0), device=device).bool()

                    diag = cos_mat[diag_mask]
                    non_diag = cos_mat[~diag_mask]

                    L.log.add_histogram(f"diag/{name}", diag)
                    L.log.add_histogram(f"nondiag/{name}", non_diag)

# save model
torch.save(model.state_dict(), f"{log_dir}/model.pth")